"""Automatic Google Calendar sync for reminders (/api/calendar/...).

Connecting runs Google's OAuth flow for installed apps (loopback redirect + PKCE): the browser goes to Google,
the user allows access, and Google sends the browser back to this server on 127.0.0.1 (through an SSH tunnel
too), which stores a refresh token next to the workspace's reminders store (gitignored, owner-only file).

Sync keeps a calendar of its own, "NASH Think", in step with the open upcoming reminders: new ones are added,
changed ones updated, done / dismissed / passed ones removed. The user's other calendars are never read or
touched: the scope is calendar.app.created, which only reaches calendars this app created. What leaves the
machine is each reminder's title, time and a short line from the message it came from.

Needs GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET (an OAuth client of type "Desktop app") in the environment
or the repo's .env file. Every HTTP call goes through one httpx client, so tests can swap in a fake transport.
"""
import base64
import datetime as dt
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
from pathlib import Path

import httpx

import llm_config

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
API = "https://www.googleapis.com/calendar/v3"
SCOPE = "https://www.googleapis.com/auth/calendar.app.created"
CALENDAR_NAME = "NASH Think"
CALLBACK_PATH = "/api/calendar/google/callback"
STATE_TTL_S = 600
HORIZON_DAYS = 400           # reminders further ahead than this are not put in the calendar yet
MAX_WRITES = 150             # calendar writes per sync run (Google rate limits); the rest wait for the next run
TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class CalendarError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def client_config(secrets_path=None):
    """(client_id, client_secret) from the environment or .env; None when unset."""
    cid = llm_config.api_key({"api_key_env": "GOOGLE_CLIENT_ID"}, secrets_path)
    secret = llm_config.api_key({"api_key_env": "GOOGLE_CLIENT_SECRET"}, secrets_path)
    return (cid, secret) if cid and secret else None


def _pkce():
    verifier = secrets.token_urlsafe(64)[:96]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


class GoogleCalendarSync:
    """Google Calendar connection and sync for one workspace's RemindersService."""

    def __init__(self, reminders, token_path, config=None, transport=None, clock=time.time):
        self.reminders = reminders
        self.token_path = Path(token_path)
        self._config = config           # (client_id, secret) for tests; else read from .env on use
        self.transport = transport
        self.clock = clock
        self._lock = threading.Lock()
        self._pending = {}              # state -> (verifier, redirect_uri, created)
        self.last_sync = None
        self.last_error = None
        self.last_result = None

    # ── configuration and token ──
    def config(self):
        return self._config or client_config()

    def configured(self):
        return self.config() is not None

    def _read_token(self):
        try:
            return json.loads(self.token_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_token(self, data):
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.token_path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.token_path)

    def connected(self):
        t = self._read_token()
        return bool(t and t.get("refresh_token"))

    def status(self):
        t = self._read_token() or {}
        return {"configured": self.configured(), "connected": bool(t.get("refresh_token")),
                "calendar": CALENDAR_NAME if t.get("calendar_id") else None, "account_connected_at": t.get("connected_at"),
                "last_sync": self.last_sync, "last_error": self.last_error, "last_result": self.last_result}

    def _http(self):
        return httpx.Client(timeout=TIMEOUT, transport=self.transport)

    # ── OAuth ──
    def auth_url(self, redirect_uri):
        cfg = self.config()
        if not cfg:
            raise CalendarError("not_configured", "Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to the .env file first.")
        verifier, challenge = _pkce()
        state = secrets.token_urlsafe(24)
        now = self.clock()
        with self._lock:
            self._pending = {s: v for s, v in self._pending.items() if now - v[2] < STATE_TTL_S}
            self._pending[state] = (verifier, redirect_uri, now)
        q = {"client_id": cfg[0], "redirect_uri": redirect_uri, "response_type": "code", "scope": SCOPE,
             "access_type": "offline", "prompt": "consent", "include_granted_scopes": "false", "state": state,
             "code_challenge": challenge, "code_challenge_method": "S256"}
        return f"{AUTH_URL}?{urllib.parse.urlencode(q)}"

    def owns_state(self, state):
        with self._lock:
            return state in self._pending

    def finish(self, state, code):
        """Exchange the code Google sent back for tokens and remember them."""
        with self._lock:
            pending = self._pending.pop(state, None)
        if not pending or self.clock() - pending[2] > STATE_TTL_S:
            raise CalendarError("bad_state", "That sign-in link has expired. Try Connect again.")
        cid, secret = self.config()
        with self._http() as http:
            r = http.post(TOKEN_URL, data={"code": code, "client_id": cid, "client_secret": secret,
                                           "redirect_uri": pending[1], "grant_type": "authorization_code",
                                           "code_verifier": pending[0]})
        if r.status_code != 200:
            raise CalendarError("google_error", f"Google refused the sign-in ({r.status_code}): {_google_message(r)}")
        tok = r.json()
        if not tok.get("refresh_token"):
            raise CalendarError("google_error", "Google sent no refresh token. Remove NASH Think's access in your Google account and connect again.")
        self._write_token({"refresh_token": tok["refresh_token"], "access_token": tok.get("access_token"),
                           "expires_at": int(self.clock()) + int(tok.get("expires_in", 0)) - 60,
                           "connected_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})

    def disconnect(self):
        """Forget the tokens (and revoke them at Google). The NASH Think calendar itself stays in the account."""
        t = self._read_token()
        if t and t.get("refresh_token"):
            try:
                with self._http() as http:
                    http.post(REVOKE_URL, data={"token": t["refresh_token"]})
            except httpx.HTTPError:
                pass
        try:
            self.token_path.unlink()
        except FileNotFoundError:
            pass
        self.reminders.clear_google()
        self.last_sync = self.last_error = self.last_result = None

    def _access_token(self, http, tok):
        if tok.get("access_token") and tok.get("expires_at", 0) > self.clock():
            return tok["access_token"]
        cid, secret = self.config()
        r = http.post(TOKEN_URL, data={"client_id": cid, "client_secret": secret, "refresh_token": tok["refresh_token"],
                                       "grant_type": "refresh_token"})
        if r.status_code in (400, 401):
            raise CalendarError("reconnect", "Google access was removed or expired. Connect Google Calendar again.")
        if r.status_code != 200:
            raise CalendarError("google_error", f"Google token refresh failed ({r.status_code}): {_google_message(r)}")
        body = r.json()
        tok["access_token"] = body["access_token"]
        tok["expires_at"] = int(self.clock()) + int(body.get("expires_in", 3600)) - 60
        self._write_token(tok)
        return tok["access_token"]

    # ── sync ──
    def _calendar_id(self, http, tok, headers):
        cal = tok.get("calendar_id")
        if cal:
            r = http.get(f"{API}/calendars/{urllib.parse.quote(cal, safe='')}", headers=headers)
            if r.status_code == 200:
                return cal
            if r.status_code not in (404, 410):
                raise CalendarError("google_error", f"Google Calendar error ({r.status_code}): {_google_message(r)}")
            self.reminders.clear_google()        # the user deleted the calendar: make a new one, re-add everything
        r = http.post(f"{API}/calendars", headers=headers, json={
            "summary": CALENDAR_NAME, "timeZone": self.reminders.tz_name,
            "description": "Reminders found in your chats by NASH Think. Managed by the app: changes here may be overwritten."})
        if r.status_code not in (200, 201):
            raise CalendarError("google_error", f"Could not create the NASH Think calendar ({r.status_code}): {_google_message(r)}")
        tok["calendar_id"] = r.json()["id"]
        self._write_token(tok)
        return tok["calendar_id"]

    def event_body(self, item):
        tz = self.reminders.tz_name
        start = dt.datetime.fromtimestamp(item["due_ts"], self.reminders.tz)
        if item["all_day"]:
            when = {"start": {"date": start.date().isoformat()},
                    "end": {"date": (start.date() + dt.timedelta(days=1)).isoformat()}}
            alarm = 15 * 60              # 9:00 the day before
        else:
            when = {"start": {"dateTime": start.isoformat(), "timeZone": tz},
                    "end": {"dateTime": (start + dt.timedelta(minutes=30)).isoformat(), "timeZone": tz}}
            alarm = 30
        who = item["said_by"] or "You"
        where = f" in {item['conversation']}" if item.get("conversation") else ""
        desc = "\n\n".join(x for x in [f"“{item['excerpt']}”" if item.get("excerpt") else "",
                                        f"{who}{where}" + (f" · {item['platform']}" if item.get("platform") else ""),
                                        "Found in your chats by NASH Think."] if x)
        body = {"summary": item["title"], "description": desc, **when,
                "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": alarm}]},
                "extendedProperties": {"private": {"nashthink_id": str(item["id"])}}, "transparency": "transparent"}
        if item.get("repeat") == "yearly":
            body["recurrence"] = ["RRULE:FREQ=YEARLY"]
        return body

    @staticmethod
    def fingerprint(body):
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]

    def sync(self):
        """Bring the NASH Think calendar in line with the open upcoming reminders. Returns counts."""
        if not self.connected():
            raise CalendarError("not_connected", "Google Calendar is not connected.")
        with self._lock:
            try:
                result = self._sync()
            except CalendarError as e:
                self.last_error = e.message
                raise
            except httpx.HTTPError as e:
                self.last_error = f"Could not reach Google: {e.__class__.__name__}"
                raise CalendarError("unreachable", self.last_error)
            self.last_error = None
            self.last_sync = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            self.last_result = result
            return result

    def _sync(self):
        tok = self._read_token()
        now = self.reminders.now()
        wanted = {i["id"]: i for i in self.reminders.upcoming(since_days=1)
                  if i["due_ts"] <= now + HORIZON_DAYS * 86400 or i.get("repeat") == "yearly"}
        synced = self.reminders.google_state()        # id -> (event_id, fingerprint)
        added = updated = removed = 0
        writes = 0
        with self._http() as http:
            headers = {"Authorization": f"Bearer {self._access_token(http, tok)}"}
            cal = urllib.parse.quote(self._calendar_id(http, tok, headers), safe="")
            synced = self.reminders.google_state()     # may have been cleared if the calendar was recreated
            for rid, (event_id, _) in synced.items():
                if rid in wanted or writes >= MAX_WRITES:
                    continue
                r = http.delete(f"{API}/calendars/{cal}/events/{urllib.parse.quote(event_id, safe='')}", headers=headers)
                writes += 1
                if r.status_code not in (200, 204, 404, 410):
                    raise CalendarError("google_error", f"Google Calendar error ({r.status_code}): {_google_message(r)}")
                self.reminders.set_google(rid, None, None)
                removed += 1
            for rid, item in wanted.items():
                if writes >= MAX_WRITES:
                    break
                body = self.event_body(item)
                fp = self.fingerprint(body)
                event_id, old_fp = synced.get(rid, (None, None))
                if event_id and old_fp == fp:
                    continue
                if event_id:
                    r = http.put(f"{API}/calendars/{cal}/events/{urllib.parse.quote(event_id, safe='')}", headers=headers, json=body)
                    if r.status_code in (404, 410):
                        event_id = None
                    elif r.status_code == 200:
                        updated += 1
                if not event_id:
                    r = http.post(f"{API}/calendars/{cal}/events", headers=headers, json=body)
                    if r.status_code in (200, 201):
                        event_id = r.json()["id"]
                        added += 1
                writes += 1
                if r.status_code not in (200, 201):
                    raise CalendarError("google_error", f"Google Calendar error ({r.status_code}): {_google_message(r)}")
                self.reminders.set_google(rid, event_id, fp)
        return {"added": added, "updated": updated, "removed": removed, "in_calendar": len(wanted),
                "pending": max(0, len(wanted) - len(self.reminders.google_state()))}


def _google_message(r):
    try:
        body = r.json()
    except ValueError:
        return (r.text or "")[:200]
    err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err.get("status") or "")[:200]
    return str(body.get("error_description") or err or "")[:200]
