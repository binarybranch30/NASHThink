"""Workspaces: separate data sets served by one Sarthink server, e.g. a public sample archive and the owner's own data.

config/workspaces.json (gitignored) lists them; without it the server has one workspace, exactly as before:

    {"default": "demo",
     "workspaces": {"demo":     {"label": "Sample data", "root": "/home/me/sarthink-demo"},
                    "personal": {"label": "My data", "root": "/home/me/sarthink", "password": "pbkdf2_sha256$..."}}}

A workspace may also carry "examples": {"ask": [...], "search": [...]}, the sample questions its home page offers.

Each root is a Sarthink folder with its own processed_data/ (LanceDB index, memory DB, graph CSVs) and
config/identity_map.json. A browser sees the default workspace until it unlocks another one with that
workspace's password; the choice is a signed, HttpOnly cookie that expires, and every server restart signs
with a new secret, so after a restart every browser is back on the default. Passwords are stored only as
salted PBKDF2 hashes (scripts/utils/set_workspace_password.py writes them) and are never logged.
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

COOKIE = "sarthink_ws"
SESSION_TTL_S = 8 * 3600
PBKDF2_ITERATIONS = 240_000
MAX_FAILURES = 5              # wrong passwords within FAILURE_WINDOW_S before unlocking pauses
FAILURE_WINDOW_S = 300
LOCKOUT_S = 60


class ConfigError(Exception):
    pass


def hash_password(password, salt=None, iterations=PBKDF2_ITERATIONS):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        scheme, iterations, salt, digest = (stored or "").split("$")
        if scheme != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(got.hex(), digest)
    except (ValueError, TypeError):
        return False


def workspace_paths(root):
    """Where a Sarthink folder keeps the data the server reads."""
    root = Path(root)
    return {
        "lancedb": root / "processed_data" / "graph" / "sarthink_lancedb",
        "memory_db": root / "processed_data" / "db" / "sarthink_memory.db",
        "graph_dir": root / "processed_data" / "graph",
        "identity_map": root / "config" / "identity_map.json",
    }


def load_config(path):
    """The parsed config, or None when the file doesn't exist. Raises ConfigError when it is unusable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{path}: {e}")
    spaces = data.get("workspaces")
    if not isinstance(spaces, dict) or not spaces:
        raise ConfigError(f"{path}: 'workspaces' must be a non-empty object")
    default = data.get("default")
    if default not in spaces:
        raise ConfigError(f"{path}: 'default' must name one of the workspaces")
    for wid, ws in spaces.items():
        if not isinstance(ws, dict) or not ws.get("root"):
            raise ConfigError(f"{path}: workspace {wid!r} needs a 'root'")
        if wid != default and not ws.get("password"):
            raise ConfigError(f"{path}: workspace {wid!r} needs a password hash (scripts/utils/set_workspace_password.py)")
    return data


@dataclass
class Workspace:
    id: str
    label: Optional[str]
    search: Any
    insights: Any
    people: Any
    graph_dir: Path
    password_hash: Optional[str] = None
    examples: Optional[dict] = None     # {"ask": [...], "search": [...]}: the home page's sample questions


@dataclass
class WorkspaceSet:
    """The server's workspaces, which one a request uses, and password unlocking."""
    spaces: dict
    default: str
    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    ttl_s: int = SESSION_TTL_S

    def __post_init__(self):
        self._lock = threading.Lock()
        self._failures = deque()
        self._locked_until = 0.0

    @property
    def switchable(self):
        return len(self.spaces) > 1

    # ── sessions ──
    def _sign(self, payload):
        return hmac.new(self.secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def issue(self, wid, now=None):
        expires = int((now or time.time()) + self.ttl_s)
        payload = f"{wid}.{expires}"
        return f"{payload}.{self._sign(payload)}"

    def read(self, token, now=None):
        """The workspace id a cookie grants, or None (missing, tampered, expired or unknown)."""
        try:
            wid, expires, sig = (token or "").rsplit(".", 2)
        except ValueError:
            return None
        if not hmac.compare_digest(sig, self._sign(f"{wid}.{expires}")):
            return None
        if not expires.isdigit() or int(expires) < (now or time.time()) or wid not in self.spaces:
            return None
        return wid

    def for_token(self, token):
        return self.spaces[self.read(token) or self.default]

    # ── unlocking ──
    def unlock(self, wid, password, now=None):
        """True when `password` opens `wid`. Raises TooManyAttempts while paused after repeated failures."""
        now = now or time.time()
        with self._lock:
            if now < self._locked_until:
                raise TooManyAttempts(int(self._locked_until - now) + 1)
            ws = self.spaces.get(wid)
            ok = ws is not None and (wid == self.default or verify_password(password or "", ws.password_hash))
            if ok:
                self._failures.clear()
                return True
            self._failures.append(now)
            while self._failures and now - self._failures[0] > FAILURE_WINDOW_S:
                self._failures.popleft()
            if len(self._failures) >= MAX_FAILURES:
                self._failures.clear()
                self._locked_until = now + LOCKOUT_S
            return False

    def describe(self, active):
        return {
            "active": active.id,
            "label": active.label,
            "default": self.default,
            "switchable": self.switchable,
            "examples": active.examples,
            "workspaces": [{"id": w.id, "label": w.label, "locked": w.id != self.default} for w in self.spaces.values()],
        }


class TooManyAttempts(Exception):
    def __init__(self, retry_after_s):
        super().__init__(f"Too many wrong passwords. Try again in {retry_after_s} s.")
        self.retry_after_s = retry_after_s


def shared_model_loader(load):
    """One embedding model per name for every workspace's SearchService (it is large; load it once)."""
    cache, lock = {}, threading.Lock()

    def loader(name):
        with lock:
            if name not in cache:
                cache[name] = load(name)
            return cache[name]
    return loader


def default_config_path(repo_root):
    return Path(os.environ.get("SARTHINK_WORKSPACES") or Path(repo_root) / "config" / "workspaces.json")
