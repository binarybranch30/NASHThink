# Reminders and calendar sync

NASH Think reads your chats for plans, deadlines, birthdays and promises ("kal 6 baje call karenge",
"submit by Friday, bhool mat", "Maa's surgery is on 12 March", "Happy birthday!") and lists them under
**Reminders** (button in the top bar, or `R`). Each reminder links to the chat it came from.

## How reminders are found

- `scripts/semantic/reminder_rules.py` checks each message for a **date or time** (ISO dates, "12 March", DD/MM,
  today/kal/parso, "agle Monday", "Friday tak", "6pm", "shaam 6 baje", "tomorrow morning") **and a reason to
  remember it** (meeting, call, deadline, exam, interview, appointment, travel, payment, "remind me",
  "bhool mat", "I'll send…"). Negated, hedged or past-tense lines are skipped ("not posting it tonight",
  "we missed the train", "kal gaya tha").
- Relative dates are resolved against the message's own time, in the workspace's timezone (Asia/Kolkata by
  default). "kal" counts as tomorrow only when the line looks ahead ("kal milte hai", "kal 6 baje").
- A birthday wish becomes a yearly reminder: "Your birthday" when someone wishes you, "Rohan's birthday" when
  you wish Rohan.
- No language model is involved. The sample archive (12,016 messages) takes about 2 s.

`scripts/api/reminders.py` stores them in `processed_data/reminders/<workspace>.db`. The memory database is
only ever read. There is one reminder per conversation per day, and repeated mentions are counted. Scans are
incremental: only messages that arrived since the last scan are read. The first scan of a workspace covers the
last 60 days (`lookback_days`).

Done, snoozed, dismissed, edited and hand-added reminders are kept in the same store. A re-scan never undoes
them. On the public default workspace of a multi-workspace server (the sample data), visitors' changes stay in
their own browser instead.

### Workspace settings

In `config/workspaces.json`, each workspace can have a `reminders` block:

```json
"demo": {"root": "…", "reminders": {"now": "latest_message", "lookback_days": null}},
"personal": {"root": "…", "password": "…", "reminders": {"tz": "Asia/Kolkata", "lookback_days": 60}}
```

- `now`: `"clock"` (default) or `"latest_message"`. The sample archive ends in the past, so its last message
  stands in for "today".
- `lookback_days`: how far back the first scan reads. `null` means the whole archive.
- `tz`: the timezone that dates like "kal 6 baje" are read in.

`config/reminders.json` (copy `config/reminders.json.example`) controls the background job:
`{"enabled": true, "interval_min": 15}`.

## Getting them into a calendar

In the Reminders panel, **Calendar** offers:

| Way | What happens | Leaves the machine |
|---|---|---|
| **Automatic Google Calendar sync** | New and changed reminders appear in a separate "NASH Think" calendar in your Google account within about 15 minutes; done or dismissed ones disappear. Your phone's Google Calendar reminds you: 30 min before, or 9:00 the day before an all-day one. | Title, time and one line from the message, for upcoming reminders only |
| **+ Google Calendar** on a card | Opens Google Calendar with that one reminder filled in; nothing is saved until you click Save there | Only what you save |
| **Download .ics** | A calendar file for Google, Apple or Outlook Calendar (a one-time import) | Nothing |
| **Subscribe link** | A secret link that Apple Calendar or Thunderbird can keep polling | Nothing (the app must be reachable from the calendar app) |
| **Alert me in this browser** | A browser notification while NASH Think is open in a tab | Nothing |

### Setting up Google Calendar sync

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project (or use one) and enable the
   **Google Calendar API**.
2. Under **APIs & Services → OAuth consent screen**, choose "External" and add yourself as a **test user**.
   The app can stay in testing mode; it is only for you.
3. Under **APIs & Services → Credentials → Create credentials → OAuth client ID**, choose
   **Desktop app**, then copy the client ID and secret into `.env`:
   ```
   GOOGLE_CLIENT_ID=….apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=…
   ```
4. Restart the server (`scripts/stop_sarthink.sh && scripts/start_sarthink.sh`).
5. Open NASH Think at **`http://127.0.0.1:8000/`**. If it runs on a server, use an SSH tunnel:
   `ssh -N -L 8000:127.0.0.1:8000 <user>@<server>`. Google only returns to 127.0.0.1 or localhost addresses.
6. Unlock your own workspace, open **Reminders → Calendar → Connect Google Calendar**, and allow access.

Details:

- The permission asked for is `calendar.app.created`: the app can only see and change calendars it created
  itself. Your other calendars are never read.
- The Google sign-in (a refresh token) is saved in `processed_data/reminders/<workspace>.google.json`, which
  only you can read (mode 600) and git ignores. **Disconnect** deletes it and revokes it at Google.
- The "NASH Think" calendar is managed by the app. Edits made there may be overwritten; change reminders in
  NASH Think instead. If you delete the calendar, the next sync creates a new one.
- In testing mode, Google expires refresh tokens after 7 days. When that happens the panel says
  "Connect Google Calendar again". To avoid it, publish the consent screen (no verification needed for your
  own use).
- The sample workspace cannot connect a Google account.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/reminders` | Grouped reminders (overdue, today, week, later, history, done), counts, badge |
| `POST /api/reminders` | Add one: `{title, due, all_day}` |
| `POST /api/reminders/{id}` | `{action: done \| open \| dismiss \| snooze \| edit, until?, title?, due?, all_day?}` |
| `POST /api/reminders/scan` | Read new messages now |
| `GET /api/reminders.ics` | Calendar file (`?feed=<secret>` from `GET /api/reminders/feed` to subscribe) |
| `GET /api/calendar/status` | Google Calendar: configured, connected, last sync, last error |
| `GET /api/calendar/google/connect` · `…/callback` | Connect (browser redirects) |
| `POST /api/calendar/google/sync`, `POST /api/calendar/google/disconnect` | Sync now; disconnect |
