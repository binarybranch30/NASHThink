"""Rule-based reminder extraction: dates, times and plans in English and Hinglish chat lines.

`extract(text, sent_at_utc)` returns the reminders one message implies ("kal 6 baje call", "review on 2026-09-07",
"I will send it by Friday", "Happy birthday Maa!"). Relative dates are resolved against the message's own time, in
the owner's timezone. No model, no I/O: a regex pass that is cheap enough to run over a whole archive.

A candidate needs a date or time AND a reason to remember it (a meeting, a deadline, a promise...). Weak lines are
scored below MIN_CONFIDENCE and dropped; negated or past-tense lines ("not posting it tonight", "we missed the
train", "kal gaya tha") are skipped.
"""
import datetime as dt
import re
from dataclasses import dataclass, field
from typing import List, Optional
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Kolkata"
MIN_CONFIDENCE = 0.55
TITLE_CHARS = 110

MONTHS = {m: i for i, names in enumerate([
    ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
    ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
    ("dec", "december")], start=1) for m in names}
WEEKDAYS = {name: i for i, names in enumerate([
    ("monday", "mon"), ("tuesday", "tue", "tues"), ("wednesday", "wed"), ("thursday", "thu", "thurs"),
    ("friday", "fri"), ("saturday", "sat"), ("sunday", "sun")]) for name in names}
# Hinglish parts of the day and the hour a bare "6 baje" / "tomorrow morning" means.
DAYPARTS = {"morning": 9, "subah": 9, "afternoon": 14, "dopahar": 14, "evening": 18, "shaam": 18, "sham": 18,
            "night": 21, "raat": 21, "tonight": 21}
PM_PARTS = {"afternoon", "dopahar", "evening", "shaam", "sham", "night", "raat", "tonight"}

_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_WD_RE = "|".join(sorted(WEEKDAYS, key=len, reverse=True))
_ORD = r"(?:st|nd|rd|th)?"

ISO_RE = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")
DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}}){_ORD}(?:\s+of)?\s+({_MONTH_RE})\b\.?(?:,?\s+(20\d\d))?", re.I)
MONTH_DAY_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}}){_ORD}\b(?:,?\s+(20\d\d))?", re.I)
SLASH_RE = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?:/(\d{2}|20\d\d))?(?![\d/])")   # DD/MM, the Indian order
REL_DAY_RE = re.compile(r"\b(day after tomorrow|parso|parson|tomorrow|tmrw|tmr|kal|today|aaj|tonight)\b", re.I)
WEEKDAY_RE = re.compile(rf"\b(?:(next|this|coming|agle|agla|is)\s+)?({_WD_RE})\b(\s+(?:ko|tak|wale|waale))?", re.I)
WEEKDAY_BEFORE_RE = re.compile(r"\b(?:on|by|till|until|before|after|from|for|see you|upto|up to)\s+$", re.I)
WEEKDAY_AFTER_RE = re.compile(r"\s*(?:$|[?.!,;]|(?:ko|tak|se|morning|evening|night|afternoon|subah|shaam|raat|at|"
                              r"\d)\b)", re.I)
IN_DAYS_RE = re.compile(r"\b(?:in\s+(\d{1,2}|a|one|two|three)\s+(day|days|week|weeks)|(\d{1,2})\s+(din|dino|hafte)\s+(?:mein|me|baad))\b", re.I)
NEXT_WEEK_RE = re.compile(r"\b(next week|agle (?:hafte|week)|this weekend|weekend pe|weekend par|next weekend)\b", re.I)

TIME_RE = re.compile(r"\b(?:at\s+)?(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)(?![a-z])", re.I)
TIME_24_RE = re.compile(r"\b(?:at\s+)?([01]?\d|2[0-3]):([0-5]\d)\b(?:\s*(?:ist|hrs|h)\b)?", re.I)
BAJE_RE = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*baje\b", re.I)
DAYPART_RE = re.compile(r"\b(morning|subah|afternoon|dopahar|evening|shaam|sham|night|raat|tonight)\b", re.I)

# kind -> (weight, patterns). The first kind that matches names the reminder.
KINDS = [
    ("birthday", 0.45, r"birthday|b'?day|bday|janamdin|janmdin"),
    ("reminder", 0.45, r"remind me|remind you|don'?t forget|do not forget|bhool mat|bhoolna mat|bhulna mat|yaad dila\w*|yaad rakh\w*|set a reminder"),
    ("exam", 0.4, r"exam|exams|viva|quiz|mid-?sem|end-?sem|pariksha|imtihaan|test on"),
    ("interview", 0.4, r"interview|interviews"),
    ("deadline", 0.4, r"deadline|due|last date|submit|submission|apply by|register by"),
    ("appointment", 0.4, r"appointment|surgery|doctor|dentist|vet|hospital|checkup|check-up|clinic"),
    ("travel", 0.35, r"train|flight|bus ticket|overnight bus|departure|check-?in|boarding|trip|ticket|tickets"),
    ("payment", 0.35, r"pay|payment|rent|fees|fee|bill|emi|transfer|₹"),
    ("meeting", 0.35, r"meeting|meet|meet-?up|milte|milna|milenge|review|discussion|catch up|sync|standup|call|video call|class|lecture|session|demo"),
    ("plan", 0.3, r"plan|outing|movie|dinner|lunch|party|match|hangout|chalenge|chalte"),
    ("task", 0.25, r"i will|i'll|will send|will bring|will move|will call|will pay|we will|we'll|"
                   r"let's|lets|bring|send|pick up|drop|dunga|dungi|denge|karunga|karungi|karenge|"
                   r"bhej\w*|le aana|le aaunga|lana|can you|could you|please"),
]
KIND_RES = [(kind, weight, re.compile(rf"(?<![\w-])(?:{pat})(?![\w])", re.I)) for kind, weight, pat in KINDS]

NEGATION_RE = re.compile(r"\b(not|no need|don't|dont|won't|wont|can't|cant|cancel\w*|postpone\w*|skip\w*|nahi|nhi|"
                         r"mat|never|called off)\b", re.I)
POSITIVE_NEGATIONS_RE = re.compile(r"\b(don'?t forget|do not forget|bhool(?:na)? mat|bhulna mat)\b", re.I)
PAST_RE = re.compile(r"\b(was|were|missed|did|had|went|happened|yesterday|last (?:week|night|time)|ago|kiya|kiye|"
                     r"helped|settles|settled|tried|confirms|confirmed|finished|sent|moved|looked|explains why|"
                     r"back with|reopening|revisited|reviewed|received|completed|"
                     r"gaya|gaye|gayi|tha|thi|hua|hui|huye|diya|liya|dekha)\b", re.I)
FUTURE_RE = re.compile(r"\b(will|shall|going to|gonna|let's|lets|can we|could we|are we|should we|karenge|"
                       r"karunga|karungi|milte|milenge|chalenge|chalte|aaunga|aaungi|aayenge|dunga|dungi|denge|"
                       r"hoga|hogi|honge|jaana|jana|jaenge|jayenge|jaunga|at \d|\d\s*baje)\b", re.I)
HEDGE_RE = re.compile(r"\b(might|maybe|may be|shayad|probably|perhaps)\b", re.I)
WISH_RE = re.compile(r"\b(happy|many happy returns of the day|hbd)\b.*\b(birthday|b'?day|bday|returns)\b|\bhbd\b", re.I)
LEAD_RE = re.compile(r"^(?:(?:hi|hello|hey|dear)\s+[\w.]+,?\s*|(?:bhai|suno|yaar|arre|achha|acha|haan|ok|okay|"
                     r"chalo|ek sec|check|repro note|note|fyi|update|reminder)\s*[:,!-]\s*)+", re.I)
SKIP_PREFIXES = ("searched for", "watched ", "visited ", "viewed ")


@dataclass
class Candidate:
    title: str
    due_at: int                   # UTC epoch seconds; for all-day items, local midnight of that day
    all_day: bool
    kind: str
    confidence: float
    span: str                     # the words that gave the date
    repeat: Optional[str] = None  # "yearly" for birthdays
    date_local: str = ""          # YYYY-MM-DD in the owner's timezone
    notes: List[str] = field(default_factory=list)


def _clamp_year(day, month, year, base):
    """An explicit day+month without a year: the occurrence nearest to `base`, preferring the future."""
    if year:
        year = int(year)
        year = year + 2000 if year < 100 else year
        try:
            return dt.date(year, month, day)
        except ValueError:
            return None
    best = None
    for y in (base.year - 1, base.year, base.year + 1):
        try:
            d = dt.date(y, month, day)
        except ValueError:
            continue
        # Dates more than ~2 months back belong to next year ("exam on 5 Jan" sent in December).
        if d >= base - dt.timedelta(days=60) and (best is None or d < best):
            best = d
    return best


def _next_weekday(base, wd, qualifier):
    ahead = (wd - base.weekday()) % 7
    if qualifier in ("next", "agle", "agla") and ahead <= 1:
        ahead += 7
    elif ahead == 0 and qualifier not in ("this", "is"):
        ahead = 7
    return base + dt.timedelta(days=ahead)


def find_date(text, base, has_future, near=None):
    """(date, span, strength) of the date in `text`, relative to the local date `base`; None when there is none.
    The most explicit date wins; among equals, the one nearest `near` (where the reason to remember it is), else
    the first. `kal` means tomorrow only with a future cue in the line (it is also 'yesterday')."""
    found = []
    for m in ISO_RE.finditer(text):
        try:
            found.append((m.start(), dt.date(int(m[1]), int(m[2]), int(m[3])), m[0], 0.5))
        except ValueError:
            pass
    for m in DAY_MONTH_RE.finditer(text):
        d = _clamp_year(int(m[1]), MONTHS[m[2].lower()], m[3], base)
        if d:
            found.append((m.start(), d, m[0], 0.5))
    for m in MONTH_DAY_RE.finditer(text):
        d = _clamp_year(int(m[2]), MONTHS[m[1].lower()], m[3], base)
        if d:
            found.append((m.start(), d, m[0], 0.5))
    for m in SLASH_RE.finditer(text):
        day, month = int(m[1]), int(m[2])
        if 1 <= day <= 31 and 1 <= month <= 12:
            d = _clamp_year(day, month, m[3], base)
            if d:
                found.append((m.start(), d, m[0], 0.45))
    for m in REL_DAY_RE.finditer(text):
        w = m[1].lower()
        if w in ("kal",) and not has_future:
            continue
        offset = {"day after tomorrow": 2, "parso": 2, "parson": 2, "tomorrow": 1, "tmrw": 1, "tmr": 1, "kal": 1}.get(w, 0)
        found.append((m.start(), base + dt.timedelta(days=offset), m[0], 0.4))
    for m in WEEKDAY_RE.finditer(text):
        name = m[2].lower()
        if name in ("sat", "sun", "mon", "wed") and not m[1] and not m[3]:
            continue    # "sat" / "sun" / "mon" / "wed" alone are ordinary words
        q = (m[1] or "").lower()
        if not q and not m[3] and not (WEEKDAY_BEFORE_RE.search(text[:m.start()]) or WEEKDAY_AFTER_RE.match(text[m.end():])):
            continue    # "the Friday idea", "in Sunday shortlist": a name, not a date ("on Friday", "Friday ko" are)
        found.append((m.start(), _next_weekday(base, WEEKDAYS[name], q), m[0], 0.35))
    for m in IN_DAYS_RE.finditer(text):
        n = m[1] or m[3]
        n = {"a": 1, "one": 1, "two": 2, "three": 3}.get(str(n).lower(), n)
        unit = (m[2] or m[4]).lower()
        days = int(n) * (7 if unit.startswith(("week", "hafte")) else 1)
        found.append((m.start(), base + dt.timedelta(days=days), m[0], 0.3))
    for m in NEXT_WEEK_RE.finditer(text):
        w = m[1].lower()
        if "weekend" in w:
            d = _next_weekday(base, 5, "next" if w.startswith("next") else "")
        else:
            d = base + dt.timedelta(days=7 - base.weekday())    # next Monday
        found.append((m.start(), d, m[0], 0.2))
    if not found:
        return None
    found.sort(key=lambda f: (-f[3], abs(f[0] - near) if near is not None else f[0]))
    _, d, span, strength = found[0]
    return d, span, strength


def find_time(text):
    """(hour, minute, span) or None. Parts of the day ('tomorrow morning', 'shaam') give a default hour."""
    part = DAYPART_RE.search(text)
    part_word = part[1].lower() if part else None
    m = TIME_RE.search(text)
    if m:
        h, mi, ap = int(m[1]), int(m[2] or 0), m[3].lower().replace(".", "")
        if 1 <= h <= 12 and mi < 60:
            h = h % 12 + (12 if ap == "pm" else 0)
            return h, mi, m[0].strip()
    m = TIME_24_RE.search(text)
    if m:
        return int(m[1]), int(m[2]), m[0].strip()
    m = BAJE_RE.search(text)
    if m:
        h, mi = int(m[1]), int(m[2] or 0)
        if 1 <= h <= 12:
            if part_word in PM_PARTS and h < 12:
                h += 12
            elif not part_word and 1 <= h <= 7:
                h += 12    # "6 baje" with no part of the day is usually the evening
            return h, mi, m[0].strip()
    if part_word:
        return DAYPARTS[part_word], 0, part[0]
    return None


def classify(text):
    """(kind, weight, position) of the strongest reason in `text` to remember it, or (None, 0, None)."""
    for kind, weight, rx in KIND_RES:
        m = rx.search(text)
        if m:
            return kind, weight, m.start()
    return None, 0.0, None


def make_title(sentence):
    t = re.sub(r"\s+", " ", sentence).strip(" -–—:;,.")
    t = LEAD_RE.sub("", t).strip()
    t = re.sub(r"\s*[\U0001F300-\U0001FAFF☀-➿✨📌]+\s*$", "", t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    return t if len(t) <= TITLE_CHARS else t[:TITLE_CHARS - 1].rstrip() + "…"


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p for p in (s.strip() for s in parts) if p]


def _local(ts, tz):
    return dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).astimezone(tz)


def extract(text, sent_at_utc, tz=DEFAULT_TZ, birthday_of=None):
    """Reminder candidates in one message. `birthday_of`: who a "Happy birthday" wish is for (the caller knows
    who the message was sent to); without it the wish is not turned into a yearly reminder."""
    if not text or not sent_at_utc:
        return []
    zone = ZoneInfo(tz) if isinstance(tz, str) else tz
    sent = _local(sent_at_utc, zone)
    base = sent.date()
    low = text.strip().lower()
    if low.startswith(SKIP_PREFIXES):
        return []

    out = []
    if WISH_RE.search(text) and birthday_of:
        nxt = dt.date(base.year + 1, base.month, 28 if (base.month, base.day) == (2, 29) else base.day)
        out.append(_candidate(f"{birthday_of}'s birthday", nxt, None, True, "birthday", 0.9, "birthday wish", zone,
                              repeat="yearly", first=base))
        return out

    seen = set()
    for s in sentences(text):
        future = bool(FUTURE_RE.search(s))
        kind, weight, kind_pos = classify(s)
        date = find_date(s, base, future, kind_pos)
        time_ = find_time(s)
        clock = time_ is not None and not DAYPART_RE.fullmatch(time_[2])
        if not date and not (clock and kind and kind != "task" and (future or kind == "reminder")):
            continue
        cleaned = POSITIVE_NEGATIONS_RE.sub("", s)
        if NEGATION_RE.search(cleaned):
            continue
        past = PAST_RE.search(s) and not future
        if date:
            d, span, strength = date
            if past and d <= base:
                continue
        else:
            # A bare time ("call at 6pm"): today, or tomorrow when that time has already gone.
            d, span, strength = base, time_[2], 0.3
            if (time_[0], time_[1]) <= (sent.hour, sent.minute):
                d = base + dt.timedelta(days=1)
        if not kind:
            weight = 0.0
        score = strength + weight + (0.1 if time_ else 0.0) + (0.05 if future else 0.0)
        if HEDGE_RE.search(s):
            score -= 0.2
        if past:
            score -= 0.15
        if score < MIN_CONFIDENCE:
            continue
        key = (d, kind)
        if key in seen:
            continue
        seen.add(key)
        title = make_title(s)
        if not title:
            continue
        out.append(_candidate(title, d, time_, time_ is None, kind or "plan", round(min(score, 1.0), 2), span, zone))
    return out


def _candidate(title, date, time_, all_day, kind, confidence, span, zone, repeat=None, first=None):
    if all_day or time_ is None:
        local = dt.datetime.combine(date, dt.time(0, 0), zone)
    else:
        local = dt.datetime.combine(date, dt.time(time_[0] % 24, time_[1]), zone)
    c = Candidate(title=title, due_at=int(local.timestamp()), all_day=all_day or time_ is None, kind=kind,
                  confidence=confidence, span=span, repeat=repeat, date_local=date.isoformat())
    if first:
        c.notes.append(f"wished on {first.isoformat()}")
    return c
