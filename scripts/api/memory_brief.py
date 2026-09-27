"""Evidence-first memory briefs for POST /api/ask. Deterministic: no language model, no network.

Given the chunks semantic search retrieved for a question, this module decides which of them actually
support it (similarity *and* keyword support *and* enough substance), removes overlapping duplicates,
quotes short sentences from the survivors and states how strong the evidence is.

Every statement in the answer is either a verbatim quote from a returned source or a count, date,
platform or title read from one. Nothing is paraphrased or inferred, so the brief can be thin, but it
never claims a feeling, event or relationship the archive doesn't show.
"""
import datetime as dt
import os
import re
import sys
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "semantic"))
import hinglish  # noqa: E402  (shared Hinglish vocabulary: filler, spelling variants, topic equivalents)

UTC = dt.timezone.utc

# Similarity alone is unreliable on this index: very short chunks ("ok", "lol") score 0.6+ against
# unrelated questions. A source only counts as evidence when all of these hold.
MIN_SIMILARITY = 0.35        # cosine similarity (1 - distance)
MIN_SIMILARITY_NO_TERMS = 0.45  # when the question has no content words to check against
# A chunk whose support comes from the *other* language (an English question answered by "neend", a Hinglish
# one by "sleep") scores low on similarity even when on topic (~0.2-0.4 on the real index), so it needs a
# lower floor. It still has to mention the topic word, have real content and meet the term rule below.
MIN_SIMILARITY_CROSS = 0.2
MIN_BODY_CHARS = 40          # message text, excluding the Platform/Title/... header
QUOTE_CHARS = 200
TITLE_CHARS = 70
SOURCE_TEXT_CHARS = 2000
MAX_SCAN_CHARS = 20000       # sentence extraction reads at most this much of a chunk
CONTEXT_LINES = 2            # lines kept on each side of a line that mentions the question's words
GAP = "…"
MAX_POINTS = 5
MAX_TIMELINE = 12

PLATFORM_NAMES = {
    "twitter": "Twitter / X", "reddit": "Reddit", "instagram": "Instagram", "discord": "Discord",
    "facebook": "Facebook", "whatsapp": "WhatsApp", "google": "Google", "chatgpt": "ChatGPT", "claude": "Claude",
}
MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
MONTHS = {name: i for i, names in enumerate((
    ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"), ("may",), ("june", "jun"),
    ("july", "jul"), ("august", "aug"), ("september", "sep", "sept"), ("october", "oct"),
    ("november", "nov"), ("december", "dec")), start=1) for name in names}

STOPWORDS = set("""
a an and are as at be been but by can could did do does doing for from had has have how i if in into is it
its me my myself of on or our so than that the their them then there these they this those to was we were
what when where which who whom why will with would you your yours about ever any some all just really
very much many more most also like get got
""".split())
# Question scaffolding: words that say what kind of answer is wanted, not what it is about.
FILLER = set("""
discuss discussed discussing discussion talk talked talking say said mention mentioned tell told
around during over time times period changed change changes evolve evolved evolving interest interests
interested feel felt feeling think thought thinking remember recall anything something things thing
year years month months day days recently lately ago back work working worked doing going happening
stuff busy
""".split())
TREND_RE = re.compile(r"\b(chang|evolv|over (the )?(time|years)|grow|shift|progress|develop|trend|still)", re.I)
MONTH_WORD = "|".join(sorted(MONTHS, key=len, reverse=True))
MONTH_YEAR_RE = re.compile(rf"\b(?:(around|about|circa|near|in|during|by)\s+)?({MONTH_WORD})\.?,?\s+((?:19|20)\d\d)\b", re.I)
YEAR_RE = re.compile(r"\b(in|during|around|throughout)\s+((?:19|20)\d\d)\b", re.I)

HEADER_PREFIXES = ("Platform:", "Title:", "Subreddit:", "Participants:", "Timeframe:")
CHAT_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] ([^:\n]{1,80}): ?(.*)$")
POST_RE = re.compile(r"^\[(?:POST|REPLY/COMMENT|TWEET|REPLY)\] (\S+) — (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
URL_RE = re.compile(r"https?://\S+")
WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


# ─── Dates ────────────────────────────────────────────────────────────────────

def parse_bound(value, end=False):
    """ISO date or datetime -> aware UTC datetime. A date-only upper bound means the end of that day."""
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        d = dt.datetime.fromisoformat(text).replace(tzinfo=UTC)
        return d + dt.timedelta(days=1) if end else d
    d = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=UTC)).astimezone(UTC)


def iso(d):
    return d.astimezone(UTC).isoformat() if d else None


def parse_time(value):
    """Stored chunk time or '%Y-%m-%d %H:%M:%S' message time -> aware UTC datetime, else None."""
    if not value:
        return None
    try:
        d = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=UTC)).astimezone(UTC)


def month_label(d):
    return f"{MONTH_ABBR[d.month - 1]} {d.year}" if d else "undated"


def add_months(d, n):
    m = d.month - 1 + n
    return d.replace(year=d.year + m // 12, month=m % 12 + 1, day=1)


def implied_window(question):
    """Date window named in the question ("around August 2026", "in 2025"), or None.

    Returns (from, to_exclusive, matched_phrase). "around"/"about" widens a month to +-1 month.
    """
    m = MONTH_YEAR_RE.search(question)
    if m:
        start = dt.datetime(int(m.group(3)), MONTHS[m.group(2).lower()], 1, tzinfo=UTC)
        fuzzy = (m.group(1) or "").lower() in ("around", "about", "circa", "near")
        return (add_months(start, -1) if fuzzy else start), add_months(start, 2 if fuzzy else 1), m.group(0)
    m = YEAR_RE.search(question)
    if m:
        year = int(m.group(2))
        return dt.datetime(year, 1, 1, tzinfo=UTC), dt.datetime(year + 1, 1, 1, tzinfo=UTC), m.group(0)
    return None


def plan_query(question, date_from=None, date_to=None):
    """Text to embed and the effective date window. Explicit bounds win over a window named in the question;
    a named window is removed from the embedded text since dates carry no meaning for the embedding."""
    notes = []
    start, end = parse_bound(date_from), parse_bound(date_to, end=True)
    query = question
    if start is None and end is None:
        found = implied_window(question)
        if found:
            start, end, phrase = found
            stripped = re.sub(r"\s+([?.!,])", r"\1", re.sub(r"\s{2,}", " ", question.replace(phrase, " "))).strip(" ,")
            if len(WORD_RE.findall(stripped)) >= 2:
                query = stripped
            notes.append(f"Limited to {month_label(start)} – {month_label(end - dt.timedelta(days=1))} "
                         f"because the question mentions “{phrase.strip()}”.")
    return {"query": query, "date_from": start, "date_to": end, "notes": notes}


# ─── Text ─────────────────────────────────────────────────────────────────────

def stem(word):
    for suffix in ("ational", "ations", "ation", "ments", "ment", "ingly", "ings", "ing", "edly", "ies", "ied",
                   "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def term_key(term):
    """Prefix that text words must start with to count as mentioning `term`."""
    s = stem(term)
    return s[:-2] if len(s) >= 9 else s   # photography -> photograp (photographs, photographer)


def content_terms(question):
    """The question's topic words. Hinglish filler and negations ("hai", "mujhe", "nahi") are skipped and
    Hinglish spellings are normalised (padhaai -> padhai); English words are kept as typed."""
    raw = WORD_RE.findall(question.lower())
    hi = hinglish.looks_hinglish(raw)
    seen = []
    for w in raw:
        # Negations ("couldn't", "nahi") and scaffolding ("feel") say how, not what: never topic words.
        if hinglish.is_filler(w, hi) or hinglish.is_negation(w) or w in hinglish.SCAFFOLD_EN:
            continue
        w = hinglish.normalize(w)
        if len(w) > 2 and w not in STOPWORDS and w not in FILLER and not w.isdigit() and w not in MONTHS and w not in seen:
            seen.append(w)
    return seen


def question_is_hinglish(question):
    return hinglish.looks_hinglish(WORD_RE.findall((question or "").lower()))


class TermMatcher:
    """Decides which question terms a piece of text mentions.

    A term is mentioned when a text word starts with its key (existing English behaviour: stressed ~ stressful),
    when a text word is the same Hinglish word in another spelling (nind ~ neend), or when a text word is the
    same topic in the other language (neend ~ sleeping). Topic words from hinglish.CONCEPTS list their forms
    explicitly, so they match whole words only (exam no longer matches "example", pain no longer "painting").
    In a Hinglish question, other words match through hinglish.word_pattern (ladai ~ ladaai ~ ladayi), and
    words of three letters or fewer (jee) must match exactly, so "jee" is not "jeet" or "jeeneetards".
    Negations and filler in the text never count, and nothing in the text is changed.
    """

    def __init__(self, terms, hinglish_question=False):
        self.terms = list(terms)
        self.keys = [term_key(t) for t in self.terms]
        self.generic = [bool((c := hinglish.concept_of(t)[0]) and not hinglish.CONCEPTS[c]["expand"]) for t in self.terms]
        self.patterns = []   # per term: compiled full-word pattern for loose Hinglish spellings, or None
        for t in self.terms:
            p = hinglish.word_pattern(t) if hinglish_question and not hinglish.concept_of(t)[0] else None
            self.patterns.append(re.compile(rf"(?:{p})", re.I) if p else None)
        self.equiv = []   # per term: {normalised word: 'same' | 'cross'}
        for t in self.terms:
            concept, lang = hinglish.concept_of(t)
            eq = {}
            if concept:
                for l in ("en", "hi"):
                    for w in hinglish.surface_forms(concept, l):
                        eq[hinglish.normalize(w)] = "same" if l == lang else "cross"
            self.equiv.append(eq)

    def match(self, words, prefix=True):
        """-> (matched question terms, display words, uses_cross) for a set of lowercase text words.
        prefix=False (thread titles) accepts whole words only: "jee" never matches "r/JEENEETards"."""
        norm = {hinglish.normalize(w) for w in words}
        matched, shown, cross_only = [], [], False
        for t, k, eq, pat in zip(self.terms, self.keys, self.equiv, self.patterns):
            if pat is not None:
                hits = sorted(w for w in norm if pat.fullmatch(w))
                if hits:
                    matched.append(t)
                    shown.append(t if t in hits else hits[0])
                continue
            exact = t in norm or t + "s" in norm   # short words (jee, car) and titles: the word itself, or its plural
            if not eq and (exact if (not prefix or len(t) <= 3) else any(w.startswith(k) for w in norm)):
                matched.append(t)
                shown.append(t)
                continue
            hits = sorted(w for w in norm if w in eq)
            if hits:
                matched.append(t)
                shown.append(t if t in hits else hits[0])
                cross_only = cross_only or all(eq[w] == "cross" for w in hits)
        return matched, shown, cross_only

    def count(self, words):
        return len(self.match(words)[0])

    def specific(self, matched):
        """True when `matched` includes a specific term, or the question has none (only ghar, dost, problem)."""
        wanted = [t for t, g in zip(self.terms, self.generic) if not g]
        return not wanted or any(t in matched for t in wanted)


def words_of(text):
    return set(WORD_RE.findall((text or "").lower()))


def mentions(words, key):
    return any(w.startswith(key) for w in words)


def parse_messages(text):
    """Chunk text -> [{author, ts, content}], skipping the context header. Formats: chunk_builder.format_message."""
    messages, cur = [], None
    for line in (text or "")[:MAX_SCAN_CHARS].splitlines():
        if line.startswith(HEADER_PREFIXES):
            continue
        m = CHAT_RE.match(line)
        if m:
            cur = {"ts": parse_time(m.group(1)), "author": m.group(2).strip(), "content": m.group(3)}
            messages.append(cur)
            continue
        m = POST_RE.match(line)
        if m:
            cur = {"ts": parse_time(m.group(2)), "author": m.group(1), "content": ""}
            messages.append(cur)
            continue
        if not line.strip():
            continue
        if cur is None:
            cur = {"ts": None, "author": None, "content": ""}
            messages.append(cur)
        cur["content"] = f"{cur['content']}\n{line}" if cur["content"] else line
    return messages


def clip(text, limit):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(" ", 1)[0] if " " in text[:limit - 1] else text[:limit - 1]
    return cut.rstrip(" ,;:") + "…"


NEGATION_CONTEXT_WORDS = 4   # words kept after a negation so its verb stays attached ("neend nahi aati thi")


def safe_clip(text, limit, hard_max=None):
    """clip() that never drops a negation ("nahi", "not", "never" ...) from the kept part of a sentence.

    If the cut would leave out a negation, the quote is extended to include it plus a few words after it,
    up to `hard_max` characters. Returns None when even that isn't possible, so callers pick another quote
    instead of showing one whose meaning could be reversed.
    """
    flat = re.sub(r"\s+", " ", text or "").strip()
    if len(flat) <= limit:
        return flat
    hard_max = hard_max or limit * 2
    kept = clip(flat, limit)
    kept_len = len(kept.rstrip("…"))
    toks = list(WORD_RE.finditer(flat))
    negs = [i for i, m in enumerate(toks) if m.start() >= kept_len and hinglish.is_negation(m.group(0))]
    if not negs:
        return kept
    last = negs[-1]
    end_tok = toks[min(len(toks) - 1, last + NEGATION_CONTEXT_WORDS)]
    end = end_tok.end()
    if end > hard_max:
        return None
    return flat if end >= len(flat) else flat[:end].rstrip(" ,;:") + "…"


def best_quote(messages, keys, matcher=None):
    """The sentence that mentions the most question terms (ties: a readable length, then the earliest).

    With a TermMatcher, spelling variants and other-language topic words count as mentions too. A sentence
    that can't be shortened without losing a negation is skipped (see safe_clip).
    """
    best, best_score = None, None
    for mi, msg in enumerate(messages):
        for si, sentence in enumerate(SENTENCE_SPLIT_RE.split(msg["content"] or "")):
            sentence = URL_RE.sub("", sentence).strip(" -*>•\t")
            words = WORD_RE.findall(sentence.lower())
            hits = matcher.count(words) if matcher else sum(1 for k in keys if any(w.startswith(k) for w in words))
            if len(words) < (3 if hits else 5):
                continue
            score = (hits, 1 if 6 <= len(words) <= 40 else 0, -mi, -si)
            if best_score is None or score > best_score:
                text = safe_clip(sentence, QUOTE_CHARS)
                if text is None:
                    continue
                best, best_score = {"text": text, "author": msg["author"], "ts": msg["ts"], "hits": hits}, score
    return best


# ─── Evidence ─────────────────────────────────────────────────────────────────

def assess(result, terms, keys, matcher=None):
    """Adds the evidence fields build_brief needs to one /api/search-style result."""
    matcher = matcher or TermMatcher(terms)
    text = result.get("text") or ""
    messages = parse_messages(text)
    body = " ".join(m["content"] for m in messages).strip()
    matched, shown, cross = matcher.match(words_of(body))
    t_matched, t_shown, t_cross = matcher.match(words_of(result.get("title")), prefix=False)
    for t, w in zip(t_matched, t_shown):
        if t not in matched:
            matched.append(t)
            shown.append(w)
            cross = cross or t_cross
    matched_order = {t: k for k, t in enumerate(matcher.terms)}
    pairs = sorted(zip(matched, shown), key=lambda p: matched_order.get(p[0], 99))
    matched, shown = [p[0] for p in pairs], [p[1] for p in pairs]
    stamps = [m["ts"] for m in messages if m["ts"]]
    start = min(stamps) if stamps else parse_time(result.get("start_time"))
    end = max(stamps) if stamps else (parse_time(result.get("end_time")) or start)
    sim = result.get("similarity") or 0.0
    substantive = len(body) >= MIN_BODY_CHARS
    if terms:
        # Short questions need one of their words; longer ones at least half, so "quantum lattice gauge
        # theory" isn't answered by a chunk that only says "theory".
        needed = 1 if len(terms) <= 2 else (len(terms) + 1) // 2
        floor = MIN_SIMILARITY_CROSS if cross else MIN_SIMILARITY
        # A generic word alone (ghar, dost, problem) doesn't answer "ghar pe kya ladai hui".
        relevant = sim >= floor and substantive and len(matched) >= needed and matcher.specific(matched)
    else:
        relevant = sim >= MIN_SIMILARITY_NO_TERMS and substantive
    coverage = len(matched) / len(terms) if terms else 0.0
    return {
        **result,
        "_body": body,
        "_norm": re.sub(r"\W+", " ", body.lower()).strip(),
        "_words": words_of(body),
        "_start": start,
        "_end": end,
        "_quote": best_quote(messages, keys, matcher),
        "matched_terms": shown,
        "relevant": relevant,
        "_coverage": coverage,
        "_score": sim + 0.2 * coverage - (0.2 if not substantive else 0.0),
        "_short": not substantive,
    }


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def dedupe(items):
    """Drops chunks that repeat a better-scored one (overlapping session windows, crossposts)."""
    kept, dropped = [], 0
    for it in sorted(items, key=lambda x: -x["_score"]):
        dup = False
        for k in kept:
            if it["_norm"] == k["_norm"]:
                dup = True
            elif it.get("node_id") and it.get("node_id") == k.get("node_id") and (
                    it["_norm"] in k["_norm"] or k["_norm"] in it["_norm"] or jaccard(it["_words"], k["_words"]) >= 0.6):
                dup = True
            elif jaccard(it["_words"], k["_words"]) >= 0.85:
                dup = True
            if dup:
                break
        if dup:
            dropped += 1
        else:
            kept.append(it)
    return kept, dropped


def in_window(it, start, end):
    """Overlap of the chunk's own message times (or stored window) with [start, end)."""
    a, b = it["_start"], it["_end"] or it["_start"]
    if a is None:
        return start is None and end is None
    return (start is None or b >= start) and (end is None or a < end)


# ─── Brief ────────────────────────────────────────────────────────────────────

def platform_name(p):
    return PLATFORM_NAMES.get(p or "", (p or "unknown").title())


def join_words(items, conj="and"):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} {conj} {items[-1]}"


def quoted(q):
    return f"“{q['text']}”" + (f" — {q['author']}" if q.get("author") else "")


def cite(it):
    bits = [month_label(it["_start"]), platform_name(it.get("platform"))]
    if it.get("title"):
        bits.append(f"“{clip(it['title'], 50)}”")
    return ", ".join(bits)


def focus_lines(text, is_hit, budget):
    """Up to `budget` chars of `text`: the lines for which is_hit(line) is true and their neighbours first,
    then the rest from the top, all in their original order with "…" marking gaps. Long chats often mention
    the topic far below their first lines, so a plain prefix would cut the evidence off."""
    lines = [l.rstrip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ""
    if sum(len(l) + 1 for l in lines) <= budget:
        return "\n".join(lines)
    hits = [i for i, l in enumerate(lines) if is_hit(l)]
    order = []
    for i in hits:   # the hit first, then its neighbours, so the budget keeps the most relevant lines
        order.append(i)
        for d in range(1, CONTEXT_LINES + 1):
            order.extend(j for j in (i - d, i + d) if 0 <= j < len(lines))
    order.extend(range(len(lines)))
    keep, used = set(), 0
    for i in order:
        if i in keep:
            continue
        line = lines[i]
        if used + len(line) + 1 > budget:
            if not keep:   # a single huge line: cut it
                return line[:budget - 1].rstrip() + GAP
            continue
        keep.add(i)
        used += len(line) + 1
    out, prev = [], -1
    for i in sorted(keep):
        if prev >= 0 and i != prev + 1:
            out.append(GAP)
        out.append(lines[i])
        prev = i
    return "\n".join(out)


def to_source(rank, it, matcher=None):
    q = it["_quote"]
    text = it.get("text") or ""
    if len(text) > SOURCE_TEXT_CHARS and matcher is not None and it["matched_terms"]:
        # The part of the chunk that mentions the question's words, not just its first lines.
        text = focus_lines(text[:MAX_SCAN_CHARS], lambda line: matcher.count(WORD_RE.findall(line.lower())) > 0,
                           SOURCE_TEXT_CHARS)
    return {
        "rank": rank,
        "node_id": it.get("node_id"),
        "title": it.get("title"),
        "platform": it.get("platform"),
        "date_start": iso(it["_start"]),
        "date_end": iso(it["_end"]),
        "similarity": it.get("similarity"),
        "snippet": q["text"] if q else (it.get("snippet") or clip(it["_body"], QUOTE_CHARS)),
        "text": text if len(text) <= SOURCE_TEXT_CHARS else text[:SOURCE_TEXT_CHARS - 1].rstrip() + "…",
        "people": it.get("people") or [],
        "relevant": it["relevant"],
        "matched_terms": it["matched_terms"],
    }


def timeline_label(it):
    """Title, or else the quote, shortened without dropping a negation (else the untitled marker)."""
    text = it.get("title") or (it["_quote"] or {}).get("text") or "(untitled)"
    return safe_clip(text, TITLE_CHARS, hard_max=TITLE_CHARS * 3) or "(untitled)"


def summary_points(relevant):
    """One point per (month, platform), chronological, each quoting its best source."""
    groups = {}
    for it in relevant:
        if not it["_quote"]:
            continue
        key = (it["_start"].strftime("%Y-%m") if it["_start"] else "", it.get("platform") or "")
        groups.setdefault(key, []).append(it)
    best = sorted(groups.items(), key=lambda kv: -max(x["_score"] for x in kv[1]))[:MAX_POINTS]
    points = []
    for (_, _), items in sorted(best, key=lambda kv: kv[0]):
        top = max(items, key=lambda x: x["_score"])
        more = f" (+{len(items) - 1} more)" if len(items) > 1 else ""
        points.append(f"{month_label(top['_start'])} · {platform_name(top.get('platform'))}: {quoted(top['_quote'])}{more}")
    return points


def scope_phrase(platforms, start, end):
    bits = []
    if platforms:
        bits.append("on " + join_words([platform_name(p) for p in platforms], "or"))
    if start or end:
        a = month_label(start) if start else "the beginning"
        b = month_label(end - dt.timedelta(seconds=1)) if end else "now"
        bits.append(f"between {a} and {b}" if a != b else f"in {a}")
    return (" " + " ".join(bits)) if bits else ""


def build_brief(question, results, limit=8, platforms=None, date_from=None, date_to=None, notes=None):
    """Grounded brief from retrieved results (dicts shaped like /api/search results)."""
    notes = list(notes or [])
    terms = content_terms(question)
    keys = [term_key(t) for t in terms]
    matcher = TermMatcher(terms, question_is_hinglish(question))
    items = [assess(r, terms, keys, matcher) for r in results]

    # Belt and braces: the prefilter already narrowed the search, this checks the chunks' own message dates.
    wanted = set(platforms or [])
    before = len(items)
    items = [it for it in items if (not wanted or it.get("platform") in wanted) and in_window(it, date_from, date_to)]
    if before - len(items):
        notes.append(f"Dropped {before - len(items)} retrieved chunk(s) whose messages fall outside the filters.")
    items, dropped = dedupe(items)
    if dropped:
        notes.append(f"Merged {dropped} overlapping or duplicate chunk(s).")
    short = sum(1 for it in items if it["_short"])
    if short:
        notes.append(f"{short} very short chunk(s) were not used as evidence.")

    relevant = [it for it in items if it["relevant"]]
    others = [it for it in items if not it["relevant"]]
    chosen = (relevant + others)[:limit]
    sources = [to_source(k, it, matcher) for k, it in enumerate(chosen, start=1)]
    shown_relevant = [it for it in chosen if it["relevant"]]
    scope = scope_phrase(platforms, date_from, date_to)
    terms_txt = join_words([f"“{t}”" for t in terms], "or")

    if not items:
        confidence = "low"
        answer = (f"No relevant info found in your memories{scope} for this question. "
                  "Try different wording, widen the date range or enable more platforms.")
        points = []
    elif not relevant:
        confidence = "low"
        answer = f"No relevant info found in your memories{scope}: none of the closest matches clearly address this question"
        answer += f" (none of them mention {terms_txt})." if terms else "."
        answer += " They are listed under Closest matches as leads only — they may be unrelated."
        points = []
    else:
        full = sum(1 for it in relevant if it["_coverage"] >= 0.999)
        confidence = "high" if terms and len(relevant) >= 3 and full >= 2 else "medium"
        dated = sorted((it for it in shown_relevant if it["_start"]), key=lambda x: x["_start"])
        plats = [platform_name(p) for p, _ in Counter(it.get("platform") for it in relevant).most_common()]
        n = len(relevant)
        one = n == 1
        what = (f"{'mentions' if one else 'mention'} {terms_txt}" if terms
                else f"{'is' if one else 'are'} semantically close to the question")
        span = ""
        if dated:
            a, b = month_label(dated[0]["_start"]), month_label(dated[-1]["_start"])
            span = f", from {a} to {b}" if a != b else f", in {a}"
        parts = [f"Found {n} memor{'y' if n == 1 else 'ies'}{scope} that {what}, on {join_words(plats)}{span}."]

        quotable = [it for it in dated if it["_quote"]]
        if TREND_RE.search(question) and len({month_label(it["_start"]) for it in quotable}) >= 2:
            first, last = quotable[0], quotable[-1]
            parts.append(f"Earliest ({cite(first)}): {quoted(first['_quote'])}. Latest ({cite(last)}): {quoted(last['_quote'])}.")
        else:
            top = next((it for it in shown_relevant if it["_quote"]), None)
            if top:
                parts.append(f"Strongest match ({cite(top)}): {quoted(top['_quote'])}.")
        months = Counter(month_label(it["_start"]) for it in relevant if it["_start"])
        if months and n >= 3:
            peak, count = months.most_common(1)[0]
            if count >= 2 and count / n >= 0.4 and len(months) > 1:
                parts.append(f"Most of it is from {peak} ({count} of {n}).")
        if confidence == "medium":
            if n < 3:
                parts.append(f"Evidence is limited to {n} source{'s' if n > 1 else ''}, so treat this as partial.")
            elif terms and not full:
                parts.append("No single source covers every part of the question.")
            else:
                parts.append("Read the sources before drawing conclusions.")
        answer = " ".join(parts)
        points = summary_points(shown_relevant)

    timeline_items = shown_relevant or chosen
    timeline = [
        {"date": iso(it["_start"]), "label": timeline_label(it),
         "node_id": it.get("node_id"), "platform": it.get("platform"), "source": chosen.index(it) + 1}
        for it in sorted((it for it in timeline_items if it["_start"]), key=lambda x: x["_start"])
    ][:MAX_TIMELINE]

    return {
        "answer": answer,
        "confidence": confidence,
        "summary_points": points,
        "timeline": timeline,
        "sources": sources,
        "notes": notes,
        "evidence": {"retrieved": len(results), "considered": len(items), "relevant": len(relevant), "terms": terms},
    }
