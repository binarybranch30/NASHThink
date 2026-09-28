"""Written answers for Ask Sarthink: a local Llama rewrites already-retrieved evidence into prose.

memory_brief.build_brief() decides which retrieved chunks are evidence. This module only turns those
sources into a prompt, streams the reply from a llama.cpp server on 127.0.0.1 (profiles in llm_config.py),
and cleans its citations. The model sees nothing but the question and the numbered evidence sources, and
is told to cite them as [n], where n is the source's rank in the /api/ask response.

Nothing leaves the machine: the only network calls are to the loopback llama.cpp servers.
"""
import asyncio
import datetime as dt
import hashlib
import json
import re
import threading
import time
from pathlib import Path

import httpx

import hinglish
import llm_config
import memory_brief

CHARS_PER_TOKEN_EN = 3.0     # fallback estimates when the server can't count (/tokenize); measured: English
CHARS_PER_TOKEN_HI = 2.0     # ~3.5 chars/token, Hinglish chat with emoji ~2
TEMPLATE_TOKENS = 40         # chat template around the two messages
MIN_SOURCE_CHARS = 200
STATUS_TTL_S = 5.0
UPSTREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=900.0, write=30.0, pool=5.0)   # 8B prompt processing is slow
CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
GAP = memory_brief.GAP
NO_INFO = "NO_RELEVANT_INFO"   # the model's reply when the sources don't answer the question
NO_INFO_MESSAGE = "No relevant info found in your memories for this question."


class LlmError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


# ─── Prompt ──────────────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Sarthink, a private assistant that answers questions about the user's own chat history. Today is {today}.
The user is {owner}. In the sources, messages written by {names} are the user's own: write about them as "you". Everyone else in the sources is another person: never present their words, plans or feelings as the user's.

Rules:
- Use ONLY the numbered sources. Never add facts, events, feelings, relationships or dates that are not in them.
- Put the source number in square brackets after each claim, like [2] or [1][3]. Only cite numbers that exist.
- If the sources only partly answer the question, say what they show and what is missing. If they do not answer it at all, reply with exactly NO_RELEVANT_INFO and nothing else.
- Write 1 to 3 short paragraphs of clear, natural prose that directly answers the question. Use a short "- " bullet list only for several distinct items, and never repeat the paragraphs as bullets. You may **bold** a few key phrases. No headings, no summary at the end, and do not start with "Based on the sources".
- Back each claim with a short exact quote (3 to 12 words) of the original message in double quotes, copied character for character, then the source number, like: you said you couldn't sleep ("neend nahi aa rha") [2].
- Always write in English, even when the question or the messages are in Hinglish. Keep quoted words exactly as written; add a short English meaning in brackets after a Hinglish quote when it helps.
- Times are UTC. Mention months or years when they help show when something happened.{glossary}{heavy}"""

GLOSSARY = """

The messages are casual chat, often Hinglish (Hindi written in English letters) with shorthand: h / hai = is; ni / nhi / nahi / na = not; mat = don't; rha / rhi / rhe = -ing (raha); m / me / mein = in; k / ke = of; kya = what; kyu = why; bhai / bro / yaar = a way of addressing a friend; bc / bkl / mc = swear words, not a topic; 🤡 and 😭 usually mean self-mockery or distress. A "nahi", "ni" or "mat" reverses the meaning of the sentence: never drop it."""

HEAVY = """

Some sources contain statements about wanting to die, suicide or self-harm. Report any such statement plainly, with an exact quote of the original words. Never soften it, reinterpret it or reword it as something else (for example as worry about someone's health, being tired or joking)."""

# Death-wish / self-harm wording (English and Hinglish), whole words. Only changes the instructions above.
HEAVY_RE = re.compile(r"(?<![^\W_])(?:suicid\w*|kill(?:ing)? myself|end(?:ing)? (?:my life|it all)|wan(?:t|na)(?: to)? die|"
                      r"self[- ]?harm\w*|cut(?:ting)? myself|no reason to live|don'?t want to live|khudkushi|aatmahatya|"
                      r"mar (?:ja(?:u|un|unga|ungi|au|ata|ati|ana|na)|jaa(?:u|un|unga|ungi|ta|ti|na))|"
                      r"marna (?:h|hai|chahta|chahti|chahiye)|jeena nahi|zinda nahi rehna)(?![^\W_])", re.I)


def owner_names(master, aliases):
    """How the user appears in chunk text: the persona chunk_builder writes, plus platform handles."""
    names = [master] if master else []
    for a in sorted(aliases or ()):
        if a and not a.isdigit() and "@" not in a and a.lower() not in {n.lower() for n in names}:
            names.append(a)
    return names[:8]


def source_header(s):
    bits = [s.get("title") or "Untitled conversation"]
    if s.get("platform"):
        bits.append(s["platform"].capitalize())
    a, b = (s.get("date_start") or "")[:10], (s.get("date_end") or "")[:10]
    if a:
        bits.append(a if not b or b == a else f"{a} to {b}")
    people = [p for p in (s.get("people") or []) if p][:6]
    if people:
        bits.append("people: " + ", ".join(people))
    return f"[{s['rank']}] " + " · ".join(bits)


def excerpt(text, terms, budget):
    """Up to `budget` chars of `text`, preferring the lines around mentions of `terms`, in their original order."""
    keys = [t.lower() for t in terms or () if t]
    return memory_brief.focus_lines(text, lambda line: any(k in line.lower() for k in keys), budget)


def evidence_sources(brief, profile, limit=None):
    n = int(profile["max_sources"]) if limit is None else min(int(limit), int(profile["max_sources"]))
    return [s for s in brief.get("sources") or () if s.get("relevant")][:n]


def allowed_example(sources):
    return sources[min(1, len(sources) - 1)]["rank"] if sources else 1


def should_write(brief):
    """Only write prose over real evidence: a weak brief would make the model guess."""
    return brief.get("confidence") != "low" and any(s.get("relevant") for s in brief.get("sources") or ())


def is_hinglish_text(text):
    """True when at least a fifth of the text's messages read as Hinglish."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return False
    hits = sum(1 for l in lines if hinglish.looks_hinglish(hinglish.words(l)))
    return hits * 5 >= len(lines)


def chars_per_token(texts):
    return CHARS_PER_TOKEN_HI if any(is_hinglish_text(t) for t in texts) else CHARS_PER_TOKEN_EN


def default_source_chars(question, brief, profile):
    """A first guess at how many chars each source can have so the prompt fits profile["prompt_tokens"]."""
    sources = evidence_sources(brief, profile)
    cpt = chars_per_token([s.get("text") or "" for s in sources])
    budget = int(profile["prompt_tokens"]) - 750 - len(question) // 2   # instructions + glossary: ~2.4k chars, ~700 tokens
    return max(MIN_SOURCE_CHARS, min(int(profile["source_chars"]), int(budget * cpt) // max(1, len(sources))))


def build_messages(question, brief, profile, owner=("", ()), today=None, source_chars=None, max_sources=None):
    """Chat messages for /v1/chat/completions. Sources keep their /api/ask rank so [n] matches the UI.
    `source_chars` caps each source's excerpt (default: an estimate from profile["prompt_tokens"])."""
    master, aliases = owner
    names = owner_names(master, aliases) or ["the user"]
    sources = evidence_sources(brief, profile, max_sources)
    terms = (brief.get("evidence") or {}).get("terms") or []
    per_source = source_chars or default_source_chars(question, brief, profile)
    blocks = []
    for s in sources:
        body = excerpt(s.get("text") or s.get("snippet") or "", list(s.get("matched_terms") or []) + list(terms), per_source)
        blocks.append(f"{source_header(s)}\n{body}")
    texts = [b.split("\n", 1)[-1] for b in blocks]
    system = SYSTEM_PROMPT.format(
        today=(today or dt.datetime.now(dt.timezone.utc).date()).isoformat(),
        owner=master or "the user",
        names=", ".join(f'"{n}"' for n in names),
        glossary=GLOSSARY if any(is_hinglish_text(t) for t in texts) else "",
        heavy=HEAVY if any(HEAVY_RE.search(t) for t in texts) else "",
    )
    user = ("Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\n"
            "Answer in English by describing what these conversations show, speaking to the user as \"you\" "
            "(for example \"You asked…\", \"You told …\"). Report what happened; do not give advice, and do not repeat "
            "the same points as a list at the end. Back each claim with a short exact quote in double quotes. End every "
            f"sentence or bullet with the number of the source it comes from, like [{allowed_example(sources)}].")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], [s["rank"] for s in sources]


async def count_tokens(profile, messages, transport=None):
    """Exact prompt size from the llama.cpp server's own tokenizer (POST /tokenize); None if unavailable
    (always for hosted models, which have no such endpoint: the size is estimated instead)."""
    if llm_config.is_remote(profile):
        return None
    content = "\n".join(m["content"] for m in messages)
    try:
        async with httpx.AsyncClient(base_url=profile["url"], timeout=10.0, transport=transport) as client:
            r = await client.post("/tokenize", json={"content": content})
            if r.status_code != 200:
                return None
            return len(r.json().get("tokens") or []) + TEMPLATE_TOKENS
    except (httpx.HTTPError, ValueError):
        return None


def estimate_tokens(messages):
    return int(sum(len(m["content"]) for m in messages) / chars_per_token([messages[1]["content"]])) + TEMPLATE_TOKENS


async def fit_messages(question, brief, profile, owner=("", ()), transport=None, today=None):
    """build_messages() shrunk until the prompt fits profile["prompt_tokens"], measured by the server's own
    tokenizer (or estimated if it can't count): shorter excerpts first, then fewer (lowest-ranked) sources.
    -> (messages, allowed, prompt_tokens, counted)."""
    target = int(profile["prompt_tokens"])
    chars = default_source_chars(question, brief, profile)
    n_src = len(evidence_sources(brief, profile))
    counted = True
    while True:
        messages, allowed = build_messages(question, brief, profile, owner, today, source_chars=chars, max_sources=n_src)
        n = await count_tokens(profile, messages, transport) if counted else None
        if n is None:
            counted = False
            n = estimate_tokens(messages)
        if n <= target:
            break
        if chars > MIN_SOURCE_CHARS:
            chars = max(MIN_SOURCE_CHARS, int(chars * target / n * 0.95))
        elif n_src > 1:
            n_src -= 1
        else:
            break
    return messages, allowed, n, counted


# ─── Output ──────────────────────────────────────────────────────────────────────────────────────

def sanitize_citations(text, allowed):
    """[2], [1, 3] -> [2], [1][3]; numbers that aren't evidence sources are dropped."""
    allowed = set(allowed)

    def fix(m):
        nums = [int(n) for n in re.split(r"\s*[,;]\s*", m.group(1))]
        return "".join(f"[{n}]" for n in nums if n in allowed)

    text = CITE_RE.sub(fix, text)
    text = re.sub(r"[,;]\s*((?:\[\d+\])+)", r" \1", text)        # "studying, [1]." -> "studying [1]."
    return re.sub(r"[ \t]+([.,;:!?])", r"\1", text).strip()


QUOTE_RE = re.compile(r'"([^"\n]{2,300})"|“([^”\n]{2,300})”')
QUOTE_CITE_RE = re.compile(r"^[^\n\"“]{0,80}?\[(\d{1,3})\]")


def _norm(text):
    return " ".join(hinglish.normalize(w) for w in hinglish.words(text))


def verify_quotes(text, sources):
    """Checks every quoted phrase in the answer against the source it cites (or any source if uncited).
    -> [{"text", "ok", "source"}]; quotes of fewer than two words aren't checked."""
    by_rank = {s["rank"]: _norm(s.get("text") or "") for s in sources}
    out = []
    for m in QUOTE_RE.finditer(text):
        quote = m.group(1) or m.group(2)
        parts = [p for p in re.split(r"\s*(?:…|\.\.\.)\s*", quote) if p.strip()]
        if sum(len(hinglish.words(p)) for p in parts) < 2:
            continue
        c = QUOTE_CITE_RE.match(text[m.end():])
        rank = int(c.group(1)) if c else None
        pool = [by_rank[rank]] if rank in by_rank else list(by_rank.values())
        ok = all(any(_norm(p) in src for src in pool) for p in parts)
        out.append({"text": quote, "ok": ok, "source": rank})
    return out


def cited(text):
    return sorted({int(n) for m in CITE_RE.finditer(text) for n in re.split(r"\s*[,;]\s*", m.group(1))})


def _sentinel_probe(text):
    """The start of a reply, normalised for comparing with NO_INFO ("**No relevant info**" -> "NO_RELEVANT_INFO")."""
    return re.sub(r"[\s-]+", "_", text.lstrip(" \t\n*_`\"'").upper())


def says_no_info(text):
    """True when the reply is (or starts with) the NO_INFO sentinel."""
    return _sentinel_probe(text).startswith(NO_INFO)


def may_become_no_info(text):
    """True while a partial reply could still turn out to be the sentinel, so its tokens are held back."""
    probe = _sentinel_probe(text)
    return NO_INFO.startswith(probe[:len(NO_INFO)])


def ungrounded_reason(text, quotes):
    """Why a finished answer can't be trusted, or None: it cites no source, most of its quotes aren't in the
    sources, or the model gave up part-way (the sentinel appears after some text)."""
    if NO_INFO in _sentinel_probe(text):
        return "model_unsure"
    if not cited(text):
        return "no_citations"
    bad = sum(1 for q in quotes if not q["ok"])
    if len(quotes) >= 2 and bad * 2 > len(quotes):
        return "unverified_quotes"
    return None


# ─── llama.cpp ───────────────────────────────────────────────────────────────────────────────────

async def stream_completion(profile, messages, transport=None):
    """Yields ("token", text) for each streamed delta, then ("usage", {...}) from the final chunk.

    Closing the generator closes the upstream request, which makes llama-server stop generating."""
    remote = llm_config.is_remote(profile)
    body = {
        "messages": messages,
        "stream": True,
        "max_tokens": int(profile["max_tokens"]),
        "temperature": float(profile["temperature"]),
    }
    headers = {}
    if remote:
        key = llm_config.api_key(profile)
        if not key:
            raise LlmError("llm_no_key", f"{profile['model_name']} needs an API key: add {profile['api_key_env']}=... to the .env file.")
        body["model"] = profile["api_model"]
        body["stream_options"] = {"include_usage": True}
        headers["Authorization"] = f"Bearer {key}"
        path, timeout = "/chat/completions", httpx.Timeout(connect=10.0, read=float(profile.get("timeout_s", 90)), write=30.0, pool=10.0)
    else:
        body["cache_prompt"] = False
        path, timeout = "/v1/chat/completions", UPSTREAM_TIMEOUT
    name = profile["model_name"] if remote else f"The {profile['label']} model"
    try:
        async with httpx.AsyncClient(base_url=profile["url"], timeout=timeout, transport=transport) as client:
            async with client.stream("POST", path, json=body, headers=headers) as r:
                if r.status_code != 200 and remote:
                    detail = (await r.aread()).decode("utf-8", "replace")[:200]
                    raise remote_error(profile, r.status_code, detail)
                if r.status_code == 503:
                    raise LlmError("llm_loading", f"The {profile['label']} model is still loading. Try again in a moment.")
                if r.status_code != 200:
                    detail = (await r.aread()).decode("utf-8", "replace")[:200]
                    raise LlmError("llm_failed", f"The {profile['label']} model answered HTTP {r.status_code}: {detail}")
                usage = {}
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    for choice in chunk.get("choices") or ():
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            yield ("token", piece)
                    if chunk.get("timings"):
                        usage["timings"] = chunk["timings"]
                    if chunk.get("usage"):
                        usage["usage"] = chunk["usage"]
                    if chunk.get("model"):
                        usage["model"] = chunk["model"]
                yield ("usage", usage)
    except httpx.ConnectError:
        if remote:
            raise LlmError("llm_offline", f"Couldn't reach {profile['provider']} (is this server online?).")
        raise LlmError("llm_offline", f"The {profile['label']} model isn't running. Start it with: scripts/llm.sh start {profile['name']}")
    except httpx.TimeoutException:
        raise LlmError("llm_failed", f"{name} took too long to answer.")
    except httpx.HTTPError as e:
        raise LlmError("llm_failed", f"{name} failed: {type(e).__name__}")


def remote_error(profile, status, detail):
    """A hosted API's HTTP error as a plain message (DeepSeek's documented codes). The key is never included."""
    who = profile["model_name"]
    if status == 401:
        return LlmError("llm_no_key", f"{profile['provider']} rejected the API key. Check {profile['api_key_env']} in the .env file.")
    if status == 402:
        return LlmError("llm_failed", f"The {profile['provider']} account has no balance left.")
    if status == 429:
        return LlmError("llm_failed", f"{profile['provider']} is rate-limiting requests. Try again in a moment.")
    if status in (500, 503):
        return LlmError("llm_failed", f"{profile['provider']} is busy or unavailable right now. Try again, or use a local model.")
    return LlmError("llm_failed", f"{who} answered HTTP {status}: {detail[:120]}")


class LlmStatus:
    """Which profiles have a llama.cpp server answering. Probes are short and cached for a few seconds."""

    def __init__(self, profiles=None, transport=None):
        self.profiles = profiles if profiles is not None else llm_config.load_profiles()
        self.transport = transport
        self._lock = threading.Lock()
        self._cache = (0.0, None)
        self._remote_lock = threading.Lock()
        self._remote_inflight, self._remote_result = {}, {}

    def _probe(self, p):
        if llm_config.is_remote(p):
            return self._probe_remote(p)
        base = {"label": p["label"], "model_name": p["model_name"], "description": p["description"], "model_file": Path(p["model_path"]).is_file(),
                "start": f"scripts/llm.sh start {p['name']}", "remote": False}
        try:
            with httpx.Client(base_url=p["url"], timeout=1.5, transport=self.transport) as c:
                r = c.get("/health")
                if r.status_code == 503:
                    return {**base, "state": "loading"}
                if r.status_code != 200:
                    return {**base, "state": "offline"}
                model = None
                try:
                    data = c.get("/v1/models").json().get("data") or []
                    model = data[0].get("id") if data else None
                except (httpx.HTTPError, ValueError, AttributeError):
                    pass
                return {**base, "state": "online", "model": model}
        except httpx.HTTPError:
            return {**base, "state": "offline"}

    def _probe_remote(self, p):
        """Online when the API accepts the key (GET /models: no tokens used). The key itself is never returned."""
        base = {"label": p["label"], "model_name": p["model_name"], "description": p["description"], "model_file": True,
                "start": "", "remote": True, "provider": p["provider"]}
        key = llm_config.api_key(p)
        if not key:
            return {**base, "state": "offline", "reason": "no_key"}
        reason = self._remote_check(p["url"], key)
        if reason:
            return {**base, "state": "offline", "reason": reason}
        return {**base, "state": "online", "model": p["api_model"]}

    def _remote_check(self, url, key):
        """None when GET /models accepts the key, else a reason. One request per (API, key) per status round."""
        slot = (url, hashlib.sha256(key.encode()).hexdigest())
        with self._remote_lock:
            at, result = self._remote_result.get(slot, (0.0, None))
            if time.monotonic() - at < STATUS_TTL_S:
                return result
            ev = self._remote_inflight.get(slot)
            owner = ev is None
            if owner:
                ev = self._remote_inflight[slot] = threading.Event()
        if not owner:
            ev.wait(6)
            return self._remote_result.get(slot, (0.0, "unreachable"))[1]
        try:
            with httpx.Client(base_url=url, timeout=3.0, transport=self.transport) as c:
                r = c.get("/models", headers={"Authorization": f"Bearer {key}"})
            result = None if r.status_code == 200 else ("bad_key" if r.status_code == 401 else f"http_{r.status_code}")
        except httpx.HTTPError:
            result = "unreachable"
        with self._remote_lock:
            self._remote_result[slot] = (time.monotonic(), result)
            del self._remote_inflight[slot]
        ev.set()
        return result

    def status(self):
        with self._lock:
            at, value = self._cache
            if value is not None and time.monotonic() - at < STATUS_TTL_S:
                return value
            # Probed in parallel (hosted APIs are a network round trip away); remote profiles sharing an API and key
            # share one check.
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max(1, len(self.profiles))) as pool:
                futures = {name: pool.submit(self._probe, p) for name, p in self.profiles.items()}
                value = {name: f.result() for name, f in futures.items()}
            self._cache = (time.monotonic(), value)
            return value

    def invalidate(self):
        with self._lock:
            self._cache = (0.0, None)


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def answer_events(question, brief, profile, owner, is_disconnected, cancelled, transport=None, heartbeat_s=5.0):
    """SSE strings after the brief: status heartbeats until the first token, tokens, then done/skipped/error.

    `cancelled` is an asyncio.Event set when a newer question for the same model supersedes this one."""
    started = time.perf_counter()
    if not should_write(brief):
        yield sse("skipped", {"code": "no_relevant_info", "message": NO_INFO_MESSAGE})
        return
    messages, allowed, prompt_tokens, _ = await fit_messages(question, brief, profile, owner, transport)
    queue = asyncio.Queue()

    async def pump():
        gen = stream_completion(profile, messages, transport)
        try:
            async for item in gen:
                await queue.put(item)
            await queue.put(("end", None))
        except LlmError as e:
            await queue.put(("error", {"code": e.code, "message": e.message}))
        except asyncio.CancelledError:
            raise
        except Exception as e:   # never leave the client hanging
            await queue.put(("error", {"code": "llm_failed", "message": f"{type(e).__name__}: {e}"}))
        finally:
            await gen.aclose()

    task = asyncio.create_task(pump())
    parts, usage, stage = [], {}, "reading"
    held = []   # the first tokens, kept back until they can't be the NO_INFO sentinel
    try:
        yield sse("status", {"stage": stage, "elapsed_s": 0, "sources": len(allowed), "profile": profile["name"]})
        while True:
            try:
                kind, value = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
            except asyncio.TimeoutError:
                if cancelled.is_set():
                    yield sse("error", {"code": "llm_superseded", "message": "A newer question replaced this one."})
                    return
                if await is_disconnected():
                    return
                yield sse("status", {"stage": stage, "elapsed_s": int(time.perf_counter() - started),
                                     "sources": len(allowed), "profile": profile["name"]})
                continue
            if cancelled.is_set():
                yield sse("error", {"code": "llm_superseded", "message": "A newer question replaced this one."})
                return
            if kind == "token":
                if stage == "reading":
                    stage = "writing"
                parts.append(value)
                if held is None:
                    yield sse("token", {"text": value})
                    continue
                held.append(value)
                sofar = "".join(held)
                if says_no_info(sofar):
                    yield sse("skipped", {"code": "no_relevant_info", "message": NO_INFO_MESSAGE})
                    return
                if not may_become_no_info(sofar):
                    yield sse("token", {"text": sofar})
                    held = None
            elif kind == "usage":
                usage = value
            elif kind == "error":
                yield sse("error", value)
                return
            elif kind == "end":
                break
        if held is not None and _sentinel_probe("".join(held)):   # a short reply that is (a prefix of) the sentinel
            yield sse("skipped", {"code": "no_relevant_info", "message": NO_INFO_MESSAGE})
            return
        text = sanitize_citations("".join(parts), allowed)
        if not text:
            yield sse("error", {"code": "llm_failed", "message": "The model returned an empty answer."})
            return
        timings = usage.get("timings") or {}
        quotes = verify_quotes(text, [s for s in brief.get("sources") or () if s.get("rank") in allowed])
        reason = ungrounded_reason(text, quotes)
        yield sse("done", {
            "text": text,
            "grounded": reason is None,
            "ungrounded_reason": reason,
            "cited": cited(text),
            "quotes": quotes,
            "unverified": sum(1 for q in quotes if not q["ok"]),
            "profile": profile["name"],
            "model": usage.get("model") or profile.get("alias"),
            "tokens": timings.get("predicted_n"),
            "prompt_tokens": timings.get("prompt_n") or prompt_tokens,
            "took_ms": int((time.perf_counter() - started) * 1000),
        })
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
