"""Written answers for Ask Sarthink: a local Llama rewrites already-retrieved evidence into prose.

memory_brief.build_brief() decides which retrieved chunks are evidence. This module only turns those
sources into a prompt, streams the reply from a llama.cpp server on 127.0.0.1 (profiles in llm_config.py),
and cleans its citations. The model sees nothing but the question and the numbered evidence sources, and
is told to cite them as [n], where n is the source's rank in the /api/ask response.

Nothing leaves the machine: the only network calls are to the loopback llama.cpp servers.
"""
import asyncio
import datetime as dt
import json
import re
import threading
import time
from pathlib import Path

import httpx

import llm_config

CHARS_PER_TOKEN = 3.0        # conservative for Hinglish, emoji and names
PROMPT_OVERHEAD_TOKENS = 700  # system prompt, headers and chat template
CONTEXT_LINES = 2            # lines kept on each side of a line that mentions the question's words
STATUS_TTL_S = 5.0
UPSTREAM_TIMEOUT = httpx.Timeout(connect=3.0, read=900.0, write=30.0, pool=5.0)   # 8B prompt processing is slow
CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3})*)\]")
GAP = "…"


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
- If the sources only partly answer the question, say what they show and what is missing. If they do not answer it, say so plainly in one sentence.
- Write 1 to 3 short paragraphs of clear, natural prose that directly answers the question. Use a short "- " bullet list only for several distinct items, and never repeat the paragraphs as bullets. You may **bold** a few key phrases. No headings, no summary at the end, and do not start with "Based on the sources".
- Answer in the language of the question. When quoting Hindi or Hinglish, keep the words exactly as written.
- Times are UTC. Mention months or years when they help show when something happened."""


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
    lines = [l.rstrip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        return ""
    if sum(len(l) + 1 for l in lines) <= budget:
        return "\n".join(lines)
    keys = [t.lower() for t in terms or () if t]
    hits = [i for i, l in enumerate(lines) if any(k in l.lower() for k in keys)]
    order = []
    for i in hits:   # the hit first, then its neighbours, so the budget keeps the most relevant lines
        order.append(i)
        for d in range(1, CONTEXT_LINES + 1):
            order.extend(j for j in (i - d, i + d) if 0 <= j < len(lines))
    order.extend(range(len(lines)))   # then the rest from the top
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


def evidence_sources(brief, profile):
    return [s for s in brief.get("sources") or () if s.get("relevant")][:int(profile["max_sources"])]


def allowed_example(sources):
    return sources[min(1, len(sources) - 1)]["rank"] if sources else 1


def should_write(brief):
    """Only write prose over real evidence: a weak brief would make the model guess."""
    return brief.get("confidence") != "low" and any(s.get("relevant") for s in brief.get("sources") or ())


def build_messages(question, brief, profile, owner=("", ()), today=None):
    """Chat messages for /v1/chat/completions. Sources keep their /api/ask rank so [n] matches the UI."""
    master, aliases = owner
    names = owner_names(master, aliases) or ["the user"]
    sources = evidence_sources(brief, profile)
    terms = (brief.get("evidence") or {}).get("terms") or []
    budget_tokens = int(profile["ctx"]) - int(profile["max_tokens"]) - PROMPT_OVERHEAD_TOKENS - len(question) // 2
    total_chars = max(800, int(budget_tokens * CHARS_PER_TOKEN))
    per_source = min(int(profile["source_chars"]), total_chars // max(1, len(sources)))
    blocks = []
    for s in sources:
        body = excerpt(s.get("text") or s.get("snippet") or "", list(s.get("matched_terms") or []) + list(terms), per_source)
        blocks.append(f"{source_header(s)}\n{body}")
    system = SYSTEM_PROMPT.format(
        today=(today or dt.datetime.now(dt.timezone.utc).date()).isoformat(),
        owner=master or "the user",
        names=", ".join(f'"{n}"' for n in names),
    )
    user = ("Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\n"
            "Answer the question by describing what these conversations show, speaking to the user as \"you\" "
            "(for example \"You asked…\", \"You told …\"). Report what happened; do not give advice, and do not repeat "
            "the same points as a list at the end. End every "
            f"sentence or bullet with the number of the source it comes from, like [{allowed_example(sources)}].")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], [s["rank"] for s in sources]


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


def cited(text):
    return sorted({int(n) for m in CITE_RE.finditer(text) for n in re.split(r"\s*[,;]\s*", m.group(1))})


# ─── llama.cpp ───────────────────────────────────────────────────────────────────────────────────

async def stream_completion(profile, messages, transport=None):
    """Yields ("token", text) for each streamed delta, then ("usage", {...}) from the final chunk.

    Closing the generator closes the upstream request, which makes llama-server stop generating."""
    body = {
        "messages": messages,
        "stream": True,
        "max_tokens": int(profile["max_tokens"]),
        "temperature": float(profile["temperature"]),
        "cache_prompt": False,
    }
    try:
        async with httpx.AsyncClient(base_url=profile["url"], timeout=UPSTREAM_TIMEOUT, transport=transport) as client:
            async with client.stream("POST", "/v1/chat/completions", json=body) as r:
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
        raise LlmError("llm_offline", f"The {profile['label']} model isn't running. Start it with: scripts/llm.sh start {profile['name']}")
    except httpx.TimeoutException:
        raise LlmError("llm_failed", f"The {profile['label']} model took too long to answer.")
    except httpx.HTTPError as e:
        raise LlmError("llm_failed", f"The {profile['label']} model failed: {type(e).__name__}: {e}")


class LlmStatus:
    """Which profiles have a llama.cpp server answering. Probes are short and cached for a few seconds."""

    def __init__(self, profiles=None, transport=None):
        self.profiles = profiles if profiles is not None else llm_config.load_profiles()
        self.transport = transport
        self._lock = threading.Lock()
        self._cache = (0.0, None)

    def _probe(self, p):
        base = {"label": p["label"], "model_name": p["model_name"], "description": p["description"], "model_file": Path(p["model_path"]).is_file(),
                "start": f"scripts/llm.sh start {p['name']}"}
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

    def status(self):
        with self._lock:
            at, value = self._cache
            if value is not None and time.monotonic() - at < STATUS_TTL_S:
                return value
            value = {name: self._probe(p) for name, p in self.profiles.items()}
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
        yield sse("skipped", {"code": "weak_evidence",
                              "message": "The evidence is too weak to write an answer from, so only the sources are shown."})
        return
    messages, allowed = build_messages(question, brief, profile, owner)
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
                yield sse("token", {"text": value})
            elif kind == "usage":
                usage = value
            elif kind == "error":
                yield sse("error", value)
                return
            elif kind == "end":
                break
        text = sanitize_citations("".join(parts), allowed)
        if not text:
            yield sse("error", {"code": "llm_failed", "message": "The model returned an empty answer."})
            return
        timings = usage.get("timings") or {}
        yield sse("done", {
            "text": text,
            "cited": cited(text),
            "profile": profile["name"],
            "model": usage.get("model") or profile.get("alias"),
            "tokens": timings.get("predicted_n"),
            "prompt_tokens": timings.get("prompt_n"),
            "took_ms": int((time.perf_counter() - started) * 1000),
        })
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
