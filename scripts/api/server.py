"""Local FastAPI server for Sarthink: semantic search API + the 3D memory graph UI.

Everything runs on this machine. Queries are embedded on CPU with the model recorded in the
table's metadata (see scripts/semantic/search.py); nothing is sent to a hosted service.

Usage:
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py            # http://127.0.0.1:8000
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py --table topics --port 8000

Endpoints:
    GET  /                    sarthink_graph.html
    GET  /api/health          index/model status (never loads the model) + whether the memory DB exists
    POST /api/search          {"query": "...", "limit": 10} -> structured results
    POST /api/ask             {"question": "...", "limit": 8, "platforms": [...], "date_from": ..., "date_to": ...}
                              -> evidence-first memory brief (deterministic; see memory_brief.py)
    POST /api/ask/stream      same body + "profile": "quick"|"best" -> text/event-stream: the brief, then an answer
                              written by a local Llama over the evidence (see answer_writer.py, scripts/llm.sh)
    GET  /api/llm             which local Llama profiles are running
    GET  /api/thread/T_<id>   indexed conversation chunks of one graph thread (no model load)
    GET  /api/person/U_<id>   "Your history with X": counts, activity, cited brief (read-only SQLite, no model load);
                              /conversations?offset&limit and /messages?thread&cursor&limit page the full history
    GET  /api/insights        read-only activity aggregates from the SQLite memory DB (see insights.py);
                              optional ?platforms=a,b&date_from=...&date_to=...&top=10
    GET  /processed_data/graph/cosmograph_{nodes,edges}.csv   graph data for the UI
    GET  /assets/nashthink-{mark,icon}.png                    the NASH Think logo
"""
import argparse
import hmac
import asyncio
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(str(REPO_ROOT / "scripts" / "semantic"))

import answer_writer  # noqa: E402
import insights  # noqa: E402
import llm_config  # noqa: E402
import people  # noqa: E402
import reminders  # noqa: E402
import memory_brief  # noqa: E402
import search  # noqa: E402
import workspaces  # noqa: E402
from embedding_config import LANCEDB_PATH  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MAX_LIMIT = 50
MAX_QUERY_CHARS = 500
SNIPPET_CHARS = 320
CONTEXT_CHUNKS = 8          # chunks returned per thread by /api/thread
CONTEXT_TEXT_CHARS = 2000   # full-text cap per returned chunk
THREAD_NODE_RE = re.compile(r"^T_(\d{1,18})$")
ASK_DEFAULT_LIMIT = 8
ASK_MAX_LIMIT = 20
ASK_CANDIDATES = 60         # chunks retrieved per question before evidence filtering + dedupe

GRAPH_HTML = REPO_ROOT / "sarthink_graph.html"
GRAPH_DIR = REPO_ROOT / "processed_data" / "graph"
# Only these files are served from processed_data/; everything else there stays private.
GRAPH_FILES = ("cosmograph_nodes.csv", "cosmograph_edges.csv")
ASSETS_DIR = REPO_ROOT / "assets"
ASSET_FILES = {"nashthink-mark.png": "image/png", "nashthink-icon.png": "image/png"}   # the UI's logo


class ApiError(Exception):
    """An error with a stable machine-readable `code` the UI can switch on."""

    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=MAX_QUERY_CHARS)
    limit: int = Field(search.DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        return v


class AskRequest(BaseModel):
    question: str = Field(..., max_length=MAX_QUERY_CHARS)
    limit: int = Field(ASK_DEFAULT_LIMIT, ge=1, le=ASK_MAX_LIMIT)
    platforms: Optional[List[str]] = Field(None, max_length=32)
    date_from: Optional[str] = Field(None, max_length=40)
    date_to: Optional[str] = Field(None, max_length=40)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("question must not be empty")
        return v

    @field_validator("platforms")
    @classmethod
    def platforms_valid(cls, v):
        if v is None:
            return None
        out = sorted({p.strip().lower() for p in v if p and p.strip()})
        bad = [p for p in out if not search.PLATFORM_RE.match(p)]
        if bad:
            raise ValueError(f"invalid platform name: {bad[0]!r}")
        return out or None

    @field_validator("date_from", "date_to")
    @classmethod
    def date_valid(cls, v):
        if v is None or not v.strip():
            return None
        try:
            memory_brief.parse_bound(v)
        except ValueError:
            raise ValueError("must be an ISO date (YYYY-MM-DD) or datetime")
        return v.strip()


class AskStreamRequest(AskRequest):
    profile: str = Field("best", max_length=20)

    @field_validator("profile")
    @classmethod
    def profile_known(cls, v):
        v = v.strip().lower()
        if v not in llm_config.DEFAULT_PROFILES:
            raise ValueError(f"must be one of: {', '.join(llm_config.DEFAULT_PROFILES)}")
        return v


class AskSource(BaseModel):
    rank: int
    node_id: Optional[str]
    title: Optional[str]
    platform: Optional[str]
    date_start: Optional[str]
    date_end: Optional[str]
    similarity: Optional[float]
    snippet: str
    text: str
    people: List[str]
    relevant: bool
    matched_terms: List[str]


class AskTimelineItem(BaseModel):
    date: Optional[str]
    label: str
    node_id: Optional[str]
    platform: Optional[str]
    source: int


class AskEvidence(BaseModel):
    retrieved: int
    considered: int
    relevant: int
    terms: List[str]


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    summary_points: List[str]
    timeline: List[AskTimelineItem]
    sources: List[AskSource]
    notes: List[str]
    evidence: AskEvidence
    filters: dict
    model: str
    took_ms: int


class SearchResult(BaseModel):
    rank: int
    similarity: Optional[float]
    distance: Optional[float]
    start_time: Optional[str]
    end_time: Optional[str]
    platform: Optional[str]
    title: Optional[str]
    channel_id: Optional[str]
    node_id: Optional[str]
    people: List[str]
    summary: Optional[str]
    snippet: str
    text: str
    matched_via: str = "query"            # "query", "expansion" (Hinglish/English topic words) or "both"
    expansion_terms: List[str] = []


class SearchResponse(BaseModel):
    query: str
    table: str
    model: str
    count: int
    took_ms: int
    results: List[SearchResult]


def graph_node_id(channel_id):
    """Thread node id used by export_cosmograph.py (T_<Threads.id>); chunk channel_id is that root thread id."""
    if channel_id is None or str(channel_id).strip() == "":
        return None
    return f"T_{channel_id}"


def make_snippet(text, limit=SNIPPET_CHARS):
    """First `limit` chars of the chunk body, skipping the Platform/Title/... context header."""
    lines = (text or "").splitlines()
    body = [l for l in lines if not l.startswith(("Platform:", "Title:", "Subreddit:", "Participants:", "Timeframe:"))]
    flat = " ".join(l.strip() for l in (body or lines) if l.strip())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


def to_api_result(result):
    channel_id = result.get("channel_id")
    return {
        **result,
        "channel_id": str(channel_id) if channel_id not in (None, "") else None,
        "node_id": graph_node_id(channel_id),
        "snippet": make_snippet(result.get("text")),
    }


class ThreadChunk(BaseModel):
    start_time: Optional[str]
    end_time: Optional[str]
    platform: Optional[str]
    title: Optional[str]
    people: List[str]
    snippet: str
    text: str


class ThreadContextResponse(BaseModel):
    node_id: str
    channel_id: str
    count: int
    first_time: Optional[str]
    last_time: Optional[str]
    chunks: List[ThreadChunk]


def to_thread_chunk(chunk):
    text = chunk.get("text") or ""
    return {
        **chunk,
        "snippet": make_snippet(text),
        "text": text if len(text) <= CONTEXT_TEXT_CHARS else text[:CONTEXT_TEXT_CHARS - 1].rstrip() + "…",
    }


class SearchService:
    """Opens the LanceDB table and loads the embedding model lazily, caching both.

    A missing table is not cached, so a server started while embedder.py is still building
    the index starts answering as soon as the table appears, without a restart.
    """

    def __init__(self, db_path=LANCEDB_PATH, table_name=search.DEFAULT_TABLE, model_loader=None, lock=None):
        self.db_path = str(db_path)
        self.table_name = table_name
        self.model_loader = model_loader or search.load_model
        self._lock = lock or threading.Lock()   # model loading + encoding (shared when workspaces share a model)
        self._index_lock = threading.Lock()  # opening the table; never held while a model loads
        self._index = None  # (table, meta)
        self._models = {}

    def _open_index(self):
        with self._index_lock:
            if self._index is None:
                table = search.open_table(self.db_path, self.table_name)
                meta = search.table_model_metadata(table, self.table_name, self.db_path)
                self._index = (table, meta)
            return self._index

    def _model(self, name):
        if name not in self._models:
            self._models[name] = self.model_loader(name)
        return self._models[name]

    def status(self):
        """Index status for /api/health. Opens the table (cheap) but never loads the model,
        and doesn't wait for a model that another request is loading."""
        base = {"table": self.table_name, "db_path": self.db_path}
        try:
            _, meta = self._open_index()
        except search.SearchError as e:
            return {**base, "available": False, "reason": str(e).splitlines()[0]}
        return {
            **base,
            "available": True,
            "model": meta.get("model"),
            "vector_dim": meta.get("vector_dim"),
            "rows": meta.get("rows"),
            "created_at": meta.get("created_at"),
            "model_loaded": meta.get("model") in self._models,
        }

    def thread_context(self, channel_id):
        """Indexed chunks of one thread for the graph's details panel. Never loads the model."""
        try:
            table, _ = self._open_index()
        except search.SearchError as e:
            raise ApiError(503, "index_unavailable", str(e))
        try:
            chunks = search.thread_chunks(table, channel_id)
        except search.SearchError as e:
            raise ApiError(422, "invalid_request", str(e))
        except Exception as e:  # lancedb/IO errors
            raise ApiError(500, "context_failed", f"{type(e).__name__}: {e}")
        return [to_thread_chunk(c) for c in chunks]

    def search(self, query, limit, where=None):
        # One lock: model loading isn't thread-safe and a CPU box gains nothing from parallel encodes.
        with self._lock:
            try:
                table, meta = self._open_index()
            except search.SearchError as e:
                raise ApiError(503, "index_unavailable", str(e))
            try:
                model = self._model(meta["model"])
            except search.SearchError as e:
                raise ApiError(500, "model_unavailable", str(e))
            try:
                results = search.search_expanded(table, model, query, limit, where=where)
            except search.SearchError as e:
                raise ApiError(500, "search_failed", str(e))
            except Exception as e:  # lancedb/IO errors: report them instead of a bare 500
                raise ApiError(500, "search_failed", f"{type(e).__name__}: {e}")
        return meta, [to_api_result(r) for r in results]


def error_body(code, message):
    return {"error": {"code": code, "message": message}}


def parse_platforms(values):
    """?platforms=a,b and/or repeated ?platforms=a&platforms=b -> sorted list, or None for all."""
    out = sorted({p.strip().lower() for v in (values or []) for p in v.split(",") if p.strip()})
    bad = [p for p in out if not search.PLATFORM_RE.match(p)]
    if bad:
        raise ApiError(422, "invalid_request", f"platforms: invalid platform name: {bad[0]!r}")
    if len(out) > 32:
        raise ApiError(422, "invalid_request", "platforms: at most 32 platforms")
    return out or None


def parse_date_param(name, value, end=False):
    if value is None or not value.strip():
        return None
    if len(value) > 40:
        raise ApiError(422, "invalid_request", f"{name}: too long")
    try:
        return memory_brief.parse_bound(value, end=end)
    except ValueError:
        raise ApiError(422, "invalid_request", f"{name}: must be an ISO date (YYYY-MM-DD) or datetime")


class ReminderAdd(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    due: str = Field(..., max_length=40, description="ISO date or datetime (a plain date means midnight UTC unless all_day)")
    all_day: bool = False
    kind: Optional[str] = Field(None, max_length=20)


class ReminderChange(BaseModel):
    action: str = Field(..., max_length=10, description="done | open | dismiss | snooze | edit")
    until: Optional[str] = Field(None, max_length=40, description="snooze: ISO datetime")
    title: Optional[str] = Field(None, max_length=200)
    due: Optional[str] = Field(None, max_length=40)
    all_day: Optional[bool] = None


class UnlockRequest(BaseModel):
    workspace: str = Field(..., min_length=1, max_length=40)
    password: str = Field("", max_length=200)


def create_app(service=None, graph_html=GRAPH_HTML, graph_dir=GRAPH_DIR, insights_service=None, people_service=None,
               llm_profiles=None, llm_transport=None, spaces=None, reminders_service=None):
    """`spaces`: a workspaces.WorkspaceSet. Without it the server has one workspace built from the other arguments."""
    if spaces is None:
        service = service or SearchService()
        insights_service = insights_service or insights.InsightsService()
        people_service = people_service or people.PeopleService(insights_service.db_path, insights_service.identity_map)
        only = workspaces.Workspace("default", None, service, insights_service, people_service, Path(graph_dir),
                                    reminders=reminders_service or reminders.RemindersService(
                                        insights_service.db_path, insights_service.identity_map,
                                        reminders.STORE_DIR / "default.db"))
        spaces = workspaces.WorkspaceSet({"default": only}, "default")

    def ws_of(request):
        return spaces.for_token(request.cookies.get(workspaces.COOKIE))
    llm_profiles = llm_profiles if llm_profiles is not None else llm_config.load_profiles()
    llm_status = answer_writer.LlmStatus(llm_profiles, transport=llm_transport)
    llm_active = {}   # profile -> asyncio.Event of the answer being written (-np 1: a newer question replaces it)
    app = FastAPI(title="Sarthink local search", version="1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.spaces = spaces
    app.state.search_service = spaces.spaces[spaces.default].search

    # Lets the graph be opened from another local server (e.g. python3 -m http.server 8080).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status, content=error_body(exc.code, exc.message))

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = first.get("msg", "invalid request").removeprefix("Value error, ")
        return JSONResponse(status_code=422, content=error_body("invalid_request", f"{field}: {message}" if field else message))

    @app.get("/api/health")
    def health(request: Request):
        ws = ws_of(request)
        return {"status": "ok", "index": ws.search.status(), "memory_db": {"available": ws.insights.available()},
                "llm": llm_status.status(), "workspace": {"id": ws.id, "label": ws.label}}

    # ── Workspaces: the default for everyone; others after their password (see workspaces.py) ──
    def workspace_response(ws, token=None, clear=False):
        resp = JSONResponse(spaces.describe(ws), headers={"Cache-Control": "no-store"})
        if token:
            resp.set_cookie(workspaces.COOKIE, token, max_age=spaces.ttl_s, httponly=True, samesite="strict", path="/")
        elif clear:
            resp.delete_cookie(workspaces.COOKIE, path="/")
        return resp

    @app.get("/api/workspace")
    def get_workspace(request: Request):
        return workspace_response(ws_of(request))

    @app.post("/api/workspace/unlock")
    def unlock_workspace(req: UnlockRequest):
        if req.workspace not in spaces.spaces:
            raise ApiError(404, "not_found", "No such workspace.")
        try:
            ok = spaces.unlock(req.workspace, req.password)
        except workspaces.TooManyAttempts as e:
            raise ApiError(429, "too_many_attempts", str(e))
        if not ok:
            raise ApiError(401, "wrong_password", "Wrong password.")
        ws = spaces.spaces[req.workspace]
        if ws.id == spaces.default:
            return workspace_response(ws, clear=True)
        return workspace_response(ws, token=spaces.issue(ws.id))

    @app.post("/api/workspace/lock")
    def lock_workspace():
        return workspace_response(spaces.spaces[spaces.default], clear=True)

    @app.get("/api/llm")
    def llm():
        return {"profiles": llm_status.status(), "order": list(llm_config.PROFILE_ORDER)}

    # Sync handler: FastAPI runs it in a worker thread, so a slow CPU encode doesn't block the event loop.
    @app.post("/api/search", response_model=SearchResponse)
    def run_search(req: SearchRequest, request: Request):
        started = time.perf_counter()
        service = ws_of(request).search
        meta, results = service.search(req.query, req.limit)
        return {
            "query": req.query,
            "table": service.table_name,
            "model": meta["model"],
            "count": len(results),
            "took_ms": int((time.perf_counter() - started) * 1000),
            "results": results,
        }

    def run_ask(req, service):
        started = time.perf_counter()
        try:
            plan = memory_brief.plan_query(req.question, req.date_from, req.date_to)
        except ValueError as e:
            raise ApiError(422, "invalid_request", str(e))
        start, end = plan["date_from"], plan["date_to"]
        if start and end and start >= end:
            raise ApiError(422, "invalid_request", "date_from must be before date_to")
        try:
            where = search.filter_expression(req.platforms, memory_brief.iso(start), memory_brief.iso(end))
        except search.SearchError as e:
            raise ApiError(422, "invalid_request", str(e))
        meta, results = service.search(plan["query"], ASK_CANDIDATES, where=where)
        brief = memory_brief.build_brief(req.question, results, limit=req.limit, platforms=req.platforms,
                                         date_from=start, date_to=end, notes=plan["notes"])
        return {
            "question": req.question,
            **brief,
            "filters": {"platforms": req.platforms, "date_from": memory_brief.iso(start), "date_to": memory_brief.iso(end),
                        "search_text": plan["query"]},
            "model": meta["model"],
            "took_ms": int((time.perf_counter() - started) * 1000),
        }

    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request):
        """Evidence-first answer: retrieve with the shared model, then summarise deterministically."""
        return run_ask(req, ws_of(request).search)

    @app.post("/api/ask/stream")
    async def ask_stream(req: AskStreamRequest, request: Request):
        """The /api/ask brief as the first event, then a local Llama's answer written from its evidence.

        Retrieval errors are ordinary JSON errors (same codes as /api/ask); once streaming, problems with the
        model arrive as an `error` event and the brief stays usable."""
        ws = ws_of(request)
        brief = await run_in_threadpool(run_ask, req, ws.search)
        profile = llm_profiles[req.profile]
        previous = llm_active.get(req.profile)
        if previous is not None:
            previous.set()
        cancelled = asyncio.Event()
        llm_active[req.profile] = cancelled
        owner = ws.insights.owner()

        async def events():
            try:
                yield answer_writer.sse("brief", AskResponse(**brief).model_dump())
                async for chunk in answer_writer.answer_events(req.question, brief, profile, owner,
                                                               request.is_disconnected, cancelled, llm_transport):
                    yield chunk
            finally:
                if llm_active.get(req.profile) is cancelled:
                    del llm_active[req.profile]

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/thread/{node_id}", response_model=ThreadContextResponse)
    def thread_context(node_id: str, request: Request):
        m = THREAD_NODE_RE.match(node_id)
        if not m:
            raise ApiError(422, "invalid_request", "node_id must look like T_<number> (a conversation thread)")
        chunks = ws_of(request).search.thread_context(m.group(1))
        return {
            "node_id": node_id,
            "channel_id": m.group(1),
            "count": len(chunks),
            "first_time": chunks[0]["start_time"] if chunks else None,
            "last_time": max((c["end_time"] or c["start_time"] or "" for c in chunks), default="") or None,
            "chunks": chunks[:CONTEXT_CHUNKS],
        }

    @app.get("/api/insights")
    def get_insights(request: Request, platforms: Optional[List[str]] = Query(None), date_from: Optional[str] = None,
                     date_to: Optional[str] = None, top: int = Query(insights.DEFAULT_TOP, ge=1, le=insights.MAX_TOP)):
        """Read-only aggregates over the memory database. A plain date_to includes that whole day."""
        plats = parse_platforms(platforms)
        start = parse_date_param("date_from", date_from)
        end = parse_date_param("date_to", date_to, end=True)
        if start and end and start >= end:
            raise ApiError(422, "invalid_request", "date_from must be before date_to")
        try:
            return ws_of(request).insights.insights(plats, start, end, top)
        except insights.InsightsUnavailable as e:
            raise ApiError(503, "database_unavailable", str(e))

    def person_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except people.BadRequest as e:
            raise ApiError(422, "invalid_request", str(e))
        except people.PersonNotFound as e:
            raise ApiError(404, "not_found", str(e))
        except people.PeopleUnavailable as e:
            raise ApiError(503, "database_unavailable", str(e))

    def person_filters(platforms, date_from, date_to):
        plats = parse_platforms(platforms)
        start = parse_date_param("date_from", date_from)
        end = parse_date_param("date_to", date_to, end=True)
        if start and end and start >= end:
            raise ApiError(422, "invalid_request", "date_from must be before date_to")
        return plats, start, end

    @app.get("/api/person/{node_id}")
    def person_profile(node_id: str, request: Request, platforms: Optional[List[str]] = Query(None), date_from: Optional[str] = None,
                       date_to: Optional[str] = None):
        """Profile of one person node: counts (all-time and filtered), activity, notable conversations, cited brief."""
        plats, start, end = person_filters(platforms, date_from, date_to)
        return person_call(ws_of(request).people.profile, node_id, plats, start, end)

    @app.get("/api/person/{node_id}/conversations")
    def person_conversations(node_id: str, request: Request, platforms: Optional[List[str]] = Query(None), date_from: Optional[str] = None,
                             date_to: Optional[str] = None, offset: int = Query(0, ge=0, le=1_000_000),
                             limit: int = Query(people.PAGE_DEFAULT, ge=1, le=people.PAGE_MAX)):
        plats, start, end = person_filters(platforms, date_from, date_to)
        return person_call(ws_of(request).people.conversations, node_id, plats, start, end, offset, limit)

    @app.get("/api/person/{node_id}/messages")
    def person_messages(node_id: str, request: Request, thread: Optional[str] = Query(None, max_length=24), cursor: Optional[str] = Query(None, max_length=200),
                        platforms: Optional[List[str]] = Query(None), date_from: Optional[str] = None, date_to: Optional[str] = None,
                        limit: int = Query(people.PAGE_DEFAULT, ge=1, le=people.PAGE_MAX)):
        plats, start, end = person_filters(platforms, date_from, date_to)
        return person_call(ws_of(request).people.messages, node_id, thread, plats, start, end, cursor, limit)

    # ── Reminders found in the chats (see reminders.py) ──
    def reminders_of(request):
        ws = ws_of(request)
        if ws.reminders is None:
            raise ApiError(503, "reminders_unavailable", "Reminders are not set up for this workspace.")
        return ws

    def reminders_writable(ws):
        """The public default workspace of a multi-workspace server is read-only: visitors keep their own
        done/snooze state in the browser instead of changing everyone's."""
        if spaces.switchable and ws.id == spaces.default:
            raise ApiError(403, "read_only", "Sample data is read-only here; your changes are kept in this browser.")

    def reminders_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except reminders.BadRequest as e:
            raise ApiError(422, "invalid_request", str(e))
        except reminders.ReminderNotFound as e:
            raise ApiError(404, "not_found", str(e))
        except reminders.RemindersUnavailable as e:
            raise ApiError(503, "database_unavailable", str(e))

    @app.get("/api/reminders")
    def list_reminders(request: Request, platforms: Optional[List[str]] = Query(None)):
        """Reminders grouped for the panel (overdue, today, week, later, history, done) and the badge count."""
        ws = reminders_of(request)
        out = reminders_call(ws.reminders.list, parse_platforms(platforms))
        out["writable"] = not (spaces.switchable and ws.id == spaces.default)
        return out

    @app.post("/api/reminders")
    def add_reminder(req: ReminderAdd, request: Request):
        ws = reminders_of(request)
        reminders_writable(ws)
        due = parse_date_param("due", req.due)
        if due is None:
            raise ApiError(422, "invalid_request", "due: required")
        return reminders_call(ws.reminders.add, req.title, int(due.timestamp()), req.all_day, req.kind or "reminder")

    @app.post("/api/reminders/scan")
    async def scan_reminders(request: Request, full: bool = False):
        """Read new messages for reminders now (the server also does this on its own)."""
        ws = reminders_of(request)
        if full:
            reminders_writable(ws)
        return await run_in_threadpool(reminders_call, ws.reminders.scan, full)

    @app.get("/api/reminders.ics", include_in_schema=False)
    def reminders_ics(request: Request, feed: Optional[str] = Query(None, max_length=100)):
        """The upcoming reminders as a calendar file. With ?feed=<token> (from /api/reminders/feed) calendar apps
        can subscribe to it without the workspace cookie."""
        ws = None
        if feed:
            for w in spaces.spaces.values():
                token = w.reminders.feed_token(create=False) if w.reminders else None
                if token and hmac.compare_digest(token, feed):
                    ws = w
                    break
            if ws is None:
                raise ApiError(404, "not_found", "Unknown calendar link.")
        else:
            ws = reminders_of(request)
        body = reminders_call(ws.reminders.to_ics, f"NASH Think · {ws.label or 'reminders'}")
        headers = {"Cache-Control": "no-store"}
        if not feed:
            headers["Content-Disposition"] = 'attachment; filename="nashthink-reminders.ics"'
        return Response(body, media_type="text/calendar; charset=utf-8", headers=headers)

    @app.get("/api/reminders/feed")
    def reminders_feed(request: Request):
        """The secret subscribe link for this workspace's calendar."""
        ws = reminders_of(request)
        reminders_writable(ws)
        return {"path": f"/api/reminders.ics?feed={ws.reminders.feed_token()}"}

    @app.post("/api/reminders/{rid}")
    def change_reminder(rid: int, req: ReminderChange, request: Request):
        """done / open / dismiss / snooze (until) / edit (title, due, all_day)."""
        ws = reminders_of(request)
        reminders_writable(ws)
        until = parse_date_param("until", req.until) if req.until else None
        due = parse_date_param("due", req.due) if req.due else None
        return reminders_call(ws.reminders.update, rid, req.action, int(until.timestamp()) if until else None,
                              req.title, int(due.timestamp()) if due else None, req.all_day)

    @app.get("/", include_in_schema=False)
    @app.get("/sarthink_graph.html", include_in_schema=False)
    def graph_page():
        if not Path(graph_html).is_file():
            raise HTTPException(404, "sarthink_graph.html not found")
        return FileResponse(graph_html, media_type="text/html")

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str):
        path = ASSETS_DIR / name
        if name not in ASSET_FILES or not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, media_type=ASSET_FILES[name], headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/processed_data/graph/{name}", include_in_schema=False)
    def graph_data(name: str, request: Request):
        path = Path(ws_of(request).graph_dir) / name
        if name not in GRAPH_FILES or not path.is_file():
            raise HTTPException(404, f"{name} not found. Run scripts/utils/export_cosmograph.py, then scripts/utils/compute_layout.py")
        # no-cache: a regenerated graph shows up on the next page refresh.
        return FileResponse(path, media_type="text/csv", headers={"Cache-Control": "no-cache"})

    return app


def _quietly(fn):
    try:
        fn()
    except Exception:   # a missing DB is reported by the endpoint itself
        pass


def clean_examples(value):
    """{"ask": [...], "search": [...]} with at most 6 short strings each, or None."""
    if not isinstance(value, dict):
        return None
    lists = {k: value.get(k) if isinstance(value.get(k), list) else [] for k in ("ask", "search")}
    out = {k: [q.strip()[:120] for q in v if isinstance(q, str) and q.strip()][:6] for k, v in lists.items()}
    return {k: v for k, v in out.items() if v} or None


def reminders_for(wid, ins, cfg=None):
    """A workspace's RemindersService. cfg (workspaces.json "reminders"): tz, now ("clock" | "latest_message"),
    lookback_days (null = the whole archive on the first scan)."""
    cfg = cfg if isinstance(cfg, dict) else {}
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", wid)[:40] or "default"
    return reminders.RemindersService(
        ins.db_path, ins.identity_map, reminders.STORE_DIR / f"{safe}.db", tz=cfg.get("tz") or reminders.reminder_rules.DEFAULT_TZ,
        now=cfg.get("now") if cfg.get("now") in ("clock", "latest_message") else "clock",
        lookback_days=cfg.get("lookback_days", reminders.DEFAULT_LOOKBACK_DAYS))


def build_spaces(config, table=search.DEFAULT_TABLE):
    """A WorkspaceSet from config/workspaces.json: one set of services per data folder, one shared embedding model."""
    loader, lock = workspaces.shared_model_loader(search.load_model), threading.Lock()
    spaces = {}
    for wid, cfg in config["workspaces"].items():
        paths = workspaces.workspace_paths(cfg["root"])
        ins = insights.InsightsService(paths["memory_db"], paths["identity_map"])
        spaces[wid] = workspaces.Workspace(
            wid, cfg.get("label") or wid, SearchService(paths["lancedb"], table, model_loader=loader, lock=lock),
            ins, people.PeopleService(ins.db_path, ins.identity_map), paths["graph_dir"], cfg.get("password"),
            clean_examples(cfg.get("examples")), reminders_for(wid, ins, cfg.get("reminders")))
    return workspaces.WorkspaceSet(spaces, config["default"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the Sarthink graph UI and local semantic search API.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default {DEFAULT_HOST}; keep it local)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})")
    parser.add_argument("--table", default=search.DEFAULT_TABLE, help=f"LanceDB table to search (default {search.DEFAULT_TABLE})")
    parser.add_argument("--db", default=LANCEDB_PATH, help="Path to the LanceDB directory")
    parser.add_argument("--memory-db", default=str(insights.DEFAULT_DB), help="SQLite memory database for /api/insights (opened read-only)")
    args = parser.parse_args(argv)

    import uvicorn

    try:
        config = workspaces.load_config(workspaces.default_config_path(REPO_ROOT))
    except workspaces.ConfigError as e:
        print(f"Sarthink: invalid workspaces config: {e}", file=sys.stderr)
        return 2
    if config:
        spaces = build_spaces(config, args.table)
        app = create_app(spaces=spaces)
        ins = spaces.spaces[spaces.default].insights
        names = ", ".join(f"{w.id}{' (default)' if w.id == spaces.default else ''}" for w in spaces.spaces.values())
        print(f"Sarthink: workspaces {names}", file=sys.stderr)
    else:
        ins = insights.InsightsService(args.memory_db)
        app = create_app(SearchService(args.db, args.table), insights_service=ins)
    # Warm the unfiltered insights in the background so the first Insights view opens instantly.
    threading.Thread(target=lambda: _quietly(ins.insights), daemon=True).start()
    print(f"Sarthink: http://{args.host}:{args.port}/  (searching table '{args.table}')", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
