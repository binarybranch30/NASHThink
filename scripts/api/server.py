"""Local FastAPI server for Sarthink: semantic search API + the 3D memory graph UI.

Everything runs on this machine. Queries are embedded on CPU with the model recorded in the
table's metadata (see scripts/semantic/search.py); nothing is sent to a hosted service.

Usage:
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py            # http://127.0.0.1:8000
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/api/server.py --table topics --port 8000

Endpoints:
    GET  /                    sarthink_graph.html
    GET  /api/health          index/model status (never loads the model)
    POST /api/search          {"query": "...", "limit": 10} -> structured results
    POST /api/ask             {"question": "...", "limit": 8, "platforms": [...], "date_from": ..., "date_to": ...}
                              -> evidence-first memory brief (deterministic; see memory_brief.py)
    GET  /api/thread/T_<id>   indexed conversation chunks of one graph thread (no model load)
    GET  /processed_data/graph/cosmograph_{nodes,edges}.csv   graph data for the UI
"""
import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.append(str(REPO_ROOT / "scripts" / "semantic"))

import memory_brief  # noqa: E402
import search  # noqa: E402
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

    def __init__(self, db_path=LANCEDB_PATH, table_name=search.DEFAULT_TABLE, model_loader=None):
        self.db_path = str(db_path)
        self.table_name = table_name
        self.model_loader = model_loader or search.load_model
        self._lock = threading.Lock()        # model loading + encoding
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
                results = search.search(table, model, query, limit, where=where)
            except search.SearchError as e:
                raise ApiError(500, "search_failed", str(e))
            except Exception as e:  # lancedb/IO errors: report them instead of a bare 500
                raise ApiError(500, "search_failed", f"{type(e).__name__}: {e}")
        return meta, [to_api_result(r) for r in results]


def error_body(code, message):
    return {"error": {"code": code, "message": message}}


def create_app(service=None, graph_html=GRAPH_HTML, graph_dir=GRAPH_DIR):
    service = service or SearchService()
    app = FastAPI(title="Sarthink local search", version="1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.state.search_service = service

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
    def health():
        return {"status": "ok", "index": service.status()}

    # Sync handler: FastAPI runs it in a worker thread, so a slow CPU encode doesn't block the event loop.
    @app.post("/api/search", response_model=SearchResponse)
    def run_search(req: SearchRequest):
        started = time.perf_counter()
        meta, results = service.search(req.query, req.limit)
        return {
            "query": req.query,
            "table": service.table_name,
            "model": meta["model"],
            "count": len(results),
            "took_ms": int((time.perf_counter() - started) * 1000),
            "results": results,
        }

    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        """Evidence-first answer: retrieve with the shared model, then summarise deterministically."""
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

    @app.get("/api/thread/{node_id}", response_model=ThreadContextResponse)
    def thread_context(node_id: str):
        m = THREAD_NODE_RE.match(node_id)
        if not m:
            raise ApiError(422, "invalid_request", "node_id must look like T_<number> (a conversation thread)")
        chunks = service.thread_context(m.group(1))
        return {
            "node_id": node_id,
            "channel_id": m.group(1),
            "count": len(chunks),
            "first_time": chunks[0]["start_time"] if chunks else None,
            "last_time": max((c["end_time"] or c["start_time"] or "" for c in chunks), default="") or None,
            "chunks": chunks[:CONTEXT_CHUNKS],
        }

    @app.get("/", include_in_schema=False)
    @app.get("/sarthink_graph.html", include_in_schema=False)
    def graph_page():
        if not Path(graph_html).is_file():
            raise HTTPException(404, "sarthink_graph.html not found")
        return FileResponse(graph_html, media_type="text/html")

    @app.get("/processed_data/graph/{name}", include_in_schema=False)
    def graph_data(name: str):
        path = Path(graph_dir) / name
        if name not in GRAPH_FILES or not path.is_file():
            raise HTTPException(404, f"{name} not found. Run scripts/utils/export_cosmograph.py, then scripts/utils/compute_layout.py")
        # no-cache: a regenerated graph shows up on the next page refresh.
        return FileResponse(path, media_type="text/csv", headers={"Cache-Control": "no-cache"})

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the Sarthink graph UI and local semantic search API.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address (default {DEFAULT_HOST}; keep it local)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})")
    parser.add_argument("--table", default=search.DEFAULT_TABLE, help=f"LanceDB table to search (default {search.DEFAULT_TABLE})")
    parser.add_argument("--db", default=LANCEDB_PATH, help="Path to the LanceDB directory")
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(SearchService(args.db, args.table))
    print(f"Sarthink: http://{args.host}:{args.port}/  (searching table '{args.table}')", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
