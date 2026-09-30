"""Data upload and ingestion manager for NASH Think.

Supports uploading chat exports from 9 platforms:
- WhatsApp (.txt or .zip)
- Instagram (.zip or .json)
- Facebook (.zip or .json)
- Discord (.zip package or channel JSONs)
- Reddit (posts.csv, comments.csv, chat_history.csv or .zip)
- Twitter / X (.zip or data/ folder)
- Google Takeout (.zip or Takeout/ folder)
- ChatGPT (conversations.json or export .zip)
- Claude (conversations.json or export .zip)

Includes auto-detection, secure file writing, archive file management,
and background ingestion pipeline execution.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("nashthink.uploader")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Allowed upload file extensions
ALLOWED_EXTENSIONS = {".zip", ".txt", ".json", ".csv", ".mbox", ".gz", ".tar"}
MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB max file size

SUPPORTED_PLATFORMS: Dict[str, Dict[str, Any]] = {
    "whatsapp": {
        "id": "whatsapp",
        "name": "WhatsApp",
        "color": "#25D366",
        "extensions": [".txt", ".zip"],
        "target_dir": "whatsapp",
        "description": "Exported chat .txt files (e.g. 'WhatsApp Chat with Alice.txt') or exported .zip per chat.",
        "parser_script": "scripts/parsers/whatsapp_parser.py",
        "instructions": [
            "Open WhatsApp on your mobile phone.",
            "Open the chat (individual or group) you want to export.",
            "Tap ⋮ (Android) or contact/group name (iOS) → More → Export chat.",
            "Choose 'Without Media' for the fastest export and smallest file size.",
            "Save or share the resulting .txt or .zip file to this computer and upload here."
        ]
    },
    "instagram": {
        "id": "instagram",
        "name": "Instagram",
        "color": "#E1306C",
        "extensions": [".zip", ".json"],
        "target_dir": "",
        "description": "Meta Account Center export in JSON format (keep as .zip or extract).",
        "parser_script": "scripts/parsers/meta_parser.py",
        "instructions": [
            "Go to Instagram Settings → Accounts Center → Your information and permissions.",
            "Click 'Download your information' → 'Download or transfer information'.",
            "Select your Instagram account, choose 'Some of your information' → 'Messages'.",
            "CRITICAL: Set format to 'JSON' (NOT HTML) and Date range: 'All time'.",
            "Once Meta emails you the download link, download the .zip file and upload it here."
        ]
    },
    "facebook": {
        "id": "facebook",
        "name": "Facebook",
        "color": "#1877F2",
        "extensions": [".zip", ".json"],
        "target_dir": "",
        "description": "Meta Account Center Facebook export in JSON format (keep as .zip).",
        "parser_script": "scripts/parsers/meta_parser.py",
        "instructions": [
            "Go to Facebook Settings & Privacy → Settings → Accounts Center.",
            "Click 'Your information and permissions' → 'Download your information'.",
            "Select your Facebook profile → Choose 'Messages'.",
            "CRITICAL: Format must be 'JSON' (NOT HTML).",
            "Download the completed zip archive and upload it here."
        ]
    },
    "discord": {
        "id": "discord",
        "name": "Discord",
        "color": "#5865F2",
        "extensions": [".zip", ".json"],
        "target_dir": "discord/package",
        "description": "Official Discord data package (.zip) or channel JSON export.",
        "parser_script": "scripts/parsers/discord_parser.py",
        "instructions": [
            "In Discord desktop/web app, click User Settings (gear icon) → Privacy & Safety.",
            "Scroll down to 'Request all of my Data' and click 'Request Data'.",
            "Discord will email you a download link (can take a few hours or days).",
            "Download the package .zip file and upload it here directly."
        ]
    },
    "chatgpt": {
        "id": "chatgpt",
        "name": "ChatGPT",
        "color": "#10A37F",
        "extensions": [".json", ".zip"],
        "target_dir": "chatgpt",
        "description": "ChatGPT data export conversations.json or the complete export .zip.",
        "parser_script": "scripts/parsers/chatgpt_parser.py",
        "instructions": [
            "In ChatGPT, click your profile picture (bottom-left) → Settings.",
            "Go to 'Data controls' → 'Export data' → Confirm export.",
            "Open the email from OpenAI and download the export .zip.",
            "Upload conversations.json or the entire .zip here."
        ]
    },
    "claude": {
        "id": "claude",
        "name": "Claude",
        "color": "#D97757",
        "extensions": [".json", ".zip"],
        "target_dir": "claude",
        "description": "Claude.ai conversations.json or data export .zip.",
        "parser_script": "scripts/parsers/claude_parser.py",
        "instructions": [
            "In Claude.ai, click your profile in the bottom-left → Settings.",
            "Go to 'Account' → 'Export Data'.",
            "Check your email for the download link and download the archive.",
            "Upload conversations.json or the .zip file here."
        ]
    },
    "reddit": {
        "id": "reddit",
        "name": "Reddit",
        "color": "#FF4500",
        "extensions": [".csv", ".zip"],
        "target_dir": "reddit-export",
        "description": "Reddit GDPR export (posts.csv, comments.csv, chat_history.csv, or .zip).",
        "parser_script": "scripts/parsers/reddit_parser.py",
        "instructions": [
            "Visit reddit.com/settings/data-request while logged in.",
            "Select 'GDPR' or 'General Data Request' and submit.",
            "Reddit will send a private message or email when the data is ready.",
            "Download the zip file containing posts.csv and comments.csv and upload here."
        ]
    },
    "twitter": {
        "id": "twitter",
        "name": "X / Twitter",
        "color": "#1DA1F2",
        "extensions": [".zip", ".js", ".json"],
        "target_dir": "twitter",
        "description": "Twitter / X data archive (.zip or data/ directory).",
        "parser_script": "scripts/parsers/twitter_parser.py",
        "instructions": [
            "In X / Twitter, go to More → Settings and Privacy → Your Account.",
            "Click 'Download an archive of your data' and verify your password.",
            "Wait for X notification/email when the archive is ready.",
            "Upload the downloaded .zip archive here."
        ]
    },
    "google": {
        "id": "google",
        "name": "Google Takeout",
        "color": "#FBBC05",
        "extensions": [".zip", ".mbox", ".json"],
        "target_dir": "google/Takeout",
        "description": "Google Takeout archive (Mail mbox, Google Chat, YouTube comments).",
        "parser_script": "scripts/parsers/google_parser.py",
        "instructions": [
            "Go to takeout.google.com.",
            "Deselect all, then select: Mail (MBOX format), Google Chat, and YouTube.",
            "Click 'Next step' → Delivery: Send download link via email → Format: .zip.",
            "Download the Takeout .zip once generated and upload here."
        ]
    }
}


def sanitize_filename(filename: str) -> str:
    """Strip dangerous path characters to prevent directory traversal."""
    cleaned = os.path.basename(filename).strip()
    cleaned = re.sub(r'[^a-zA-Z0-9_.\- ]', '_', cleaned)
    return cleaned or "upload.bin"


def detect_platform_from_file(filename: str, sample_bytes: bytes = b"") -> str:
    """Smart auto-detection of platform based on filename and header inspection."""
    name_lower = filename.lower()

    if name_lower.startswith("whatsapp chat") or name_lower.endswith(".txt") or "_chat.txt" in name_lower:
        return "whatsapp"
    if "instagram" in name_lower:
        return "instagram"
    if "facebook" in name_lower:
        return "facebook"
    if "discord" in name_lower or "package" in name_lower:
        return "discord"
    if "reddit" in name_lower or name_lower in ("posts.csv", "comments.csv", "chat_history.csv"):
        return "reddit"
    if "twitter" in name_lower or "tweets" in name_lower or "account.js" in name_lower:
        return "twitter"
    if "takeout" in name_lower or "google" in name_lower:
        return "google"

    # Inspect JSON content if available
    if sample_bytes and (name_lower.endswith(".json") or name_lower.endswith(".zip")):
        sample_str = sample_bytes[:4096].decode("utf-8", errors="ignore")
        if '"current_node"' in sample_str or '"mapping"' in sample_str:
            return "chatgpt"
        if '"chat_messages"' in sample_str or '"claude"' in sample_str:
            return "claude"
        if '"messages"' in sample_str and '"channel_id"' in sample_str:
            return "discord"

    if name_lower == "conversations.json":
        # Default to ChatGPT if ambiguous
        return "chatgpt"

    return "unknown"


def get_archive_base(ws: Any) -> Path:
    """Resolves the archive directory for the given workspace."""
    root = getattr(ws, "root", None)
    if root is None:
        if hasattr(ws, "graph_dir"):
            root = Path(ws.graph_dir).parent.parent
        else:
            root = REPO_ROOT
    return Path(root) / "archive"


def safe_extract_zip(zip_path: Path, target_dir: Path):
    """Safely extracts a zip file preventing Zip Slip vulnerability."""
    target_dir.mkdir(parents=True, exist_ok=True)
    target_dir_resolved = target_dir.resolve()
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.namelist():
            member_path = (target_dir / member).resolve()
            if not str(member_path).startswith(str(target_dir_resolved)):
                raise ValueError(f"Unsafe zip member detected: {member}")
        zf.extractall(target_dir)


class IngestJob:
    """Thread-safe background ingestion job tracker."""

    def __init__(self):
        self.lock = threading.Lock()
        self.status: str = "idle"  # idle, running, completed, error
        self.platform: str = "all"
        self.step: str = ""
        self.progress: int = 0
        self.logs: List[str] = []
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def start(self, platform: str):
        with self.lock:
            self.status = "running"
            self.platform = platform
            self.step = f"Starting ingest for {platform}..."
            self.progress = 5
            self.logs = [f"[{time.strftime('%H:%M:%S')}] Started ingest job for '{platform}'"]
            self.error = None
            self.started_at = time.time()
            self.finished_at = None

    def update(self, step: str, progress: int, log_line: Optional[str] = None):
        with self.lock:
            self.step = step
            self.progress = progress
            if log_line:
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] {log_line}")
                if len(self.logs) > 500:
                    self.logs = self.logs[-500:]

    def finish(self, success: bool, error_msg: Optional[str] = None):
        with self.lock:
            self.status = "completed" if success else "error"
            self.progress = 100 if success else self.progress
            self.error = error_msg
            self.finished_at = time.time()
            elapsed = int(self.finished_at - (self.started_at or self.finished_at))
            msg = f"Completed in {elapsed}s" if success else f"Failed after {elapsed}s: {error_msg}"
            self.logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "platform": self.platform,
                "step": self.step,
                "progress": self.progress,
                "logs": self.logs[-50:],  # return the most recent 50 lines
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


INGEST_JOB = IngestJob()


def run_pipeline_worker(repo_root: Path, platform: str, db_path: Path, graph_dir: Path):
    """Executes the ingestion pipeline in a background worker."""
    python_bin = sys.executable or str(repo_root / ".venv" / "bin" / "python")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'scripts' / 'utils'}:{repo_root / 'scripts' / 'semantic'}:{repo_root / 'scripts' / 'parsers'}"

    try:
        # Determine which parser(s) to execute
        parsers_to_run = []
        if platform == "all":
            parsers_to_run = [
                ("whatsapp", "scripts/parsers/whatsapp_parser.py"),
                ("instagram/facebook", "scripts/parsers/meta_parser.py"),
                ("discord", "scripts/parsers/discord_parser.py"),
                ("reddit", "scripts/parsers/reddit_parser.py"),
                ("twitter", "scripts/parsers/twitter_parser.py"),
                ("google", "scripts/parsers/google_parser.py"),
                ("chatgpt", "scripts/parsers/chatgpt_parser.py"),
                ("claude", "scripts/parsers/claude_parser.py"),
            ]
        elif platform in SUPPORTED_PLATFORMS:
            parsers_to_run = [(platform, SUPPORTED_PLATFORMS[platform]["parser_script"])]
        else:
            raise ValueError(f"Unknown platform for ingestion: {platform}")

        total_steps = len(parsers_to_run) + 2  # parsers + export_cosmograph + compute_layout
        current_step = 0

        for plat_name, script_rel in parsers_to_run:
            script_path = repo_root / script_rel
            if not script_path.exists():
                INGEST_JOB.update(f"Skipping {plat_name} (script not found)", int(current_step / total_steps * 80), f"Script {script_rel} not found.")
                continue

            current_step += 1
            progress_pct = int(current_step / total_steps * 70)
            INGEST_JOB.update(f"Parsing {plat_name}...", progress_pct, f"Running {script_rel}...")

            cmd = [python_bin, str(script_path)]
            proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                line = line.strip()
                if line:
                    INGEST_JOB.update(f"Parsing {plat_name}...", progress_pct, line)
            proc.wait()
            if proc.returncode != 0:
                logger.warning(f"Parser {script_rel} exited with code {proc.returncode}")

        # Step: export_cosmograph.py
        current_step += 1
        INGEST_JOB.update("Building 3D memory graph...", 75, "Running export_cosmograph.py...")
        export_script = repo_root / "scripts" / "utils" / "export_cosmograph.py"
        cmd = [python_bin, str(export_script), "--db", str(db_path), "--out", str(graph_dir)]
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"export_cosmograph failed: {proc.stderr}")
            INGEST_JOB.update("Memory graph export warning", 80, proc.stderr.strip()[:200])

        # Step: compute_layout.py
        current_step += 1
        INGEST_JOB.update("Computing galaxy 3D layout...", 90, "Running compute_layout.py...")
        layout_script = repo_root / "scripts" / "utils" / "compute_layout.py"
        proc = subprocess.run([python_bin, str(layout_script)], cwd=str(repo_root), env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"compute_layout failed: {proc.stderr}")
            INGEST_JOB.update("Galaxy layout warning", 95, proc.stderr.strip()[:200])

        INGEST_JOB.update("Ingestion complete!", 100, "Memory graph refreshed successfully.")
        INGEST_JOB.finish(True)

    except Exception as e:
        logger.exception("Ingest pipeline failed")
        INGEST_JOB.finish(False, str(e))


class IngestRequest(BaseModel):
    platform: str = Field("all", max_length=50)


def create_upload_router(get_current_workspace_fn) -> APIRouter:
    """Creates the FastAPI router with data upload and ingestion endpoints."""
    router = APIRouter(prefix="/api", tags=["upload"])

    @router.get("/upload/platforms")
    def list_platforms():
        """Lists all supported platforms, accepted file types, and export instructions."""
        return {
            "platforms": list(SUPPORTED_PLATFORMS.values()),
            "allowed_extensions": sorted(list(ALLOWED_EXTENSIONS)),
            "max_file_size_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
        }

    @router.get("/upload/status")
    def archive_status(request: Request):
        """Returns details about files currently stored in archive/ and DB records."""
        ws = get_current_workspace_fn(request)
        archive_base = get_archive_base(ws)
        archive_base.mkdir(parents=True, exist_ok=True)

        platform_status = {}
        for pid, pinfo in SUPPORTED_PLATFORMS.items():
            platform_dir = archive_base / pinfo["target_dir"] if pinfo["target_dir"] else archive_base
            files = []
            total_bytes = 0

            if platform_dir.exists():
                # Scan directory for relevant files
                for root_dir, _, filenames in os.walk(platform_dir):
                    for fn in filenames:
                        fp = Path(root_dir) / fn
                        ext = fp.suffix.lower()
                        # If pinfo target_dir is empty (like Meta in root), only match instagram/facebook
                        if not pinfo["target_dir"] and pid not in fn.lower():
                            continue
                        if ext in pinfo["extensions"]:
                            try:
                                sz = fp.stat().st_size
                                total_bytes += sz
                                files.append({
                                    "filename": fn,
                                    "rel_path": str(fp.relative_to(archive_base)),
                                    "size_bytes": sz,
                                    "modified_at": int(fp.stat().st_mtime),
                                })
                            except OSError:
                                pass

            # Also check message count in memory DB if available
            msg_count = 0
            try:
                db_path = Path(ws.insights.db_path) if hasattr(ws, "insights") else (Path(ws.root) / "processed_data" / "db" / "sarthink_memory.db")
                if db_path.exists():
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    cur = conn.cursor()
                    cur.execute("SELECT count(*) FROM Messages m JOIN Threads t ON m.thread_id = t.id WHERE t.platform = ?", (pid,))
                    row = cur.fetchone()
                    if row:
                        msg_count = row[0]
                    conn.close()
            except Exception:
                pass

            platform_status[pid] = {
                "id": pid,
                "name": pinfo["name"],
                "color": pinfo["color"],
                "file_count": len(files),
                "total_bytes": total_bytes,
                "files": files[:20],  # show first 20 files
                "messages_in_db": msg_count,
            }

        return {
            "archive_path": str(archive_base),
            "platforms": platform_status,
            "workspace_id": ws.id if hasattr(ws, "id") else "default"
        }

    @router.post("/upload")
    async def upload_file(
        request: Request,
        file: UploadFile = File(...),
        platform: Optional[str] = Form("auto"),
        extract_zip: Optional[bool] = Form(True),
    ):
        """Uploads a chat export file and routes it to the proper archive directory."""
        ws = get_current_workspace_fn(request)
        archive_base = get_archive_base(ws)
        archive_base.mkdir(parents=True, exist_ok=True)

        original_filename = file.filename or "upload.bin"
        safe_name = sanitize_filename(original_filename)
        ext = Path(safe_name).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file extension '{ext}'. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

        # Read small sample to detect platform
        sample_chunk = await file.read(8192)
        await file.seek(0)

        chosen_platform = platform.lower().strip() if platform else "auto"
        if chosen_platform == "auto" or chosen_platform not in SUPPORTED_PLATFORMS:
            detected = detect_platform_from_file(safe_name, sample_chunk)
            if detected != "unknown":
                chosen_platform = detected
            elif chosen_platform != "auto" and chosen_platform in SUPPORTED_PLATFORMS:
                pass
            else:
                chosen_platform = "unknown"

        # Determine target directory
        if chosen_platform in SUPPORTED_PLATFORMS:
            target_sub = SUPPORTED_PLATFORMS[chosen_platform]["target_dir"]
            dest_dir = archive_base / target_sub if target_sub else archive_base
        else:
            dest_dir = archive_base / "incoming"

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / safe_name

        # Stream file to disk
        total_size = 0
        try:
            with open(dest_file, "wb") as out_f:
                while True:
                    chunk = await file.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > MAX_UPLOAD_SIZE:
                        out_f.close()
                        dest_file.unlink(missing_ok=True)
                        raise HTTPException(413, f"File exceeds maximum allowed size of {MAX_UPLOAD_SIZE // (1024*1024)} MB")
                    out_f.write(chunk)
        except Exception as e:
            dest_file.unlink(missing_ok=True)
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(500, f"Failed to save upload: {e}")

        extracted_files = []
        # If it's a zip and extraction was requested for specific platforms
        if ext == ".zip" and extract_zip:
            if chosen_platform in ("chatgpt", "claude"):
                try:
                    # Check if conversations.json is inside zip
                    with zipfile.ZipFile(dest_file, 'r') as zf:
                        for member in zf.namelist():
                            if member.endswith("conversations.json"):
                                target_json = dest_dir / "conversations.json"
                                with zf.open(member) as src, open(target_json, "wb") as dst:
                                    shutil.copyfileobj(src, dst)
                                extracted_files.append("conversations.json")
                                break
                except Exception as e:
                    logger.warning(f"Could not extract conversations.json from zip: {e}")
            elif chosen_platform == "reddit":
                try:
                    with zipfile.ZipFile(dest_file, 'r') as zf:
                        for member in zf.namelist():
                            if member.endswith(".csv"):
                                csv_target = dest_dir / os.path.basename(member)
                                with zf.open(member) as src, open(csv_target, "wb") as dst:
                                    shutil.copyfileobj(src, dst)
                                extracted_files.append(os.path.basename(member))
                except Exception as e:
                    logger.warning(f"Could not extract reddit CSVs from zip: {e}")
            elif chosen_platform == "twitter":
                try:
                    safe_extract_zip(dest_file, dest_dir)
                    extracted_files.append("Extracted Twitter archive")
                except Exception as e:
                    logger.warning(f"Could not extract twitter zip: {e}")
            elif chosen_platform == "google":
                try:
                    safe_extract_zip(dest_file, dest_dir)
                    extracted_files.append("Extracted Takeout archive")
                except Exception as e:
                    logger.warning(f"Could not extract google zip: {e}")
            elif chosen_platform == "discord":
                try:
                    discord_pkg = dest_dir / Path(safe_name).stem
                    safe_extract_zip(dest_file, discord_pkg)
                    extracted_files.append(f"Extracted Discord package to {discord_pkg.name}")
                except Exception as e:
                    logger.warning(f"Could not extract discord zip: {e}")

        return {
            "status": "ok",
            "message": "File uploaded successfully",
            "filename": safe_name,
            "platform": chosen_platform,
            "platform_name": SUPPORTED_PLATFORMS.get(chosen_platform, {}).get("name", "Unknown"),
            "size_bytes": total_size,
            "saved_to": str(dest_file.relative_to(archive_base)),
            "extracted": extracted_files,
        }

    @router.post("/upload/delete")
    def delete_archive_file(request: Request, rel_path: str = Query(...)):
        """Deletes an uploaded archive file securely."""
        ws = get_current_workspace_fn(request)
        archive_base = get_archive_base(ws)
        target = (archive_base / rel_path).resolve()

        if not str(target).startswith(str(archive_base.resolve())):
            raise HTTPException(400, "Invalid file path.")
        if not target.is_file():
            raise HTTPException(404, "File not found.")

        try:
            target.unlink()
            return {"status": "ok", "message": f"Deleted {rel_path}"}
        except Exception as e:
            raise HTTPException(500, f"Failed to delete file: {e}")

    @router.post("/ingest/run")
    def trigger_ingest(req: IngestRequest, request: Request):
        """Starts background ingestion/parsing for the uploaded data."""
        if INGEST_JOB.snapshot()["status"] == "running":
            raise HTTPException(409, "An ingestion job is already running.")

        ws = get_current_workspace_fn(request)
        root = getattr(ws, "root", None)
        if root is None:
            root = Path(ws.graph_dir).parent.parent if hasattr(ws, "graph_dir") else REPO_ROOT

        db_path = Path(ws.insights.db_path) if hasattr(ws, "insights") else (Path(root) / "processed_data" / "db" / "sarthink_memory.db")
        graph_dir = Path(ws.graph_dir) if hasattr(ws, "graph_dir") else (Path(root) / "processed_data" / "graph")

        INGEST_JOB.start(req.platform)
        worker_thread = threading.Thread(
            target=run_pipeline_worker,
            args=(Path(root), req.platform, db_path, graph_dir),
            daemon=True
        )
        worker_thread.start()

        return {"status": "started", "platform": req.platform}

    @router.get("/ingest/status")
    def get_ingest_status():
        """Returns the current ingestion pipeline status and logs."""
        return INGEST_JOB.snapshot()

    return router
