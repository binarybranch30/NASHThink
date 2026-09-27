"""Safely stages the parts of an official Discord data package (package.zip) that the parser needs.

The ZIP is only read, never modified. Before anything is written it checks (optionally) the byte size,
every member's CRC, and every member path: absolute paths, '..' components, backslashes, drive letters,
symlinks and encrypted members are rejected, and each target must resolve inside --dest. Only these
members are extracted, into private (0700/0600) directories:

    README.txt, Messages/index.json, Messages/c<id>/{channel.json,messages.json,messages.csv},
    Servers/index.json, and a minimised Account/user.json (identity + relationship names only)

The Activity/ analytics logs (gigabytes), avatars, billing and other account data are skipped. Output is
aggregate counts only: no names, IDs or message text.

    python3 scripts/utils/stage_discord_export.py --zip incoming/discord/package.zip \
        --dest archive/discord/naitik --expected-size 1445638511
"""
import argparse
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# Case-insensitive: older packages used lower-case folders and messages.csv.
KEEP_RE = re.compile(r"^(readme\.txt|messages/index\.json|messages/c?\d+/(channel\.json|messages\.json|messages\.csv)|"
                     r"servers/index\.json|account/user\.json)$", re.IGNORECASE)
USER_KEYS = ("id", "username", "global_name")
RELATIONSHIP_KEYS = ("id", "type", "nickname")
DISK_MARGIN = 1.2          # require this much more free space than the bytes being extracted
MAX_MEMBER_BYTES = 2 << 30  # a single needed member above 2 GiB is not a Discord message file


class StagingError(Exception):
    pass


def unsafe_reason(info, dest):
    """Why a ZIP member must not be extracted to `dest`, or None when its path is safe."""
    name = info.filename
    if "\\" in name or "\x00" in name:
        return "backslash_or_nul"
    path = PurePosixPath(name)
    if path.is_absolute() or name.startswith("/"):
        return "absolute_path"
    if any(part == ".." for part in path.parts):
        return "parent_traversal"
    if path.parts and re.match(r"^[A-Za-z]:", path.parts[0]):
        return "drive_letter"
    if stat.S_ISLNK(info.external_attr >> 16):
        return "symlink"
    if info.flag_bits & 0x1:
        return "encrypted"
    target = (dest / name).resolve()
    if target != dest and dest not in target.parents:
        return "outside_destination"
    return None


def minimise_user(raw):
    """Account/user.json reduced to what identity resolution needs: no email, phone, IP, sessions, billing."""
    user = {k: raw.get(k) for k in USER_KEYS if k in raw}
    rels = []
    for rel in raw.get("relationships") or []:
        entry = {k: rel.get(k) for k in RELATIONSHIP_KEYS if k in rel}
        u = rel.get("user") or {}
        entry["user"] = {k: u.get(k) for k in USER_KEYS if k in u}
        rels.append(entry)
    user["relationships"] = rels
    user["_minimised_by"] = "stage_discord_export.py"
    return user


def write_private(target, data):
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = target.with_name(target.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, target)
    os.chmod(target, 0o600)


def git_ignored(path):
    import subprocess
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(path)], capture_output=True)
    return r.returncode == 0


def stage(zip_path, dest, expected_size=None, verify_crc=True, log=print):
    zip_path, dest = Path(zip_path), Path(dest)
    report = {"zip_bytes": zip_path.stat().st_size}
    if expected_size is not None and report["zip_bytes"] != expected_size:
        raise StagingError(f"size mismatch: {report['zip_bytes']} bytes, expected {expected_size} (transfer incomplete?)")

    for path in (zip_path, dest):
        if (REPO_ROOT in path.resolve().parents) and not git_ignored(path.resolve()):
            raise StagingError(f"{path} is inside the repository but not ignored by Git")

    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(dest, 0o700)
    dest = dest.resolve()

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        report.update(members=len(infos), uncompressed_bytes=sum(i.file_size for i in infos))
        unsafe = {}
        for info in infos:
            reason = unsafe_reason(info, dest)
            if reason:
                unsafe[reason] = unsafe.get(reason, 0) + 1
        if unsafe:
            raise StagingError(f"unsafe ZIP members, nothing extracted: {unsafe}")

        wanted = [i for i in infos if not i.is_dir() and KEEP_RE.match(i.filename)]
        too_big = [i for i in wanted if i.file_size > MAX_MEMBER_BYTES]
        if too_big:
            raise StagingError(f"{len(too_big)} needed member(s) exceed {MAX_MEMBER_BYTES} bytes")
        needed = sum(i.file_size for i in wanted)
        free = shutil.disk_usage(dest).free
        report.update(selected_members=len(wanted), selected_bytes=needed, free_bytes=free,
                      skipped_members=len(infos) - len(wanted))
        if free < needed * DISK_MARGIN:
            raise StagingError(f"not enough disk space: {needed} bytes needed, {free} free")

        if verify_crc:
            log("Verifying CRC of every member (reads the whole archive)...")
            bad = zf.testzip()
            if bad is not None:
                raise StagingError("CRC check failed for a member; archive is corrupt")
            report["crc_ok"] = True

        for info in wanted:
            data = zf.read(info)
            if info.filename.lower() == "account/user.json":
                data = json.dumps(minimise_user(json.loads(data)), ensure_ascii=False, indent=1).encode("utf-8")
            write_private(dest / info.filename, data)
    for root, dirs, _ in os.walk(dest):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o700)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate package.zip and extract only what the Discord parser needs.")
    parser.add_argument("--zip", default=str(REPO_ROOT / "incoming" / "discord" / "package.zip"))
    parser.add_argument("--dest", default=str(REPO_ROOT / "archive" / "discord" / "naitik"))
    parser.add_argument("--expected-size", type=int, default=None, help="Byte size of the original file")
    parser.add_argument("--skip-crc", action="store_true", help="Skip the full CRC pass")
    args = parser.parse_args(argv)
    try:
        report = stage(args.zip, args.dest, args.expected_size, verify_crc=not args.skip_crc)
    except (StagingError, zipfile.BadZipFile, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
