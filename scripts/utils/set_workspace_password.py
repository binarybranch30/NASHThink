"""Sets the password of a workspace in config/workspaces.json (stored only as a salted PBKDF2 hash).

    python3 scripts/utils/set_workspace_password.py personal          # asks for the password twice
    printf '%s' "$PW" | python3 scripts/utils/set_workspace_password.py personal --stdin

The password is never printed or logged. Restart the server afterwards (scripts/stop_sarthink.sh, start_sarthink.sh).
"""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(REPO_ROOT / "scripts" / "api"))
import workspaces  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Set a Sarthink workspace password (hash only).")
    parser.add_argument("workspace")
    parser.add_argument("--stdin", action="store_true", help="read the password from stdin instead of prompting")
    parser.add_argument("--config", default=str(workspaces.default_config_path(REPO_ROOT)))
    args = parser.parse_args()

    path = Path(args.config)
    data = json.loads(path.read_text(encoding="utf-8"))
    if args.workspace not in data.get("workspaces", {}):
        sys.exit(f"No workspace {args.workspace!r} in {path}")
    if args.stdin:
        password = sys.stdin.read().rstrip("\n")
    else:
        password = getpass.getpass("New password: ")
        if password != getpass.getpass("Again: "):
            sys.exit("The passwords don't match.")
    if len(password) < 4:
        sys.exit("Use at least 4 characters.")
    data["workspaces"][args.workspace]["password"] = workspaces.hash_password(password)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(path)
    print(f"Password set for workspace {args.workspace!r} in {path}.")


if __name__ == "__main__":
    main()
