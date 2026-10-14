#!/bin/bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "This package requires macOS on Apple Silicon."
  exit 1
fi
TASK_PYTHON=""
for candidate in python3.13 python3.12 python3.14 python3.11 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import platform,sys; sys.exit(not (platform.machine()=="arm64" and (3,11)<=sys.version_info[:2]<(3,15)))' 2>/dev/null; then
    TASK_PYTHON="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$TASK_PYTHON" ]]; then
  echo "Install an Apple Silicon Python 3.11–3.14 from python.org, then run this again."
  exit 1
fi
mkdir -p "$TASK_DIR/.runtime"
"$TASK_PYTHON" -m venv "$TASK_DIR/.runtime/venv"
"$TASK_DIR/.runtime/venv/bin/python" -m pip install --cache-dir "$TASK_DIR/.runtime/pip-cache" --upgrade pip
"$TASK_DIR/.runtime/venv/bin/python" -m pip install --cache-dir "$TASK_DIR/.runtime/pip-cache" --only-binary=:all: -r "$TASK_DIR/requirements.txt"
"$TASK_DIR/.runtime/venv/bin/python" -m pip freeze > "$TASK_DIR/.runtime/installed-versions.txt"
echo "Installed locally in .runtime. Checking Apple GPU access..."
bash "$TASK_DIR/run.sh" --check
