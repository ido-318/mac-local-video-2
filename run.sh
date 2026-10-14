#!/bin/bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_PYTHON="$TASK_DIR/.runtime/venv/bin/python"
if [[ ! -x "$TASK_PYTHON" ]]; then
  echo "Run this first: bash \"$TASK_DIR/setup.sh\""
  exit 1
fi
export HF_HOME="$TASK_DIR/.runtime/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$TASK_PYTHON" "$TASK_DIR/local_video.py" "$@"
