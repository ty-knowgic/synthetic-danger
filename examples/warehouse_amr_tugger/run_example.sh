#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)

INPUT="$ROOT_DIR/examples/warehouse_amr_tugger/input.yaml"
OUTPUTS_DIR="$ROOT_DIR/outputs"
OUTPUTS_ARCHIVE_DIR="$ROOT_DIR/outputs_archive"

mkdir -p "$OUTPUTS_ARCHIVE_DIR"

# Prefer the repo's venv python if it exists (prevents accidentally using Homebrew/system python)
PYTHON="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
  else
    echo "ERROR: python executable not found. Activate your venv or install python." >&2
    exit 1
  fi
fi

# Discover the pipeline entry script (avoid guessing a module name)
ENTRY=""
CANDIDATES=(
  "$ROOT_DIR/run.py"
  "$ROOT_DIR/scripts/run.py"
  "$ROOT_DIR/scripts/run_pipeline.py"
  "$ROOT_DIR/src/run.py"
  "$ROOT_DIR/pipeline/run.py"
  "$ROOT_DIR/pipeline/main.py"
  "$ROOT_DIR/main.py"
)

for c in "${CANDIDATES[@]}"; do
  if [ -f "$c" ]; then
    ENTRY="$c"
    break
  fi
done

if [ -z "$ENTRY" ]; then
  echo "ERROR: Could not find an entry script (e.g., run.py)." >&2
  echo "Run one of these to locate it, then update this script accordingly:" >&2
  echo "  find . -maxdepth 3 -type f -name 'run.py' -o -name 'run_pipeline.py' -o -name 'main.py' -o -name '__main__.py'" >&2
  exit 1
fi

echo "Using python: $PYTHON" >&2
echo "Using entry:  $ENTRY" >&2
echo "Running: $PYTHON $ENTRY --mode yaml --yaml $INPUT" >&2

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

if [ -d "$OUTPUTS_DIR" ]; then
  if [ "$(ls -A "$OUTPUTS_DIR" 2>/dev/null | wc -l | tr -d ' ')" != "0" ]; then
    ARCHIVE_PATH="$OUTPUTS_ARCHIVE_DIR/outputs_$TIMESTAMP"
    echo "Archiving existing outputs to: $ARCHIVE_PATH" >&2
    mv "$OUTPUTS_DIR" "$ARCHIVE_PATH"
  fi
fi

"$PYTHON" "$ENTRY" \
  --mode yaml \
  --yaml "$INPUT"

if [ -d "$OUTPUTS_DIR" ]; then
  SNAPSHOT_DIR="$OUTPUTS_ARCHIVE_DIR/run_$TIMESTAMP"
  mkdir -p "$SNAPSHOT_DIR"
  echo "Snapshotting outputs to: $SNAPSHOT_DIR" >&2
  rsync -a "$OUTPUTS_DIR/" "$SNAPSHOT_DIR/"
  ln -sfn "$SNAPSHOT_DIR" "$OUTPUTS_ARCHIVE_DIR/latest"
else
  echo "WARNING: $OUTPUTS_DIR not found after run; skipping snapshot." >&2
fi

echo "" >&2
echo "Done. Check snapshots under: $OUTPUTS_ARCHIVE_DIR" >&2
