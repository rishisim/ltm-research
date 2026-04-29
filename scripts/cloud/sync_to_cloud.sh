#!/usr/bin/env bash
set -euo pipefail

# Sync this repo to a cloud machine.
# Usage:
#   scripts/cloud/sync_to_cloud.sh ubuntu@1.2.3.4:/home/ubuntu/ltm-research

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 user@host:/absolute/path/to/ltm-research" >&2
  exit 2
fi

TARGET="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${REPO_ROOT}"

rsync -azP \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".ruff_cache/" \
  --exclude ".DS_Store" \
  --exclude "logs/" \
  ./ "${TARGET%/}/"

echo "[sync] synced ${REPO_ROOT} -> ${TARGET}"
