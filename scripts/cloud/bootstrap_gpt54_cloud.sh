#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh Ubuntu Lambda Cloud instance for the GPT-5.4 rerun.
# Run from the repository root on the cloud machine after syncing the repo.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[bootstrap] repo=${REPO_ROOT}"

if command -v apt-get >/dev/null 2>&1; then
  echo "[bootstrap] installing system packages"
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    default-jdk \
    docker.io \
    docker-compose-plugin \
    git \
    jq \
    python3-dev \
    python3-pip \
    python3-venv \
    rsync
fi

if command -v sudo >/dev/null 2>&1 && command -v docker >/dev/null 2>&1; then
  sudo systemctl enable --now docker || true
  if ! groups "${USER}" | grep -q '\bdocker\b'; then
    echo "[bootstrap] adding ${USER} to docker group; you may need to re-login for this to apply"
    sudo usermod -aG docker "${USER}" || true
  fi
fi

if [[ ! -d .venv ]]; then
  echo "[bootstrap] creating .venv"
  "${PYTHON_BIN}" -m venv .venv
fi

echo "[bootstrap] installing Python requirements"
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

if [[ -f webshop/requirements.txt ]]; then
  .venv/bin/python -m pip install -r webshop/requirements.txt
fi

echo "[bootstrap] verifying API env file presence"
if [[ ! -f .env ]]; then
  echo "[bootstrap] WARNING: .env is missing. Sync it before running the matrix." >&2
fi

echo "[bootstrap] building WebShop Docker image"
docker build \
  --platform=linux/amd64 \
  -t ltm-webshop-amd64 \
  -f webshop/Dockerfile.amd64 \
  webshop

echo "[bootstrap] starting SQL Docker"
docker compose -f data/intercode_sql/docker/docker-compose.yml up -d

echo "[bootstrap] done"
