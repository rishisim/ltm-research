#!/usr/bin/env bash
# run_webshop_in_docker.sh
#
# Wrapper that runs scripts/run_webshop_suite.py inside a linux/amd64 Docker
# container, working around the OpenJDK SIGBUS crash on Apple Silicon aarch64.
#
# Usage:
#   ./scripts/run_webshop_in_docker.sh [--num-tasks N] [--frameworks LIST] \
#       [--model MODEL] [--seed N] [--runs-root PATH] [--splits LIST] \
#       [--workers N] [--quiet] [--resume] [extra run_webshop_suite.py args...]
#
# Example (1-task smoke):
#   ./scripts/run_webshop_in_docker.sh \
#       --num-tasks 1 --frameworks react --model gemini-2.5-flash \
#       --seed 0 --runs-root /tmp/webshop_smoke

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="ltm-webshop-amd64"
DOCKERFILE="${REPO_ROOT}/webshop/Dockerfile.amd64"

# ── Ensure at most one WebShop run is active at a time ────────────────────────
LOCKFILE="/tmp/ltm_webshop.lock"
if [ -e "${LOCKFILE}" ]; then
    LOCK_PID="$(cat "${LOCKFILE}" 2>/dev/null || echo '?')"
    echo "[run_webshop_in_docker] ERROR: lockfile ${LOCKFILE} exists (PID ${LOCK_PID})." >&2
    echo "[run_webshop_in_docker] Another WebShop run may already be active." >&2
    echo "[run_webshop_in_docker] If not, remove the lockfile manually and retry." >&2
    exit 1
fi
echo $$ > "${LOCKFILE}"
trap 'rm -f "${LOCKFILE}"' EXIT INT TERM

# ── Build image if not present ─────────────────────────────────────────────────
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "[run_webshop_in_docker] Image '${IMAGE_NAME}' not found — building..."
    docker build \
        --platform=linux/amd64 \
        -t "${IMAGE_NAME}" \
        -f "${DOCKERFILE}" \
        "${REPO_ROOT}/webshop"
    echo "[run_webshop_in_docker] Build complete."
else
    echo "[run_webshop_in_docker] Image '${IMAGE_NAME}' already present, skipping build."
fi

# ── Resolve --runs-root: if relative, prepend /repo so output lands inside the
#    mounted repo (and is visible from the host).  If absolute and outside /repo
#    we leave it as-is (the caller must ensure it exists or is writable). ──────
RUNS_ROOT_ARG=""
PASS_ARGS=()
i=1
while [[ $i -le $# ]]; do
    arg="${!i}"
    if [[ "$arg" == "--runs-root" ]]; then
        i=$((i + 1))
        val="${!i}"
        # If value doesn't start with / it's relative — anchor to /repo
        if [[ "$val" != /* ]]; then
            val="/repo/${val}"
        fi
        RUNS_ROOT_ARG="$val"
        PASS_ARGS+=("--runs-root" "$val")
    else
        PASS_ARGS+=("$arg")
    fi
    i=$((i + 1))
done

# ── Mount host ~/.openrouter if it exists (for API key files) ─────────────────
OPENROUTER_MOUNT=""
if [[ -d "${HOME}/.openrouter" ]]; then
    OPENROUTER_MOUNT="-v ${HOME}/.openrouter:/root/.openrouter:ro"
fi

# ── Forward key environment variables ─────────────────────────────────────────
ENV_FLAGS=()
ENV_FLAGS+=("-e" "WEBSHOP_PATH=/repo/webshop")

# Pass through API keys from the host environment (takes priority over .env
# inside the container, but the container will also read /repo/.env via python-dotenv).
if [[ -n "${LTM_OPENROUTER_API_KEY:-}" ]]; then
    ENV_FLAGS+=("-e" "LTM_OPENROUTER_API_KEY=${LTM_OPENROUTER_API_KEY}")
fi
if [[ -n "${GEMINI_API_KEY:-}" ]]; then
    ENV_FLAGS+=("-e" "GEMINI_API_KEY=${GEMINI_API_KEY}")
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    ENV_FLAGS+=("-e" "OPENAI_API_KEY=${OPENAI_API_KEY}")
fi
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    ENV_FLAGS+=("-e" "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
fi

# ── Ensure runs-root exists (in-container path may be outside /repo) ──────────
if [[ -n "${RUNS_ROOT_ARG}" ]]; then
    # For /repo-relative paths create the directory on the host side now so
    # Docker volume sees it.  Strip /repo prefix to get host path.
    if [[ "$RUNS_ROOT_ARG" == /repo/* ]]; then
        HOST_RUNS_ROOT="${REPO_ROOT}/${RUNS_ROOT_ARG#/repo/}"
        mkdir -p "${HOST_RUNS_ROOT}"
    fi
fi

echo "[run_webshop_in_docker] Starting container with platform=linux/amd64..."
echo "[run_webshop_in_docker] Args: ${PASS_ARGS[*]:-<none>}"

# ── Run the suite inside the container ────────────────────────────────────────
# --rm          : clean up container after run
# --platform    : force amd64 emulation (key fix for SIGBUS on M-series Macs)
# -v /repo      : mount host repo read-write so script/data/output are accessible
# --shm-size    : prevent /dev/shm OOM for torch DataLoader workers
#
# NOTE: We do NOT set --memory here.  WebShop's load_products() parses a 5.5 GB
# JSON file and builds ~1.18M product dicts; peak RSS on the host is ~11 GB.
# Under QEMU amd64 emulation the overhead is higher still.  Any hard memory cap
# (e.g. --memory=12g or --memory=20g) causes the container to be OOM-killed
# mid-load (exit 137).  Docker Desktop on Mac shares the host's physical RAM;
# leave --memory unset so the container can use as much as it needs.
# Required: Docker Desktop must have ≥20 GB RAM allocated (Preferences →
# Resources → Memory).  Recommended: ≥22 GB for comfortable headroom.
docker run \
    --rm \
    --platform=linux/amd64 \
    -v "${REPO_ROOT}:/repo" \
    ${OPENROUTER_MOUNT} \
    "${ENV_FLAGS[@]}" \
    -e PYTHONUNBUFFERED=1 \
    --shm-size=4g \
    "${IMAGE_NAME}" \
    python -u /repo/scripts/run_webshop_suite.py "${PASS_ARGS[@]:-}"
