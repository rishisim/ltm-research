#!/usr/bin/env bash
set -euo pipefail

# Run the clean GPT-5.4 replacement matrix.
# Default order: WebShop first, then ALFWorld, then SQL.
# The runs are restartable; every suite call uses --resume.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL="${MODEL:-gpt-5.4}"
SEEDS="${SEEDS:-0 1 2}"
FRAMEWORKS="${FRAMEWORKS:-react,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr}"
WEBSHOP_WORKERS="${WEBSHOP_WORKERS:-4}"
LOG_DIR="${LOG_DIR:-logs/gpt54_rerun}"

WEB_ROOT="${WEB_ROOT:-webshop_runs/rerun_clean/${MODEL}}"
ALF_ROOT="${ALF_ROOT:-alfworld_runs/rerun_clean/${MODEL}}"
SQL_ROOT="${SQL_ROOT:-intercode_sql_runs/rerun_clean/${MODEL}}"

mkdir -p "${LOG_DIR}"

run_logged() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"
  echo
  echo "============================================================"
  echo "[run] ${name}"
  echo "[run] log=${log_file}"
  echo "============================================================"
  "$@" 2>&1 | tee "${log_file}"
}

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "[run] required path missing: $1" >&2
    exit 1
  fi
}

require_file ".env"
require_file "webshop_runs/memory_retrieval_v2/memory_agent_runs/react_reflexion_train/knowledge_base.json"
require_file "alfworld_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json"
require_file "intercode_sql_runs/memory_agent_runs/train/react_reflexion/knowledge_base.json"

echo "[run] model=${MODEL}"
echo "[run] seeds=${SEEDS}"
echo "[run] frameworks=${FRAMEWORKS}"
echo "[run] webshop_workers=${WEBSHOP_WORKERS}"

echo "[run] ensuring SQL Docker is up"
docker compose -f data/intercode_sql/docker/docker-compose.yml up -d

echo "[run] phase 1: WebShop"
for seed in ${SEEDS}; do
  run_logged "webshop_seed_${seed}" \
    ./scripts/run_webshop_in_docker.sh \
      --num-tasks 200 \
      --splits test \
      --frameworks "${FRAMEWORKS}" \
      --model "${MODEL}" \
      --seed "${seed}" \
      --runs-root "${WEB_ROOT}" \
      --workers "${WEBSHOP_WORKERS}" \
      --quiet \
      --resume
done

echo "[run] phase 2: ALFWorld"
for seed in ${SEEDS}; do
  run_logged "alfworld_seed_${seed}" \
    .venv/bin/python scripts/run_framework_suite.py \
      --num-tasks 134 \
      --splits valid_unseen \
      --frameworks "${FRAMEWORKS}" \
      --model "${MODEL}" \
      --seed "${seed}" \
      --runs-root "${ALF_ROOT}" \
      --quiet \
      --resume
done

echo "[run] phase 3: SQL"
for seed in ${SEEDS}; do
  echo "[run] refreshing SQL Docker before SQL seed ${seed}"
  docker compose -f data/intercode_sql/docker/docker-compose.yml restart
  run_logged "sql_seed_${seed}" \
    .venv/bin/python scripts/run_intercode_sql_suite.py \
      --num-tasks 200 \
      --splits test \
      --frameworks "${FRAMEWORKS}" \
      --model "${MODEL}" \
      --seed "${seed}" \
      --runs-root "${SQL_ROOT}" \
      --quiet \
      --resume
done

echo "[run] auditing GPT-5.4 matrix"
run_logged "audit" \
  .venv/bin/python scripts/cloud/audit_gpt54_matrix.py \
    --model "${MODEL}" \
    --web-root "${WEB_ROOT}" \
    --alf-root "${ALF_ROOT}" \
    --sql-root "${SQL_ROOT}"

echo "[run] complete"
