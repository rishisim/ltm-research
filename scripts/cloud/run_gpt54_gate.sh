#!/usr/bin/env bash
set -euo pipefail

# Small GPT-5.4 gate before full cloud reruns.
# Default is WebShop-only because WebShop was the known failure point.
# Override PHASES/FRAMEWORKS/SEEDS/task counts as needed.

MODEL="${MODEL:-gpt-5.4}"
GATE_NAME="${GATE_NAME:-webshop_sanitized_20}"

export MODEL
export PHASES="${PHASES:-webshop}"
export SEEDS="${SEEDS:-0}"
export FRAMEWORKS="${FRAMEWORKS:-react,react_cr,react_tr,react_cr_tr,react_hard_neg_cr_tr}"
export WEBSHOP_NUM_TASKS="${WEBSHOP_NUM_TASKS:-20}"
export ALF_NUM_TASKS="${ALF_NUM_TASKS:-5}"
export SQL_NUM_TASKS="${SQL_NUM_TASKS:-10}"
export WEBSHOP_WORKERS="${WEBSHOP_WORKERS:-4}"
export WEBSHOP_MAX_LEARNINGS="${WEBSHOP_MAX_LEARNINGS:-3}"
export WEBSHOP_MIN_VALID_LEVEL="${WEBSHOP_MIN_VALID_LEVEL:-VALID_NEXT_TRIAL}"
export ALF_MAX_LEARNINGS="${ALF_MAX_LEARNINGS:-3}"
export ALF_MIN_VALID_LEVEL="${ALF_MIN_VALID_LEVEL:-VALID_NEXT_TRIAL}"
export SQL_MAX_LEARNINGS="${SQL_MAX_LEARNINGS:-3}"
export SQL_MIN_VALID_LEVEL="${SQL_MIN_VALID_LEVEL:-VALID_NEXT_TRIAL}"
export LOG_DIR="${LOG_DIR:-logs/gpt54_gate/${GATE_NAME}}"
export WEB_ROOT="${WEB_ROOT:-webshop_runs/gates/${MODEL}/${GATE_NAME}}"
export ALF_ROOT="${ALF_ROOT:-alfworld_runs/gates/${MODEL}/${GATE_NAME}}"
export SQL_ROOT="${SQL_ROOT:-intercode_sql_runs/gates/${MODEL}/${GATE_NAME}}"

exec "$(dirname "$0")/run_gpt54_matrix.sh"
