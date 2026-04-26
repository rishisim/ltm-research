#!/bin/bash
# Monitor InterCode SQL runs for progress and rate limit errors
# Usage: ./scripts/monitor_runs.sh [interval_seconds]
# Default: checks every 900s (15 minutes)

INTERVAL=${1:-900}
REACT_LOG="intercode_sql_runs/memory_agent_runs/train/react/world.log"
REFLEXION_LOG="intercode_sql_runs/memory_agent_runs/train/react_reflexion/world.log"
MONITOR_LOG="intercode_sql_runs/monitor.log"

echo "=== Run Monitor Started at $(date) ===" | tee -a "$MONITOR_LOG"
echo "Checking every ${INTERVAL}s" | tee -a "$MONITOR_LOG"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "--- Check at $TIMESTAMP ---" | tee -a "$MONITOR_LOG"

    # ReAct progress
    if [ -f "$REACT_LOG" ]; then
        REACT_TOTAL=$(grep -c "^Task #" "$REACT_LOG" 2>/dev/null || echo 0)
        REACT_SUCCESS=$(grep -c "SUCCESS" "$REACT_LOG" 2>/dev/null || echo 0)
        REACT_FAIL=$(grep -c "FAIL" "$REACT_LOG" 2>/dev/null || echo 0)
        REACT_LAST=$(tail -1 "$REACT_LOG" 2>/dev/null)
        echo "[ReAct] Tasks: $REACT_TOTAL/516 | Success: $REACT_SUCCESS | Fail: $REACT_FAIL" | tee -a "$MONITOR_LOG"
        echo "  Last: $REACT_LAST" | tee -a "$MONITOR_LOG"

        # Check if done
        if echo "$REACT_LAST" | grep -q "ACCURACY"; then
            echo "  ✓ ReAct COMPLETED" | tee -a "$MONITOR_LOG"
        fi
    else
        echo "[ReAct] Not started yet" | tee -a "$MONITOR_LOG"
    fi

    # Reflexion progress
    if [ -f "$REFLEXION_LOG" ]; then
        REFL_TOTAL=$(grep -c "^Task #" "$REFLEXION_LOG" 2>/dev/null || echo 0)
        REFL_SUCCESS=$(grep "^Task #" "$REFLEXION_LOG" 2>/dev/null | grep -c "SUCCESS" || echo 0)
        REFL_FAIL=$(grep "^Task #" "$REFLEXION_LOG" 2>/dev/null | grep -c "FAIL" || echo 0)
        REFL_TRIAL=$(grep -c "Start Trial" "$REFLEXION_LOG" 2>/dev/null || echo 0)
        REFL_LAST=$(grep "^Task #\|End Trial\|ACCURACY" "$REFLEXION_LOG" 2>/dev/null | tail -1)
        echo "[Reflexion] Trials started: $REFL_TRIAL | Task attempts: $REFL_TOTAL | Success: $REFL_SUCCESS | Fail: $REFL_FAIL" | tee -a "$MONITOR_LOG"
        echo "  Last: $REFL_LAST" | tee -a "$MONITOR_LOG"
    else
        echo "[Reflexion] Not started yet" | tee -a "$MONITOR_LOG"
    fi

    # Check for rate limit / retry errors across all log files
    ALL_LOGS="$REACT_LOG $REFLEXION_LOG"
    ALL_LOGS="$ALL_LOGS /private/tmp/claude-501/-Users-rishisim-Documents-research-ltm-research/a4619c16-464a-44ef-addf-e2b163815df8/tasks/bu18zpzt5.output"
    ALL_LOGS="$ALL_LOGS /private/tmp/claude-501/-Users-rishisim-Documents-research-ltm-research/a4619c16-464a-44ef-addf-e2b163815df8/tasks/b9taf0kx7.output"

    RATE_LIMIT_LINES=$(cat $ALL_LOGS 2>/dev/null | grep -i "rate.limit\|429\|too many requests\|quota\|ReadTimeout\|Retrying" | wc -l | tr -d ' ')

    if [ "$RATE_LIMIT_LINES" -gt 0 ]; then
        echo "⚠️  RATE LIMIT / RETRY WARNINGS: $RATE_LIMIT_LINES occurrences" | tee -a "$MONITOR_LOG"
        cat $ALL_LOGS 2>/dev/null | grep -i "rate.limit\|429\|too many requests\|quota\|ReadTimeout\|Retrying" | tail -5 | tee -a "$MONITOR_LOG"
    else
        echo "✓ No rate limit issues detected" | tee -a "$MONITOR_LOG"
    fi

    echo "" | tee -a "$MONITOR_LOG"

    # Check if both are done
    REACT_DONE=false
    REFL_DONE=false
    if [ -f "$REACT_LOG" ] && grep -q "^ACCURACY:" "$REACT_LOG" 2>/dev/null; then
        REACT_DONE=true
    fi
    if [ -f "$REFLEXION_LOG" ] && grep -q "Suite run complete\|^ACCURACY:" "$REFLEXION_LOG" 2>/dev/null; then
        REFL_DONE=true
    fi

    if $REACT_DONE && $REFL_DONE; then
        echo "=== Both runs completed! ===" | tee -a "$MONITOR_LOG"
        break
    fi

    sleep "$INTERVAL"
done
