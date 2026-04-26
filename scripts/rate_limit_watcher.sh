#!/bin/bash
# Lightweight rate limit watcher - only outputs when problems are detected
# Checks every 5 minutes, writes to a sentinel file if rate limits hit

WATCH_DIR="/Users/rishisim/Documents/research/ltm-research/intercode_sql_runs"
REACT_LOG="$WATCH_DIR/memory_agent_runs/train/react/world.log"
REFLEXION_LOG="$WATCH_DIR/memory_agent_runs/train/react_reflexion/world.log"
TASK_OUTPUTS="/private/tmp/claude-501/-Users-rishisim-Documents-research-ltm-research/a4619c16-464a-44ef-addf-e2b163815df8/tasks"
ALERT_FILE="$WATCH_DIR/RATE_LIMIT_ALERT"
STATUS_FILE="$WATCH_DIR/run_status.txt"

PREV_REACT=0
PREV_REFL=0
STALL_COUNT=0

while true; do
    NOW=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Count progress
    REACT_DONE=$(grep -c "^Task #" "$REACT_LOG" 2>/dev/null || echo 0)
    REFL_DONE=$(grep -c "^Task #" "$REFLEXION_LOG" 2>/dev/null || echo 0)
    REFL_TRIALS=$(grep -c "Start Trial" "$REFLEXION_LOG" 2>/dev/null || echo 0)
    
    # Check for rate limit strings across all outputs
    RATE_HITS=$(cat "$TASK_OUTPUTS"/bu18zpzt5.output "$TASK_OUTPUTS"/b9taf0kx7.output 2>/dev/null | grep -ci "rate.limit\|429\|too many requests\|quota exceeded" || true)
    RETRY_HITS=$(cat "$TASK_OUTPUTS"/bu18zpzt5.output "$TASK_OUTPUTS"/b9taf0kx7.output 2>/dev/null | grep -ci "Retrying\|ReadTimeout" || true)
    
    # Check if processes are still alive
    REACT_ALIVE=$(pgrep -f "frameworks react " >/dev/null 2>&1 && echo "yes" || echo "no")
    REFL_ALIVE=$(pgrep -f "frameworks react_reflexion" >/dev/null 2>&1 && echo "yes" || echo "no")
    
    # Write status file (cheap, always updated)
    cat > "$STATUS_FILE" << EOF
Last check: $NOW
ReAct: $REACT_DONE/516 tasks (process: $REACT_ALIVE)
Reflexion: $REFL_DONE tasks, $REFL_TRIALS trials (process: $REFL_ALIVE)
Rate limit hits: $RATE_HITS | Retry warnings: $RETRY_HITS
EOF
    
    # Alert on rate limits
    if [ "$RATE_HITS" -gt 0 ]; then
        echo "[$NOW] RATE LIMIT DETECTED: $RATE_HITS hits" > "$ALERT_FILE"
        cat "$TASK_OUTPUTS"/bu18zpzt5.output "$TASK_OUTPUTS"/b9taf0kx7.output 2>/dev/null | grep -i "rate.limit\|429\|too many requests\|quota exceeded" | tail -10 >> "$ALERT_FILE"
    fi
    
    # Detect stalls (no progress for 3 consecutive checks = 15 min)
    if [ "$REACT_DONE" -eq "$PREV_REACT" ] && [ "$REFL_DONE" -eq "$PREV_REFL" ]; then
        STALL_COUNT=$((STALL_COUNT + 1))
        if [ "$STALL_COUNT" -ge 3 ]; then
            echo "[$NOW] STALL DETECTED: No progress for 15+ minutes" >> "$ALERT_FILE"
            echo "ReAct: stuck at $REACT_DONE (alive: $REACT_ALIVE)" >> "$ALERT_FILE"
            echo "Reflexion: stuck at $REFL_DONE (alive: $REFL_ALIVE)" >> "$ALERT_FILE"
        fi
    else
        STALL_COUNT=0
    fi
    PREV_REACT=$REACT_DONE
    PREV_REFL=$REFL_DONE
    
    # Exit if both processes are done
    if [ "$REACT_ALIVE" = "no" ] && [ "$REFL_ALIVE" = "no" ]; then
        echo "[$NOW] Both runs completed." >> "$STATUS_FILE"
        echo "DONE" > "$WATCH_DIR/RUNS_COMPLETE"
        break
    fi
    
    sleep 300  # Check every 5 minutes
done
