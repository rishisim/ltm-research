#!/bin/bash

echo "=== WebShop Experiments Status ==="
echo ""

# Check running processes
RUNNING_JOBS=$(pgrep -f "run_webshop_suite.py")
if [ -z "$RUNNING_JOBS" ]; then
    echo "[!] No WebShop evaluation suite is currently running."
else
    echo "[✓] WebShop evaluation is RUNNING (PIDs: $(echo $RUNNING_JOBS | tr '\n' ' '))"
fi

echo ""

# Find all run directories from recent logs
RUNS_DIR="webshop_runs"

if [ ! -d "$RUNS_DIR" ]; then
    echo "No WebShop runs found in $RUNS_DIR"
    exit 0
fi

# Print status of latest runs
for fw_dir in $(find "$RUNS_DIR" -maxdepth 3 -name "world.log" -print0 | xargs -0 ls -t | head -5 | xargs -n1 dirname); do
    echo "--- Framework Run: $(basename "$fw_dir") ---"
    
    # Extract total vs success from world.log endings
    WORLD_LOG="$fw_dir/world.log"
    
    if grep -q "ACCURACY:" "$WORLD_LOG"; then
        echo "Status: COMPLETED"
        tail -n 7 "$WORLD_LOG" | grep -E "SUCCESS:|FAIL:|TOTAL:|ACCURACY:|AVG REWARD:"
    else
        echo "Status: IN PROGRESS"
        
        # Count total evaluated so far
        SUCCESS_COUNT=$(cat "$WORLD_LOG" | grep -c "SUCCESS (reward=")
        FAIL_COUNT=$(cat "$WORLD_LOG" | grep -c "FAIL (reward=")
        let TOTAL=$SUCCESS_COUNT+$FAIL_COUNT
        
        echo "Completed tasks: $TOTAL"
        echo "Current Successes: $SUCCESS_COUNT"
        echo "Current Failures:  $FAIL_COUNT"
        echo ""
        echo "Latest task:"
        tail -n 1 "$WORLD_LOG"
    fi
    echo ""
done

echo "=================================="

