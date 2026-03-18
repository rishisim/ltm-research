#!/usr/bin/env python3
"""Live monitor for parallel WebShop trajectory generation."""

import os, sys, time, json, subprocess
from pathlib import Path

CHUNKS     = [150, 160, 170, 180, 190]
BASE       = Path("webshop_runs/memory_retrieval_v2/memory_agent_runs")
MASTER_DIR = BASE / "react_reflexion_train"
REFRESH    = 5  # seconds

def get_pids():
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "run_webshop_suite.py"], text=True
        ).strip().split()
        return set(out)
    except subprocess.CalledProcessError:
        return set()

def parse_log(log_path):
    if not log_path.exists():
        return 0, 0, 0, "loading products..."
    lines = log_path.read_text().splitlines()
    trials_started = sum(1 for l in lines if "Start Trial #" in l)
    successes      = sum(1 for l in lines if ": SUCCESS" in l and "previous" not in l)
    skipped        = sum(1 for l in lines if ": SKIPPED" in l)
    # last meaningful line
    meaningful = [l for l in reversed(lines) if l.strip() and not l.startswith("**")]
    last = meaningful[0][:72] if meaningful else "—"
    return trials_started, successes, skipped, last

def traj_count(chunk):
    tf = BASE.parent / f"memory_agent_runs_chunk_{chunk}" / "react_reflexion_train" / "trajectories.json"
    if tf.exists():
        try:
            return len(json.loads(tf.read_text()))
        except Exception:
            return "?"
    return 0

def master_count():
    tf = MASTER_DIR / "trajectories.json"
    if tf.exists():
        try:
            return len(json.loads(tf.read_text()))
        except Exception:
            return "?"
    return 0

def bar(n, total=10, width=10):
    filled = int(width * n / max(total, 1))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {n}/{total}"

def main():
    start = time.time()
    while True:
        os.system("clear")
        elapsed = int(time.time() - start)
        pids = get_pids()
        alive = len(pids)

        print(f"╔══════════════════════════════════════════════════════════════════╗")
        print(f"║       WebShop Trajectory Generation — Live Monitor              ║")
        print(f"║  Workers alive: {alive}/5   Elapsed: {elapsed//60:02d}m{elapsed%60:02d}s   Refresh: {REFRESH}s      ║")
        print(f"╠══════════════════════════════════════════════════════════════════╣")

        all_done = True
        for chunk in CHUNKS:
            log   = BASE.parent / f"memory_agent_runs_chunk_{chunk}" / "react_reflexion_train" / "world.log"
            t, s, sk, last = parse_log(log)
            done  = traj_count(chunk)
            chunk_dir = BASE.parent / f"memory_agent_runs_chunk_{chunk}"
            exists = chunk_dir.exists()
            status = "✓ done" if (not exists and done == 0) else ("🔄 running" if exists else "waiting")
            if exists:
                all_done = False

            print(f"  chunk_{chunk}  {status}")
            print(f"    Trials started: {t}   Successes: {s}   Skipped: {sk}")
            print(f"    Tasks saved:    {bar(done)}")
            print(f"    Last log:       {last}")
            print()

        mc = master_count()
        print(f"╠══════════════════════════════════════════════════════════════════╣")
        print(f"  Master trajectories.json: {mc} tasks total")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        print(f"\n  Press Ctrl+C to exit monitor (workers keep running).")

        if all_done and alive == 0:
            print("\n  ✅ All chunks finished and merged!")
            break

        time.sleep(REFRESH)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor exited. Workers are still running in background.")
