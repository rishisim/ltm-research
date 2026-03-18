import argparse
import subprocess
import concurrent.futures
import sys
import os
import json
from pathlib import Path
import shutil

# Use the WebShop venv's Python (has torch, selenium, flask needed by WebAgentTextEnv)
_WEBSHOP_VENV_PY = os.path.join(
    os.path.dirname(__file__), "../../../webshop/.webshop_venv/bin/python"
)
PYTHON_EXE = os.path.abspath(_WEBSHOP_VENV_PY) if os.path.exists(
    os.path.abspath(_WEBSHOP_VENV_PY)
) else sys.executable

def run_task(cmd, chunk_runs_root):
    try:
        print(f"Starting chunk -> {chunk_runs_root}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, cmd, chunk_runs_root
    except subprocess.CalledProcessError as e:
        print(f"Failed chunk -> {chunk_runs_root}\nError: {e.stderr}")
        return False, cmd, chunk_runs_root

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--framework", type=str, default="react_reflexion")
    args = parser.parse_args()

    commands = []
    chunk_size = max(1, (args.end - args.start) // args.workers)
    
    base_runs_root = "webshop_runs/memory_retrieval_v2/memory_agent_runs"
    
    for i in range(args.workers):
        chunk_start = args.start + i * chunk_size
        chunk_num_tasks = chunk_size if i < args.workers - 1 else (args.end - chunk_start)
        
        if chunk_num_tasks <= 0:
            continue
            
        chunk_runs_root = f"{base_runs_root}_chunk_{chunk_start}"
        cmd = [
            PYTHON_EXE,
            "scripts/legacy/run_webshop_suite.py",
            "--splits", "train",
            "--frameworks", args.framework,
            "--start-idx", str(chunk_start),
            "--num-tasks", str(chunk_num_tasks),
            "--runs-root", chunk_runs_root,
            "--quiet"
        ]
        commands.append((cmd, chunk_runs_root))

    successes = 0
    completed_roots = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, cmd, root): (cmd, root) for cmd, root in commands}
        for future in concurrent.futures.as_completed(futures):
            cmd, root = futures[future]
            try:
                success, _, chunk_root = future.result()
                if success:
                    successes += 1
                    completed_roots.append(chunk_root)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    print(f"Finished {successes}/{len(commands)} workers successfully.")
    
    # Merge trajectories
    master_dir = Path(f"{base_runs_root}/{args.framework}_train")
    master_dir.mkdir(parents=True, exist_ok=True)
    master_traj_file = master_dir / "trajectories.json"
    
    all_trajs = []
    if master_traj_file.exists():
        try:
            with open(master_traj_file, "r") as f:
                all_trajs = json.load(f)
        except json.JSONDecodeError:
            pass
            
    existing_ids = {t["task_id"] for t in all_trajs}
    added = 0
    
    for root in completed_roots:
        chunk_traj_file = Path(root) / f"{args.framework}_train" / "trajectories.json"
        if chunk_traj_file.exists():
            with open(chunk_traj_file, "r") as f:
                chunk_data = json.load(f)
                for t in chunk_data:
                    if t["task_id"] not in existing_ids:
                        all_trajs.append(t)
                        existing_ids.add(t["task_id"])
                        added += 1
        
        # Cleanup the chunk temporary directory
        shutil.rmtree(root, ignore_errors=True)
        
    with open(master_traj_file, "w") as f:
        json.dump(all_trajs, f, indent=2)
        
    print(f"Merged {added} new trajectories into {master_traj_file}")

if __name__ == "__main__":
    main()
