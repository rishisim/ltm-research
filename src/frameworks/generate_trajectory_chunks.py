
import json
import os
import argparse
from typing import List, Dict, Any
from src.core.llm import get_chat

# Constants
MODEL_NAME = "gemini-2.0-flash"

SYSTEM_PROMPT_TEMPLATE = """You are given one task trajectory (steps with action, observation).
Convert it into TrajectoryChunks.

Output MUST be valid JSON only. No markdown. No extra keys.

Schema:
{
  "task_id": string,
  "trial_num": number,
  "chunks": [
    {
      "chunk_id": number,
      "step_range": [start_step, end_step],
      "phase": one of ["PLAN","SEARCH","ACQUIRE","TRANSFORM","PLACE","RECOVER"],
      "goal": short string (<=12 words),
      "state_before": {
        "location": string|null,
        "inventory": [string],
        "open_containers": [string],
        "known_target_locations": [{"item": string, "location": string}]
      },
      "actions": string,
      "key_action": string,
      "result": one of ["SUCCESS","NO_EFFECT","FAIL"],
      "failure_type": one of ["PRECONDITION_MISSING","WRONG_VERB","CONTAINER_CLOSED_BLOCK","CONTAINER_OCCUPIED_BLOCK","ACTION_NO_EFFECT","OTHER", null],
      "state_after_delta": string,
      "progress_flag": true/false
    }
  ]
}

Rules:
1) Chunk boundary based on sub-goal completion or failure streaks.
2) actions MUST be a single string using arrows: "action1 -> action2".
   - OMIT "think" steps to keep it simple.
   - Format: verb(object [prep secondary]) e.g., "go(fridge1)", "take(apple)", "put(apple in microwave)".
   - If action failed, append "=NO_EFFECT".
   - If repeated identical actions fail, summarize: "take(egg from microwave)=NO_EFFECT (repeated)".
   - Examples:
     * "open(fridge1)=NO_EFFECT -> go(counter1) -> take(apple)"
     * "go(microwave1) -> open(microwave1)"
     * "close(microwave1) -> heat(apple with microwave1)"
3) key_action: A short string identifying the critical action e.g., "take(apple)", "heat(...)", "open(microwave1)".
4) state_after_delta: A concise string summary of state changes or learnings.
   - Examples:
     * "holding: apple; learned: fridge interactions may no-op"
     * "observed: egg in microwave"
     * "learned: open-state + egg causes interaction failures"
     * "apple heated; closing microwave enables heat"
5) Use env IDs from text exactly (e.g., "microwave 1").
6) If observation is exactly "Nothing happens.", set result="NO_EFFECT".
7) failure_type must be null if result="SUCCESS".

Now process:
<TRAJECTORY_JSON>"""

def load_json(path: str) -> List[Dict[str, Any]]:
    """Safe JSON loader."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {path}")
        return []

def save_json(path: str, data: List[Dict[str, Any]]):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def is_processed(chunk_data: List[Dict[str, Any]], task_id: str, trial_num: int) -> bool:
    """Check if a trial has already been processed."""
    for chunk in chunk_data:
        if chunk.get("task_id") == task_id and chunk.get("trial_num") == trial_num:
            return True
    return False

def format_trajectory_for_prompt(trajectory: Dict[str, Any]) -> str:
    """Formats the trajectory steps into a JSON string for the prompt."""
    # We strip down to just what is needed: steps with action/observation
    # The prompt expects <TRAJECTORY_JSON>
    
    # Minimal cleaning if needed, but passing the raw steps usually works best for LLMs 
    # if they are already clean JSON.
    return json.dumps(trajectory, indent=2)

def process_trajectory(trajectory: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Process a single trajectory to generate chunks."""
    
    traj_str = format_trajectory_for_prompt(trajectory)
    
    # Replace placeholder
    prompt = SYSTEM_PROMPT_TEMPLATE.replace("<TRAJECTORY_JSON>", traj_str)
    
    print(f"Generating chunks for {trajectory.get('task_id')} trial {trajectory.get('trial_num')}...")
    
    try:
        # Increase max_tokens to support long structured output
        response_text = get_chat(prompt, model=model, max_tokens=8192)
        
        # Clean up response
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
            
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        response_text = response_text.strip()
        
        chunk_data = json.loads(response_text)
        
        # Ensure task_id and trial_num are correct in the output
        chunk_data["task_id"] = trajectory.get("task_id")
        chunk_data["trial_num"] = trajectory.get("trial_num")
        
        return chunk_data
        
    except Exception as e:
        print(f"Error processing trajectory: {e}")
        print(f"Raw response: {response_text}")
        # Return partial object or None to indicate failure (handled by caller)
        return None

def generate_chunks(log_dir: str):
    """Main logic to handle file I/O and processing."""
    
    trajectories_path = os.path.join(log_dir, "trajectories.json")
    chunks_path = os.path.join(log_dir, "trajectory_chunks.json")
    
    if not os.path.exists(trajectories_path):
        print(f"No trajectories found at {trajectories_path}")
        return

    trajectories = load_json(trajectories_path)
    current_chunks = load_json(chunks_path)
    
    processed_count = 0
    
    for traj in trajectories:
        task_id = traj.get("task_id")
        trial_num = traj.get("trial_num")
        
        if is_processed(current_chunks, task_id, trial_num):
            print(f"Skipping already processed: {task_id} trial {trial_num}")
            continue
            
        result = process_trajectory(traj, MODEL_NAME)
        
        if result:
            current_chunks.append(result)
            # Save immediately to append effectively
            save_json(chunks_path, current_chunks)
            processed_count += 1
            
    print(f"Processed {processed_count} new trajectories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trajectory chunks from raw trajectories using LLM.")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing trajectories.json")
    
    args = parser.parse_args()
    
    generate_chunks(args.log_dir)
