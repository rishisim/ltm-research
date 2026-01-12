
import json
import os
import argparse
import hashlib
from typing import List, Dict, Any, Optional
from src.core.llm import get_chat

# Constants
MODEL_NAME = "gemini-2.5-flash"

# Enums for canonical vocabulary
POLARITY_VALUES = ["FAILURE_AVOIDANCE", "SUCCESS_RECIPE"]
PHASE_VALUES = ["SEARCH", "ACQUIRE", "TRANSFORM", "PLACE", "RECOVER", "ALL"]
PATCH_TYPE_VALUES = ["DO_INSTEAD", "AVOID", "ORDERING", "VERB_SWAP", "STATE_RESET"]
ORIGIN_VALUES = ["REFLEXION", "TRAJECTORY_DIFF", "SUCCESS_TRACE", "GENERALIZED"]

SYSTEM_PROMPT_FAILURE_AVOIDANCE = """You are extracting Strategy Memory entries from failed task trajectories and reflexions.

Output MUST be valid JSON only. No markdown. No extra keys.

Schema for each strategy:
{
  "polarity": "FAILURE_AVOIDANCE",
  "task_type": string (e.g., "pick_cool_then_place_in_recep"),
  "phase": one of ["SEARCH","ACQUIRE","TRANSFORM","PLACE","RECOVER"],
  "trigger_signature": object with canonical keys like:
    {
      "verb": string (the action verb),
      "target_type": string (object or appliance type),
      "result": "NO_EFFECT",
      "likely_reason": one of ["NOT_AT_APPLIANCE","WRONG_VERB","CONTAINER_CLOSED","CONTAINER_OCCUPIED","LOOP_DETECTED","OTHER"]
    },
  "patch_type": one of ["DO_INSTEAD","AVOID","ORDERING","VERB_SWAP","STATE_RESET"],
  "patch": string (canonical action template using format like:
    "IF not_at(appliance) THEN go_to(appliance) BEFORE verb(obj, appliance)"
    "IF verb1(obj, target) NO_EFFECT THEN verb2(obj, target)"
    "IF open(container) NO_EFFECT THEN skip(container) AND continue_search"
  ),
  "validation_evidence": string (brief description of what went wrong and how it was corrected)
}

Rules:
1) Extract ONLY strategies that can be validated from the trajectory data
2) Use canonical action templates, NOT prose descriptions
3) trigger_signature should use type-based references (e.g., "fridge" not "fridge 1")
4) Each strategy should be atomic and reusable across similar tasks
5) Focus on:
   - Verb swaps (put→move, put→cool, put→heat)
   - Ordering fixes (go_to before action)
   - State resets (close before heat)
   - Search abandonment (skip failing containers)

Input data:
<TASK_ID>
<TRIAL_NUM>
<CHUNKS_JSON>
<REFLEXION_TEXT>

Output a JSON object:
{
  "strategies": [...]
}
"""

SYSTEM_PROMPT_SUCCESS_RECIPE = """You are extracting a SUCCESS_RECIPE strategy from a successful task trajectory.

Output MUST be valid JSON only. No markdown. No extra keys.

Schema:
{
  "polarity": "SUCCESS_RECIPE",
  "task_type": string,
  "phase": "ALL",
  "trigger_signature": {
    "task_type": string,
    "obj_type": string,
    "target_recep_type": string,
    "appliance_type": string (if applicable, else null)
  },
  "patch_type": "DO_INSTEAD",
  "patch": string (ordered recipe like:
    "RECIPE: find(obj) -> take(obj) -> go_to(appliance) -> transform(obj, appliance) -> go_to(recep) -> move(obj, recep)"
  ),
  "validation_evidence": string (brief description of the successful path)
}

Rules:
1) Extract the minimal successful path (milestone chunks only)
2) Use type-based references (e.g., "tomato" not "tomato 1")
3) Include only actions that contributed to success
4) Use canonical verbs: find, take, go_to, open, close, cool, heat, move, put

Input data:
<TASK_ID>
<TRIAL_NUM>
<CHUNKS_JSON>

Output a JSON object with a single strategy:
{
  "strategy": {...}
}
"""


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


def save_json(path: str, data: Any):
    """Save data to JSON file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def generate_strategy_id(task_id: str, trial_num: int, index: int) -> str:
    """Generate a stable strategy ID."""
    hash_input = f"{task_id}_{trial_num}_{index}"
    return f"S{hashlib.md5(hash_input.encode()).hexdigest()[:6].upper()}"


def extract_task_type(task_id: str) -> str:
    """Extract the task type from a task_id."""
    # e.g., "pick_cool_then_place_in_recep-Tomato-None-Microwave-10/trial_..."
    parts = task_id.split("-")
    if parts:
        return parts[0]
    return "unknown"


def get_failure_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify failure anchor chunks."""
    failure_chunks = []
    for chunk in chunks:
        result = chunk.get("result", "")
        progress = chunk.get("progress_flag", True)
        if result in ["NO_EFFECT", "FAIL"] or not progress:
            failure_chunks.append(chunk)
    return failure_chunks


def get_success_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get milestone chunks from a successful trajectory."""
    success_chunks = []
    for chunk in chunks:
        if chunk.get("progress_flag", False) or chunk.get("result") == "SUCCESS":
            success_chunks.append(chunk)
    return success_chunks


def extract_failure_strategies(
    task_id: str,
    trial_num: int,
    chunks: List[Dict[str, Any]],
    reflexion: str,
    model: str
) -> List[Dict[str, Any]]:
    """Extract failure avoidance strategies from a failed trial."""
    
    failure_chunks = get_failure_chunks(chunks)
    if not failure_chunks and not reflexion:
        return []
    
    task_type = extract_task_type(task_id)
    
    prompt = SYSTEM_PROMPT_FAILURE_AVOIDANCE.replace(
        "<TASK_ID>", task_id
    ).replace(
        "<TRIAL_NUM>", str(trial_num)
    ).replace(
        "<CHUNKS_JSON>", json.dumps(failure_chunks, indent=2)
    ).replace(
        "<REFLEXION_TEXT>", reflexion if reflexion else "No reflexion available."
    )
    
    try:
        response_text = get_chat(prompt, model=model, max_tokens=4096)
        response_text = response_text.strip()
        
        # Clean up markdown if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        strategies = result.get("strategies", [])
        
        # Add metadata to each strategy
        for i, strategy in enumerate(strategies):
            strategy["strategy_id"] = generate_strategy_id(task_id, trial_num, i)
            strategy["origin"] = "REFLEXION" if reflexion else "TRAJECTORY_DIFF"
            strategy["source_task_ids"] = [task_id]
            strategy["source_trials"] = [trial_num]
            strategy["validated"] = False  # Will be validated later
            strategy["validation_score"] = 0.0
            
            # Ensure task_type is set
            if "task_type" not in strategy:
                strategy["task_type"] = task_type
        
        return strategies
        
    except Exception as e:
        print(f"Error extracting failure strategies: {e}")
        return []


def extract_success_recipe(
    task_id: str,
    trial_num: int,
    chunks: List[Dict[str, Any]],
    model: str
) -> Optional[Dict[str, Any]]:
    """Extract a success recipe from a successful trial."""
    
    success_chunks = get_success_chunks(chunks)
    if not success_chunks:
        return None
    
    task_type = extract_task_type(task_id)
    
    prompt = SYSTEM_PROMPT_SUCCESS_RECIPE.replace(
        "<TASK_ID>", task_id
    ).replace(
        "<TRIAL_NUM>", str(trial_num)
    ).replace(
        "<CHUNKS_JSON>", json.dumps(success_chunks, indent=2)
    )
    
    try:
        response_text = get_chat(prompt, model=model, max_tokens=2048)
        response_text = response_text.strip()
        
        # Clean up markdown if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        result = json.loads(response_text)
        strategy = result.get("strategy", {})
        
        if strategy:
            strategy["strategy_id"] = generate_strategy_id(task_id, trial_num, 0)
            strategy["origin"] = "SUCCESS_TRACE"
            strategy["source_task_ids"] = [task_id]
            strategy["source_trials"] = [trial_num]
            strategy["validated"] = True  # Success recipes are inherently validated
            strategy["validation_score"] = 1.0
            
            if "task_type" not in strategy:
                strategy["task_type"] = task_type
        
        return strategy
        
    except Exception as e:
        print(f"Error extracting success recipe: {e}")
        return None


def validate_strategies(
    strategies: List[Dict[str, Any]],
    all_chunks: Dict[str, Dict[int, List[Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    """
    Validate strategies by checking if the correction appears in later trials.
    
    all_chunks: Dict mapping task_id -> trial_num -> list of chunks
    """
    validated_strategies = []
    
    for strategy in strategies:
        if strategy.get("validated", False):
            validated_strategies.append(strategy)
            continue
        
        task_ids = strategy.get("source_task_ids", [])
        source_trials = strategy.get("source_trials", [])
        
        if not task_ids or not source_trials:
            validated_strategies.append(strategy)
            continue
        
        task_id = task_ids[0]
        max_source_trial = max(source_trials)
        
        # Check if there's a later successful trial
        if task_id in all_chunks:
            task_trials = all_chunks[task_id]
            later_trials = [t for t in task_trials.keys() if t > max_source_trial]
            
            for later_trial in later_trials:
                later_chunks = task_trials[later_trial]
                # Check if any later chunk shows progress on the same phase
                for chunk in later_chunks:
                    if chunk.get("progress_flag", False):
                        strategy["validated"] = True
                        strategy["validation_score"] = 1.0
                        strategy["source_trials"].append(later_trial)
                        break
                if strategy["validated"]:
                    break
        
        validated_strategies.append(strategy)
    
    return validated_strategies


def generate_strategy_memory(log_dir: str):
    """Main logic to generate strategy memory from trajectory chunks and reflexions."""
    
    chunks_path = os.path.join(log_dir, "trajectory_chunks.json")
    reflexions_path = os.path.join(log_dir, "reflexions.json")
    output_path = os.path.join(log_dir, "strategy_memory.json")
    
    if not os.path.exists(chunks_path):
        print(f"No trajectory chunks found at {chunks_path}")
        return
    
    trajectory_chunks = load_json(chunks_path)
    reflexions = load_json(reflexions_path)
    
    # Build lookup structures
    chunks_by_task_trial: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for entry in trajectory_chunks:
        task_id = entry.get("task_id")
        trial_num = entry.get("trial_num")
        chunks = entry.get("chunks", [])
        
        if task_id not in chunks_by_task_trial:
            chunks_by_task_trial[task_id] = {}
        chunks_by_task_trial[task_id][trial_num] = chunks
    
    reflexions_lookup: Dict[str, Dict[int, str]] = {}
    success_lookup: Dict[str, Dict[int, bool]] = {}
    for r in reflexions:
        task_id = r.get("task_id")
        trial_num = r.get("trial_num")
        is_success = r.get("is_success", False)
        reflexion_text = r.get("reflexion", "")
        
        if task_id not in reflexions_lookup:
            reflexions_lookup[task_id] = {}
            success_lookup[task_id] = {}
        reflexions_lookup[task_id][trial_num] = reflexion_text
        success_lookup[task_id][trial_num] = is_success
    
    all_strategies: List[Dict[str, Any]] = []
    
    # Process each trajectory
    for entry in trajectory_chunks:
        task_id = entry.get("task_id")
        trial_num = entry.get("trial_num")
        chunks = entry.get("chunks", [])
        
        is_success = success_lookup.get(task_id, {}).get(trial_num, False)
        reflexion = reflexions_lookup.get(task_id, {}).get(trial_num, "")
        
        print(f"Processing {task_id} trial {trial_num} (success={is_success})...")
        
        if is_success:
            # Extract success recipe
            recipe = extract_success_recipe(task_id, trial_num, chunks, MODEL_NAME)
            if recipe:
                all_strategies.append(recipe)
                print(f"  Extracted SUCCESS_RECIPE")
        else:
            # Extract failure avoidance strategies
            failure_strategies = extract_failure_strategies(
                task_id, trial_num, chunks, reflexion, MODEL_NAME
            )
            all_strategies.extend(failure_strategies)
            print(f"  Extracted {len(failure_strategies)} FAILURE_AVOIDANCE strategies")
    
    # Validate strategies
    print("\nValidating strategies...")
    all_strategies = validate_strategies(all_strategies, chunks_by_task_trial)
    
    validated_count = sum(1 for s in all_strategies if s.get("validated", False))
    print(f"Validated {validated_count}/{len(all_strategies)} strategies")
    
    # Save output
    output = {
        "metadata": {
            "source_log_dir": log_dir,
            "total_strategies": len(all_strategies),
            "failure_avoidance_count": sum(1 for s in all_strategies if s.get("polarity") == "FAILURE_AVOIDANCE"),
            "success_recipe_count": sum(1 for s in all_strategies if s.get("polarity") == "SUCCESS_RECIPE"),
            "validated_count": validated_count
        },
        "strategies": all_strategies
    }
    
    save_json(output_path, output)
    print(f"\nSaved {len(all_strategies)} strategies to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate strategy memory from trajectory chunks and reflexions."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing trajectory_chunks.json and reflexions.json"
    )
    
    args = parser.parse_args()
    
    generate_strategy_memory(args.log_dir)
