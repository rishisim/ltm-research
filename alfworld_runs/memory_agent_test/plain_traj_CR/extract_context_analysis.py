"""
Extract Context Retrieval Analysis for Trajectory-Based CR Ablation Study

This script analyzes the context sent to agents in the with +LTM runs and outputs:
- task_id | task_split | system_prompt | context_sent_to_agent | word_count

Source: trajectories_valid_seen.json and trajectories_valid_unseen.json in with +LTM/
Output: plain_traj_CR/context_analysis.csv
"""

import json
import csv
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent  # memory_agent_test/
LTM_DIR = BASE_DIR / "with +LTM"
OUTPUT_DIR = BASE_DIR / "plain_traj_CR"

# System prompt template used in context_only_agent.py
SYSTEM_PROMPT_TEMPLATE = """[CONTEXT FROM PREVIOUS SIMILAR TASKS]
The following learnings are from previous tasks similar to yours. Use them to avoid common mistakes:

{context_learnings}

[END CONTEXT]

{base_prompt}"""


def format_learnings_for_prompt(learnings):
    """Reconstruct the context text from learnings (matches context_retrieval.py)"""
    if not learnings:
        return "No relevant learnings found."
    
    formatted_lines = ["Relevant learnings from previous tasks:"]
    
    for i, learning in enumerate(learnings, 1):
        phase = learning.get("goal_phase", "Unknown")
        issue = learning.get("issue", "")
        learning_text = learning.get("learning", "")
        
        formatted_lines.append(
            f"\n{i}. Phase: [{phase}]"
            f"\n   Issue: {issue}"
            f"\n   Learning: {learning_text}"
        )
    
    return "\n".join(formatted_lines)


def count_words(text):
    """Count words in text"""
    return len(text.split())


def process_trajectories(trajectories_path, split_name):
    """Process a trajectories file and return rows for CSV"""
    with open(trajectories_path, 'r') as f:
        trajectories = json.load(f)
    
    rows = []
    for traj in trajectories:
        task_id = traj.get("task_id", "")
        context_from_retrieval = traj.get("context_from_retrieval", [])
        
        # Format the context as it would appear to the agent
        context_text = format_learnings_for_prompt(context_from_retrieval)
        word_count = count_words(context_text)
        
        # Escape newlines so each row stays on one CSV line
        system_prompt_escaped = SYSTEM_PROMPT_TEMPLATE.replace('\n', '\\n')
        context_text_escaped = context_text.replace('\n', '\\n')
        
        rows.append({
            "task_id": task_id,
            "task_split": split_name,
            "system_prompt": system_prompt_escaped,
            "context_sent_to_agent": context_text_escaped,
            "word_count": word_count
        })
    
    return rows


def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    all_rows = []
    
    # Process valid_seen
    seen_path = LTM_DIR / "trajectories_valid_seen.json"
    if seen_path.exists():
        seen_rows = process_trajectories(seen_path, "seen")
        all_rows.extend(seen_rows)
        print(f"Processed {len(seen_rows)} tasks from valid_seen")
    else:
        print(f"Warning: {seen_path} not found")
    
    # Process valid_unseen
    unseen_path = LTM_DIR / "trajectories_valid_unseen.json"
    if unseen_path.exists():
        unseen_rows = process_trajectories(unseen_path, "unseen")
        all_rows.extend(unseen_rows)
        print(f"Processed {len(unseen_rows)} tasks from valid_unseen")
    else:
        print(f"Warning: {unseen_path} not found")
    
    # Write CSV
    output_path = OUTPUT_DIR / "context_analysis.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=["task_id", "task_split", "system_prompt", "context_sent_to_agent", "word_count"],
            delimiter='|',
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\nOutput written to: {output_path}")
    print(f"Total tasks: {len(all_rows)}")
    
    # Print word count stats
    word_counts = [row["word_count"] for row in all_rows]
    if word_counts:
        print(f"\nWord Count Statistics:")
        print(f"  Min: {min(word_counts)}")
        print(f"  Max: {max(word_counts)}")
        print(f"  Average: {sum(word_counts) / len(word_counts):.1f}")


if __name__ == "__main__":
    main()
