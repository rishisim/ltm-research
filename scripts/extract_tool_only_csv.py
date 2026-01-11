import json
import csv
import os

def extract_to_csv(json_path, csv_path):
    print(f"Reading from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    fieldnames = [
        'task_type', 'task_id', 'task_desc', 'trial_num', 
        'num_steps', 'result', 'context_from_retrieval', 
        'max_similarity_score', 'min_similarity_score', 'num_retrievals', 'num_candidates',
        'help_calls', 'help_call_count'
    ]

    print(f"Writing to {csv_path}")
    with open(csv_path, 'w', newline='') as f:
        # Use pipe delimiter for better parsing with complex JSON fields
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()

        for entry in data:
            # Extract basic fields
            row = {
                'task_type': entry.get('task_type', ''),
                'task_id': entry.get('task_id', ''),
                'task_desc': entry.get('task_desc', ''),
                'trial_num': entry.get('trial_num', ''),
                'num_steps': len(entry.get('steps', [])),  # Count actual steps
                'help_call_count': entry.get('help_call_count', 0)
            }
            
            # Map success/fail to Result column
            if entry.get('success'):
                row['result'] = 'SUCCESS'
            else:
                row['result'] = 'FAIL'

            # Dump context_from_retrieval to compact JSON (always empty for TR-only)
            context_retrieval = entry.get('context_from_retrieval', [])
            row['context_from_retrieval'] = json.dumps(context_retrieval, separators=(',', ':'))
            
            # Extract statistics from context_from_retrieval (will be empty for TR-only)
            if context_retrieval:
                similarity_scores = [item.get('similarity_score', 0) for item in context_retrieval]
                row['max_similarity_score'] = max(similarity_scores)
                row['min_similarity_score'] = min(similarity_scores)
                row['num_retrievals'] = len(context_retrieval)
                row['num_candidates'] = sum(1 for item in context_retrieval if item.get('valid_level') == 'CANDIDATE')
            else:
                row['max_similarity_score'] = ''
                row['min_similarity_score'] = ''
                row['num_retrievals'] = 0
                row['num_candidates'] = 0
            
            # Use the help_calls field directly from the JSON
            help_calls = entry.get('help_calls', [])
            row['help_calls'] = json.dumps(help_calls, separators=(',', ':'))

            writer.writerow(row)

    print(f"Successfully extracted {len(data)} rows to {csv_path}")

if __name__ == "__main__":
    base_dir = "alfworld_runs/memory_agent_test/with TR"
    json_file = os.path.join(base_dir, "trajectories_valid_unseen.json")
    csv_file = os.path.join(base_dir, "valid_unseen.csv")
    extract_to_csv(json_file, csv_file)
