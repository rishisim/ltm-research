import json
import csv
import os

def extract_to_csv(json_path, csv_path):
    print(f"Reading from {json_path}")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()

        for entry in data:
            row = {
                'task_type': entry.get('task_type', ''),
                'task_id': entry.get('task_id', ''),
                'task_desc': entry.get('task_desc', ''),
                'trial_num': entry.get('trial_num', ''),
                'num_steps': len(entry.get('steps', [])),
                'help_call_count': entry.get('help_call_count', 0)
            }
            row['result'] = 'SUCCESS' if entry.get('success') else 'FAIL'
            row['context_from_retrieval'] = '[]'
            row['max_similarity_score'] = ''
            row['min_similarity_score'] = ''
            row['num_retrievals'] = 0
            row['num_candidates'] = 0
            row['help_calls'] = json.dumps(entry.get('help_calls', []), separators=(',', ':'))
            writer.writerow(row)

    print(f"Successfully extracted {len(data)} rows to {csv_path}")

if __name__ == "__main__":
    base_dir = "alfworld_runs/memory_agent_test/with TR"
    json_file = os.path.join(base_dir, "trajectories_valid_seen.json")
    csv_file = os.path.join(base_dir, "valid_seen.csv")
    extract_to_csv(json_file, csv_file)
