import json
import re

file_path = 'data/alfworld/prompts/alfworld_3prompts.json'

with open(file_path, 'r') as f:
    data = json.load(f)

# Revert: change "move it to" back to "put it in" in think steps
patterns = [
    # "then move it to X" -> "then put it in X"
    (r'then move it to (\w+)', r'then put it in \1'),
    # "move it to X" -> "put it in X"
    (r'move it to (stoveburner|microwave|garbagecan|diningtable|sidetable|fridge|drawer|shelf|dresser|sofa|countertop|toilet)', r'put it in \1'),
]

count = 0
for key in data:
    original = data[key]
    updated = original
    for pattern, replacement in patterns:
        updated = re.sub(pattern, replacement, updated)
    if updated != original:
        data[key] = updated
        count += 1
        print(f"Reverted: {key}")

print(f"\nTotal reverted: {count} prompts")

with open(file_path, 'w') as f:
    json.dump(data, f)

print("Saved.")
