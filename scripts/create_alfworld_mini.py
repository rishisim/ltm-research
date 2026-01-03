import os
import shutil
import random
from tqdm import tqdm
from collections import defaultdict

# Official task types mapping (prefixes)
TASK_TYPES = {
    "pick_and_place_simple": "pick_and_place_simple",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep": "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep": "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place": "pick_two_obj_and_place"
}

def get_task_type(game_path):
    """Determine task type from the game path (typically the third to last component)."""
    # Path looks like: split/task_name/trial_name/game.tw-pddl
    parts = game_path.split(os.sep)
    if len(parts) < 3:
        return None
    task_name = parts[-3]
    for prefix in TASK_TYPES:
        if task_name.startswith(prefix):
            return prefix
    return None

def create_mini_dataset():
    src_base = "/Users/rishisim/.cache/alfworld/json_2.1.1"
    dst_base = "/Users/rishisim/Documents/research/ltm-research/alfworld_mini"
    
    # Clean and create target base
    if os.path.exists(dst_base):
        shutil.rmtree(dst_base)
    os.makedirs(dst_base)
    
    # 1. Handle Training Split (Sample 20 games per type)
    src_train = os.path.join(src_base, "train")
    dst_train = os.path.join(dst_base, "train")
    os.makedirs(dst_train)
    
    print("Analyzing training games...")
    all_train_games = []
    for root, dirs, files in os.walk(src_train):
        if "game.tw-pddl" in files:
            all_train_games.append(os.path.join(root, "game.tw-pddl"))
            
    train_by_type = defaultdict(list)
    for game in all_train_games:
        tt = get_task_type(game)
        if tt:
            train_by_type[tt].append(game)
    
    sampled_games = []
    for tt, games in train_by_type.items():
        print(f"Found {len(games)} games for {tt}")
        count = min(len(games), 20)
        sampled = random.sample(games, count)
        sampled_games.extend(sampled)
        
    print(f"Total sampled training games: {len(sampled_games)}")
    
    for game_path in tqdm(sampled_games, desc="Copying sampled train games"):
        # We need to copy the trial directory (parent of game.tw-pddl)
        # BUT we also need it to be inside its task directory.
        trial_dir = os.path.dirname(game_path)
        task_dir = os.path.dirname(trial_dir)
        
        rel_task_path = os.path.relpath(task_dir, src_train)
        rel_trial_path = os.path.relpath(trial_dir, src_train)
        
        dst_task_dir = os.path.join(dst_train, rel_task_path)
        dst_trial_dir = os.path.join(dst_train, rel_trial_path)
        
        os.makedirs(dst_task_dir, exist_ok=True)
        if not os.path.exists(dst_trial_dir):
            shutil.copytree(trial_dir, dst_trial_dir)

    # 2. Handle Valid Splits (Keep ALL valid games)
    for split in ["valid_seen", "valid_unseen"]:
        src_split = os.path.join(src_base, split)
        dst_split = os.path.join(dst_base, split)
        os.makedirs(dst_split)
        
        print(f"Analyzing {split} games...")
        split_games = []
        for root, dirs, files in os.walk(src_split):
            if "game.tw-pddl" in files:
                split_games.append(os.path.join(root, "game.tw-pddl"))
        
        print(f"Found {len(split_games)} valid games in {split}")
        for game_path in tqdm(split_games, desc=f"Copying {split}"):
            trial_dir = os.path.dirname(game_path)
            task_dir = os.path.dirname(trial_dir)
            
            rel_task_path = os.path.relpath(task_dir, src_split)
            rel_trial_path = os.path.relpath(trial_dir, src_split)
            
            dst_task_dir = os.path.join(dst_split, rel_task_path)
            dst_trial_dir = os.path.join(dst_split, rel_trial_path)
            
            os.makedirs(dst_task_dir, exist_ok=True)
            if not os.path.exists(dst_trial_dir):
                shutil.copytree(trial_dir, dst_trial_dir)

    print("Done! Alfworld-Mini dataset created at:", dst_base)

if __name__ == "__main__":
    create_mini_dataset()
