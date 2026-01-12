
import os
import sys
import yaml
import json
import alfworld.agents.environment as envs
from src.frameworks.memory_allocation.rag_react import RAGReAct

def main():
    # Set up environment
    os.environ["ALFWORLD_DATA"] = "/Users/rishisim/Documents/research/ltm-research/alfworld_runs/data"
    
    # Load a config (using a default one or minimal setup)
    # We'll just manually setup what we need
    
    # Initialize the RAG ReAct agent
    agent = RAGReAct(to_print=True)
    
    # Define a task description that we know exists or is similar to one in our truncated trajectories
    # First row in truncated_trajectories.csv is "look at alarmclock under the desklamp"
    task_desc = "look at alarmclock under the desklamp"
    print(f"Testing with task: {task_desc}")
    
    # We need to mock the environment step just enough to capture the first prompt
    # Or we can just inspect the agent internals if we exposed them, but running it is better.
    # Since we don't want to actually run a full ALFWorld game which requires complex setup,
    # we can try to retrieve directly first to verify that part, 
    # and then do a dummy run with a mock env to verify prompt construction.
    
    print("\n--- Testing Retrieval Directly ---")
    try:
        from src.frameworks.memory_allocation.rag_context_retrieval import retrieve_similar_trajectory
        csv_path = agent.csv_path
        cache_path = agent.cache_path
        
        result = retrieve_similar_trajectory(task_desc, str(csv_path), str(cache_path))
        if result:
            print("Retrieval SUCCESS!")
            print(f"Retrieved excerpt: {result[:100]}...")
        else:
            print("Retrieval FAILED (returned None)")
    except Exception as e:
        print(f"Retrieval ERROR: {e}")
        
    print("\n--- Testing Agent Prompt Construction ---")
    # Mock environment
    class MockEnv:
        def step(self, action):
            return "You are in a room.", 0.0, False, {}
            
    mock_env = MockEnv()
    
    # Run one step with logging enabled
    # We will override the _llm method to just print the prompt and exit/return
    original_llm = agent._llm
    
    def captured_llm(prompt, stop=None):
        print("\n[CAPTURED PROMPT START]")
        print(prompt)
        print("[CAPTURED PROMPT END]\n")
        return "think: stop test"
        
    agent._llm = captured_llm
    
    log_dir = "alfworld_runs/memory_agent_test/tests/logs"
    
    try:
        agent.run(
            env=mock_env,
            base_prompt="Interact with a household to solve a task.",
            memory=[],
            start_ob="You see a desk.",
            task_desc=task_desc,
            task_id="test_task_001",
            trial_num=1,
            log_dir=log_dir
        )
        print(f"\nAgent run completed. Checking for logs in {log_dir}/trajectories.json...")
        if os.path.exists(f"{log_dir}/trajectories.json"):
             print("Log file CREATED successfully.")
             with open(f"{log_dir}/trajectories.json", 'r') as f:
                 logs = json.load(f)
                 print(f"Log content check: {len(logs)} entries found.")
                 print(f"First entry task_id: {logs[0].get('task_id')}")
        else:
             print("Log file NOT FOUND.")
             
    except Exception as e:
        print(f"Agent Run Error: {e}")

if __name__ == "__main__":
    main()
