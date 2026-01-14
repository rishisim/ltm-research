import sys
import os
import json
from typing import List, Tuple, Any, Dict
from pathlib import Path

from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct

from src.frameworks.memory_allocation.retrieval.rag_context_retrieval import retrieve_similar_trajectory

class RAGReAct(ReAct):
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        # Define paths relative to project root
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
        self.csv_path = self.project_root / "alfworld_runs/memory_agent_test/rag_react/truncated_trajectories.csv"
        self.cache_path = self.project_root / "alfworld_runs/memory_agent_test/rag_react/trajectory_embeddings.json"

    def run(self, env: BaseEnv, base_prompt: str, memory: List[str], start_ob: str = "", task_desc: str = "", task_id: str = "", trial_num: int = 1, log_dir: str = "") -> Tuple[EnvironmentHistory, bool]:
        """
        Run the RAG ReAct agent.
        
        Args:
            env: The environment
            base_prompt: The base prompt
            memory: List of memory strings
            start_ob: The starting observation
            task_desc: The description of the current task
            task_id: Unique identifier for the task
            trial_num: Trial number for this task
            log_dir: Directory to save logs
        """
        
        # Retrieval Step
        trunc_trajectory_text = ""
        retrieval_metadata = {}
        
        if task_desc:
            try:
                # We need to modify retrieve_similar_trajectory to return metadata if we want strict schema compliance
                # But for now, since retrieve_similar_trajectory only returns text, we'll manually construct what we can.
                # Actually, let's see if we can get more info. Rag context retrieval returns just string currently.
                # We will just verify it worked and log the text as context.
                
                # To get full metadata like similarity score, we'd need to update the retrieval function.
                # For this implementation, we will log the retrieved text.
                
                trunc_trajectory_text = retrieve_similar_trajectory(
                    task_desc, 
                    str(self.csv_path), 
                    str(self.cache_path)
                )
                
                if trunc_trajectory_text:
                    retrieval_metadata = {
                        "source": "rag_retrieval",
                        "retrieved_trajectory": trunc_trajectory_text
                    }
                    
            except Exception as e:
                if self.to_print:
                    print(f"Warning: RAG retrieval failed: {e}")
        
        enhanced_prompt = base_prompt
        if trunc_trajectory_text:
            enhanced_prompt = f"""{base_prompt}

[CONTEXT FROM A SIMILAR TASK]
Below is a partial trajectory from a similar task. Use this as a reference for the strategy, but adapt it to your current environment.
{trunc_trajectory_text}
[END CONTEXT]
"""
            if self.to_print:
                print(f"\n[RAG] Injected context from similar task.")

        env_history = EnvironmentHistory(enhanced_prompt, start_ob, memory[-3:] if len(memory) > 3 else memory)
        
        if self.to_print:
            print(start_ob)
            sys.stdout.flush()

        # Collect steps for logging
        steps: List[Dict[str, Any]] = []
        cur_step = 0
        reward = 0
        is_success = False # Default to False
        
        try:
            while cur_step < 49:
                action = self._llm(str(env_history) + "Action:", stop=['\n']).strip()
                
                # Clean up action
                if action.startswith('Action:'):
                    action = action[7:].strip()
                elif action.startswith('action:'):
                    action = action[7:].strip()
                if action.startswith('>'):
                    action = action[1:].strip()
                
                env_history.add("action", action)
                
                observation, reward, done, info = env.step(action)
                
                if action.startswith('think:'):
                    observation = 'OK.'
                
                env_history.add("observation", observation)
                
                # Record step
                steps.append({
                    "step": cur_step + 1,
                    "action": action,
                    "observation": observation,
                    "is_help_call": False
                })
                
                if self.to_print:
                    print(f'Action: {action}\nObs: {observation}')
                    sys.stdout.flush()
                
                if done:
                    is_success = reward > 0 or (isinstance(reward, bool) and reward)
                    break
                elif env_history.check_is_exhausted():
                    break
                
                cur_step += 1
                
        except KeyboardInterrupt:
            print("\nRun interrupted by user.")
            raise
        except Exception as e:
            print(f"\nRun failed with error: {e}")
            raise
        finally:
            if log_dir:
                # Log whatever we have
                self._log_trajectory(
                    log_dir, task_id, trial_num, steps, 
                    is_success, task_desc, retrieval_metadata
                )
            
            # Note: returns might be tricky in finally if we re-raise.
            # But we want to return normally if no exception.
            # If exception, we log and re-raise.
            pass
            
        return env_history, is_success

    def _log_trajectory(
        self, 
        log_dir: str, 
        task_id: str, 
        trial_num: int,
        steps: List[Dict[str, Any]], 
        success: bool, 
        task_desc: str = "",
        retrieval_metadata: Dict[str, Any] = None
    ) -> None:
        """Log the complete trajectory to trajectories.json"""
        # Extract task_type from task_id if possible
        task_type = task_id.split('-')[0] if task_id else ""
        
        trajectory = {
            "task_id": task_id,
            "task_type": task_type,
            "task_desc": task_desc,
            "trial_num": trial_num,
            "context_source": "rag_retrieval",
            "retrieval_metadata": retrieval_metadata or {},
            "steps": steps,
            "success": success,
            "help_calls": [], # RAG agent doesn't use help tool
            "help_call_count": 0,
            "step_num": len(steps)
        }
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        trajectories_path = os.path.join(log_dir, "trajectories.json")
        
        trajectories = []
        if os.path.exists(trajectories_path):
            try:
                with open(trajectories_path, 'r') as f:
                    trajectories = json.load(f)
            except json.JSONDecodeError:
                pass # Start fresh if corrupt
        
        trajectories.append(trajectory)
        
        with open(trajectories_path, 'w') as f:
            json.dump(trajectories, f, indent=2)
