import os
from typing import List, Dict, Any, Tuple
from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct

class Reflexion(ReAct):
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)
        # Load few-shot examples
        try:
            with open("reflexion_few_shot_examples.txt", 'r') as f:
                self.few_shot_examples = f.read()
        except FileNotFoundError:
            self.few_shot_examples = ""

    def update_memory(self, env_configs: List[Dict[str, Any]], trial_log_path: str) -> List[Dict[str, Any]]:
        """Updates memory with reflections on failed tasks."""
        with open(trial_log_path, 'r') as f:
            full_log: str = f.read()
            
        env_logs: List[str] = full_log.split('#####\n\n#####')
        # We assume one-to-one mapping here for simplicity in this refactor
        # In a real scenario, we'd be more careful about matching
        
        for i, env_conf in enumerate(env_configs):
            if not env_conf['is_success'] and not env_conf.get('skip', False):
                log_segment = env_logs[i] if i < len(env_logs) else ""
                reflection = self._generate_reflection(log_segment, env_conf['memory'])
                env_conf['memory'].append(reflection)
        
        return env_configs

    def _generate_reflection(self, log_str: str, memory: List[str]) -> str:
        scenario = log_str.split("Here is the task:")[-1].strip()
        query = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{self.few_shot_examples}

{scenario}"""

        if memory:
            query += '\n\nPlans from past attempts:\n'
            for i, m in enumerate(memory):
                query += f'Trial #{i}: {m}\n'

        query += '\n\nNew plan:'
        return get_chat(query, model=self.model)
