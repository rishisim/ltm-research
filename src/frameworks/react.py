import sys
from typing import List, Tuple, Any
from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model

class ReAct(Framework):
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        self.model = model
        self.to_print = to_print

    def _llm(self, prompt: str, stop: List[str] = ["\n"]) -> str:
        try:
            cur_try = 0
            while cur_try < 6:
                text = get_chat(prompt=prompt, model=self.model, temperature=cur_try * 0.2, stop_strs=stop)
                if text is not None and len(text.strip()) >= 5:
                    return text
                cur_try += 1
            return ""
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""

    def run(self, env: BaseEnv, base_prompt: str, memory: List[str], start_ob: str = "") -> Tuple[EnvironmentHistory, bool]:
        env_history = EnvironmentHistory(base_prompt, start_ob, memory[-3:] if len(memory) > 3 else memory)
        
        if self.to_print:
            print(start_ob)
            sys.stdout.flush()

        cur_step = 0
        while cur_step < 49:
            action = self._llm(str(env_history) + "Action:", stop=['\n']).strip()
            
            # Clean up action
            if action.startswith('Action:'):
                action = action[7:].strip()
            if action.startswith('>'):
                action = action[1:].strip()
            
            env_history.add("action", action)
            
            observation, reward, done, info = env.step(action)
            
            if action.startswith('think:'):
                observation = 'OK.'
            
            env_history.add("observation", observation)
            
            if self.to_print:
                print(f'Action: {action}\nObs: {observation}')
                sys.stdout.flush()
            
            if done:
                return env_history, True
            elif env_history.check_is_exhausted():
                return env_history, False
            
            cur_step += 1
        
        return env_history, False
