import sys
from typing import List, Tuple, Any, Dict
from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model


def clean_action_text(action_text: str) -> str:
    """Return exactly the next action from a model completion.

    Providers do not always honor stop sequences for reasoning models. Keep the
    agent/environment contract stable by stripping prompt labels and truncating
    any generated continuation before it reaches env.step().
    """
    action = (action_text or "").strip()
    if action.startswith("Action:"):
        action = action[len("Action:"):].strip()
    if action.startswith(">"):
        action = action[1:].strip()

    earliest = len(action)
    for marker in ("\r", "\n", "Obs:", "Observation:", "Action:"):
        idx = action.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return action[:earliest].strip()


class ReAct(Framework):
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        self.model = model
        self.to_print = to_print

    def _llm(
        self,
        prompt: str,
        stop: List[str] = ["\n"],
        system_prompt: str = None,
    ) -> Tuple[str, Dict[str, int]]:
        try:
            cur_try = 0
            while cur_try < 6:
                text, usage = get_chat(
                    prompt=prompt,
                    model=self.model,
                    temperature=cur_try * 0.2,
                    stop_strs=stop,
                    system_prompt=system_prompt,
                )
                # Client-side enforcement of stop sequences. OpenRouter does not
                # reliably honor `stop` for Anthropic and OpenAI reasoning models
                # (verified: Claude / gpt-5-mini emit multi-line output despite
                # body["stop"]=["\n"]). Truncating here makes the agent loop
                # robust regardless of upstream stop-sequence support.
                if text is not None and stop:
                    earliest = len(text)
                    for s in stop:
                        idx = text.find(s)
                        if idx != -1 and idx < earliest:
                            earliest = idx
                    text = text[:earliest]
                if text is not None and len(text.strip()) >= 5:
                    return text, usage
                cur_try += 1
            return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
        except Exception as e:
            print(f"LLM Error: {e}")
            return "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_tokens": 0}

    def run(self, env: BaseEnv, base_prompt: str, memory: List[str], start_ob: str = "") -> Tuple[EnvironmentHistory, bool]:
        # Add a brief framing instruction to stabilize the language model and avoid API timeouts
        stabilized_prompt = "You are an AI agent playing a text-based game. Follow the exact format of the examples below to solve the task.\n\n" + base_prompt
        
        env_history = EnvironmentHistory(stabilized_prompt, start_ob, memory[-3:] if len(memory) > 3 else memory)
        
        if self.to_print:
            print(start_ob)
            sys.stdout.flush()

        cur_step = 0
        while cur_step < 49:
            action_text, _usage = self._llm(str(env_history) + "Action:", stop=['\n'])
            action = clean_action_text(action_text)
            
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
