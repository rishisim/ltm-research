import sys
import re
from typing import List, Tuple, Any, Dict
from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model


_SQL_START_RE = re.compile(
    r"^(select|with|show|describe|desc|explain|insert|update|delete|create|drop|alter|submit)\b",
    re.IGNORECASE,
)
_EMBEDDED_SQL_ACTION_RE = re.compile(
    r"(?<=[A-Za-z0-9_`'\")\]])"
    r"(?=(?:SHOW\s+COLUMNS\s+FROM|SHOW\s+TABLES|SELECT|WITH|DESCRIBE|DESC|EXPLAIN|submit)\b)",
    re.IGNORECASE,
)


def _paren_balance(text: str) -> int:
    balance = 0
    quote = ""
    escape = False
    for char in text:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "(":
            balance += 1
        elif char == ")":
            balance = max(0, balance - 1)
    return balance


def _clean_multiline_sql_action(action: str) -> str:
    """Normalize one SQL action while dropping leaked follow-up actions.

    InterCode SQL accepts one action per turn. The action can be a multi-line
    SQL statement, but model completions sometimes continue with the next turn
    (`submit`, another SQL query, or a prompt label). Keep the first statement
    and normalize it to a single line for the environment.
    """
    action = _EMBEDDED_SQL_ACTION_RE.sub("\n", action)
    lines = [line.strip() for line in action.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""

    first_lower = lines[0].lower()
    if first_lower.startswith("think:") or first_lower.startswith("help["):
        return lines[0]
    if first_lower == "submit" or first_lower.startswith("submit "):
        return "submit"

    kept = [lines[0]]
    for line in lines[1:]:
        lower = line.lower()
        if lower in {"submit", "obs:", "observation:", "action:"}:
            break
        if lower.startswith(("submit ", "obs:", "observation:", "action:")):
            break
        # A continuation line may legitimately start with SELECT/WITH inside a
        # subquery, e.g. HAVING COUNT(*) = (\nSELECT MIN(...). Only treat a new
        # SQL-looking line as a leaked next action when the current statement is
        # not inside open parentheses.
        if _SQL_START_RE.match(line) and _paren_balance("\n".join(kept)) == 0:
            break
        kept.append(line)

    return " ".join(" ".join(kept).split())


def clean_action_text(action_text: str, allow_newlines: bool = False) -> str:
    """Return exactly the next action from a model completion.

    Providers do not always honor stop sequences for reasoning models. Keep the
    agent/environment contract stable by stripping prompt labels and truncating
    any generated continuation before it reaches env.step().

    Most environments use one-line actions, but InterCode SQL can validly use
    multi-line SQL statements. In that mode we keep the whole SQL statement and
    normalize internal whitespace, while still stripping prompt-label leakage.
    """
    action = (action_text or "").strip()
    if action.startswith("Action:"):
        action = action[len("Action:"):].strip()
    if action.startswith(">"):
        action = action[1:].strip()

    single_line_action = (
        not allow_newlines
        or action.lower().startswith("think:")
        or action.lower().startswith("help[")
    )
    earliest = len(action)
    markers = (
        ("\r", "\n", "Obs:", "Observation:", "Action:")
        if single_line_action
        else (
            "\nObs:", "\nObservation:", "\nAction:",
            "\rObs:", "\rObservation:", "\rAction:",
            "Obs:", "Observation:", "Action:",
        )
    )
    for marker in markers:
        idx = action.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    action = action[:earliest].strip()
    if allow_newlines and not single_line_action:
        action = _clean_multiline_sql_action(action)
    return action


def allow_multiline_actions_for_env(env: BaseEnv) -> bool:
    """InterCode SQL is the only current env where multiline actions are valid."""
    return env.__class__.__name__ == "InterCodeSQLEnv"


def action_stop_sequences(allow_newlines: bool) -> List[str]:
    if allow_newlines:
        return [
            "\nObs:",
            "\nObservation:",
            "\nAction:",
            "\nsubmit",
            "\nSubmit",
            "\nSUBMIT",
        ]
    return ["\n"]


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
        allow_newlines = allow_multiline_actions_for_env(env)
        stop_sequences = action_stop_sequences(allow_newlines)
        while cur_step < 49:
            action_text, _usage = self._llm(str(env_history) + "Action:", stop=stop_sequences)
            action = clean_action_text(action_text, allow_newlines=allow_newlines)
            
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
