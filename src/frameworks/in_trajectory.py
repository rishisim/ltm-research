from typing import List, Tuple, Any
from src.core.base import Framework, BaseEnv
from src.core.history import EnvironmentHistory
from src.core.llm import get_chat, Model
from src.frameworks.react import ReAct

class InTrajectoryReflexion(ReAct):
    """
    A framework that allows for reflection and plan adjustment during a single trajectory.
    """
    def __init__(self, model: Model = "gemini-2.5-flash", to_print: bool = True):
        super().__init__(model, to_print)

    def run(self, env: BaseEnv, base_prompt: str, memory: List[str], start_ob: str = "") -> Tuple[EnvironmentHistory, bool]:
        # TODO: Implement in-trajectory reflection logic
        # This would typically involve checking the observation for failures
        # and calling a reflection prompt before the next action.
        return super().run(env, base_prompt, memory, start_ob)
