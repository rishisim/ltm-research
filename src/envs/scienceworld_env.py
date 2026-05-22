"""
ScienceWorld Environment Wrapper

Wraps ScienceWorld's Python API into the BaseEnv interface used by this
repository. ScienceWorld is a text-based science/procedure environment with
task variations and train/dev/test variation splits.

Setup:
  1. Install Java 1.8+.
  2. pip install scienceworld

The wrapped reward is the normalized total task score (0.0-1.0), not the
ScienceWorld delta reward. This matches the suite-level success metric and lets
memory-agent trajectory logs use the same partial-score semantics as WebShop
and InterCode SQL.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.core.base import BaseEnv


class ScienceWorldEnv(BaseEnv):
    """BaseEnv wrapper for ScienceWorld."""

    def __init__(
        self,
        task_name: str,
        variation_idx: int,
        simplification_str: str = "easy",
        jar_path: Optional[str] = None,
        env_step_limit: int = 100,
        max_valid_actions: int = 80,
        include_valid_actions: bool = True,
        scienceworld_env: Any = None,
    ):
        """
        Args:
            task_name: ScienceWorld task ID or task name accepted by env.load().
            variation_idx: Task variation index.
            simplification_str: ScienceWorld simplifications string. The official
                "easy" preset is the default for smoke/evaluation ergonomics.
            jar_path: Optional path to a ScienceWorld JAR. None uses the package
                bundled JAR.
            env_step_limit: Maximum ScienceWorld moves before the env terminates.
            max_valid_actions: Number of valid actions to expose in observations.
            include_valid_actions: Whether to append valid-action hints.
            scienceworld_env: Optional shared ScienceWorldEnv instance for
                sequential suite runs.
        """
        self.task_name = task_name
        self.variation_idx = variation_idx
        self.simplification_str = simplification_str
        self.jar_path = jar_path
        self.env_step_limit = env_step_limit
        self.max_valid_actions = max_valid_actions if max_valid_actions and max_valid_actions > 0 else None
        self.include_valid_actions = include_valid_actions
        self._shared_env = scienceworld_env
        self.env = scienceworld_env
        self.last_score = 0.0
        self.last_reward = 0.0
        self.last_info: Dict[str, Any] = {}
        self._init_env()

    def _init_env(self) -> None:
        """Initialize ScienceWorld, reusing a supplied JVM-backed env if present."""
        if self.env is not None:
            return
        try:
            from scienceworld import ScienceWorldEnv as _ScienceWorldEnv
        except ImportError as exc:
            raise ImportError(
                "Could not import ScienceWorld. Install with: pip install scienceworld. "
                "Java 1.8+ is also required."
            ) from exc

        self.env = _ScienceWorldEnv(
            "",
            self.jar_path,
            envStepLimit=self.env_step_limit,
        )

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Load the configured task variation and reset the simulator."""
        self.env.load(
            self.task_name,
            self.variation_idx,
            self.simplification_str,
        )
        observation, info = self.env.reset()
        self.last_score = float(info.get("score", 0)) / 100.0
        self.last_reward = self.last_score
        self.last_info = dict(info)
        return self._format_observation(observation, info, initial=True), self._format_info(info)

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Take one ScienceWorld action."""
        observation, _delta_reward, done, info = self.env.step(action)
        self.last_score = float(info.get("score", 0)) / 100.0
        self.last_reward = self.last_score
        self.last_info = dict(info)
        return (
            self._format_observation(observation, info, initial=False),
            self.last_reward,
            bool(done),
            self._format_info(info),
        )

    def close(self) -> None:
        """Close the JVM-backed ScienceWorld environment if this wrapper owns it."""
        if self.env is not None and self._shared_env is None:
            self.env.close()
        self.env = None

    def _format_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        formatted = dict(info)
        formatted.update(
            {
                "task_id": self.task_name,
                "task_name": info.get("taskName", self.task_name),
                "variation_idx": info.get("variationIdx", self.variation_idx),
                "score_normalized": self.last_score,
                "raw_score": info.get("score", 0),
            }
        )
        return formatted

    def _format_observation(
        self,
        observation: str,
        info: Dict[str, Any],
        initial: bool,
    ) -> str:
        parts: List[str] = []

        task_desc = info.get("taskDesc") or self._safe_call("taskdescription")
        if initial and task_desc:
            parts.append(f"Task: {task_desc}")

        if observation:
            parts.append(str(observation).strip())

        inventory = info.get("inv")
        if inventory:
            parts.append(f"Inventory: {str(inventory).strip()}")

        parts.append(f"Score: {self.last_score:.2f}")

        if self.include_valid_actions:
            actions = self.get_valid_actions(limit=self.max_valid_actions)
            if actions:
                parts.append("Valid actions:\n" + "\n".join(f"- {a}" for a in actions))

        return "\n\n".join(part for part in parts if part)

    def _safe_call(self, method_name: str) -> str:
        try:
            method = getattr(self.env, method_name)
            return str(method())
        except Exception:
            return ""

    def get_valid_actions(self, limit: Optional[int] = None) -> List[str]:
        """Return valid action strings for the current state."""
        try:
            raw_actions = self.env.get_valid_action_object_combinations_with_templates()
        except Exception:
            return []

        actions: List[str] = []
        for item in raw_actions:
            if isinstance(item, dict):
                action = item.get("action", "")
            else:
                action = str(item)
            action = action.strip()
            if action and action not in actions:
                actions.append(action)
            if limit is not None and len(actions) >= limit:
                break
        return actions

    @staticmethod
    def get_task_description(observation: str) -> str:
        """Extract the task description from a formatted observation."""
        lines = observation.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("Task:"):
                task_text = line.split("Task:", 1)[1].strip()
                if task_text and task_text != "Task Description:":
                    return task_text
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1].strip()
                    if next_line:
                        return next_line
        return observation.strip().splitlines()[0].strip() if observation.strip() else ""
