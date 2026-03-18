"""
WebShop Environment Wrapper

Wraps WebShop's OpenAI Gym text environment into the BaseEnv interface
used by the LTM-Research framework.

WebShop uses 'simple' (text) mode with two primary actions:
  - search[query]       : search for products
  - click[element]      : click on page elements (items, options, Buy Now)

Observations are text descriptions of the current page state.
Reward is a 0-1 continuous score based on attribute match quality.

Setup:
  WebShop must be installed in its own environment. See scripts/setup_webshop.sh
  for installation instructions.
"""

import sys
import os
from typing import Dict, Any, Tuple
from src.core.base import BaseEnv


class WebShopEnv(BaseEnv):
    """BaseEnv wrapper for WebShop text-mode environment."""

    def __init__(
        self,
        task_id: int,
        webshop_path: str = None,
        num_products: int = None,
        server=None,
    ):
        """
        Args:
            task_id: The instruction/goal index in WebShop (0-indexed).
            webshop_path: Path to the cloned WebShop repo. If None, tries
                          WEBSHOP_PATH env var or defaults to ./webshop.
            num_products: Number of products to load (None = all).
            server: Optional pre-loaded SimServer instance to share across
                    multiple envs (avoids reloading 1.18M products).
        """
        self.task_id = task_id
        self.webshop_path = webshop_path or os.getenv(
            "WEBSHOP_PATH",
            os.path.join(os.path.dirname(__file__), "..", "..", "webshop"),
        )
        self.num_products = num_products
        self._shared_server = server
        self.env = None
        self._init_env()

    def _init_env(self):
        """Initialize the WebShop gym environment."""
        # Add WebShop to sys.path so we can import its modules
        webshop_abs = os.path.abspath(self.webshop_path)
        if webshop_abs not in sys.path:
            sys.path.insert(0, webshop_abs)

        import gym  # noqa: E402
        try:
            from web_agent_site.envs import WebAgentTextEnv  # noqa: E402, F401
        except ImportError:
            raise ImportError(
                f"Could not import WebShop from {webshop_abs}. "
                "Make sure WebShop is installed. See scripts/setup_webshop.sh"
            )

        env_kwargs = {
            "observation_mode": "text",
            "num_products": self.num_products,
        }
        if self._shared_server is not None:
            env_kwargs["server"] = self._shared_server
        # Filter out None values
        env_kwargs = {k: v for k, v in env_kwargs.items() if v is not None}

        self.env = gym.make("WebAgentTextEnv-v0", **env_kwargs)

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment to the task specified by task_id.

        Returns:
            observation: Text description of the initial page (search page with instruction).
            info: Dictionary with task metadata.
        """
        # WebShop gym env uses session-based indexing;
        # we pass the goal index to get a specific task
        result = self.env.reset(session=self.task_id)

        # gym 0.24 wraps reset to return (obs, info_or_None)
        if isinstance(result, tuple):
            observation = result[0]
        else:
            observation = result

        # Ensure observation is a string
        if not isinstance(observation, str):
            observation = str(observation)

        info = {
            "task_id": self.task_id,
            "goal": getattr(self.env, "goal", ""),
        }
        return observation, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Take an action in the WebShop environment.

        Args:
            action: Action string, e.g. 'search[blue shirt]' or 'click[Buy Now]'

        Returns:
            observation: Text description of the resulting page.
            reward: 0-1 score based on attribute match quality.
            done: Whether the episode is finished (Buy clicked or max steps).
            info: Additional info dict.
        """
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def get_server(self):
        """Return the underlying SimServer for sharing across envs."""
        return self.env.server

    def close(self):
        """Clean up the gym environment."""
        if self.env:
            self.env.close()
            self.env = None

    @staticmethod
    def get_task_instruction(observation: str) -> str:
        """
        Extract the task instruction from an initial observation.

        WebShop observations start with:
          "WebShop [SEP] Instruction: [SEP] <instruction> [SEP] ..."

        Args:
            observation: The initial observation from reset().

        Returns:
            The task instruction string.
        """
        if "[SEP]" in observation:
            parts = observation.split("[SEP]")
            for i, part in enumerate(parts):
                if "Instruction:" in part.strip() and i + 1 < len(parts):
                    return parts[i + 1].strip()
            # Fallback: return second segment
            if len(parts) >= 3:
                return parts[2].strip()
        return observation.strip()
