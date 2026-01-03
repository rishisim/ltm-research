import alfworld
import alfworld.agents.environment
from typing import List, Dict, Any, Tuple
from src.core.base import BaseEnv

class AlfworldEnv(BaseEnv):
    def __init__(self, config: Dict[str, Any], split: str = "eval_out_of_distribution"):
        self.config = config
        self.split = split
        self.env = None
        self._init_env()

    def _init_env(self):
        self.env = alfworld.agents.environment.get_environment(self.config["env"]["type"])(self.config, train_eval=self.split)
        self.env = self.env.init_env(batch_size=1)

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        ob, info = self.env.reset()
        # Clean up observation
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        return ob, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        observation, reward, done, info = self.env.step([action])
        obs_str = observation[0]
        # Custom processing for Alfworld observations
        if obs_str.startswith('You arrive at loc '):
            obs_str = obs_str[obs_str.find('. ')+2:]
        return obs_str, reward[0], done[0], info

    def close(self):
        if self.env:
            self.env.close()
