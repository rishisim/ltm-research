from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from src.core.history import EnvironmentHistory

class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> Tuple[str, Dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        pass

    @abstractmethod
    def close(self):
        pass

class Framework(ABC):
    @abstractmethod
    def run(self, env: BaseEnv, base_prompt: str, memory: List[str]) -> Tuple[EnvironmentHistory, bool]:
        pass
