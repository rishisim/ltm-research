"""
InterCode SQL Environment Wrapper

Wraps InterCode's SqlEnv into the BaseEnv interface used by the LTM-Research
framework.

InterCode-SQL uses the Spider dataset. The agent issues SQL queries against a
MySQL database running inside Docker. Observations are query execution results
(table rows or error messages). The episode ends when the agent issues a
"submit" action, and reward is computed as IoU between agent and gold results.

Setup:
  1. pip install intercode-bench
  2. Install and start Docker Desktop
  3. Build the SQL Docker image: from intercode.assets import sql_build_docker; sql_build_docker()
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.base import BaseEnv

logger = logging.getLogger(__name__)


# Module-level preprocess function with required annotations for InterCode.
# InterCode checks annotations: record: Dict -> str
def _preprocess_sql(record: Dict) -> str:
    """Switch to the correct database before each episode."""
    extra = record.get("extra", {})
    db = extra.get("db", "")
    if db:
        return f"USE {db}"
    return ""


# Track the db name from the most recent preprocess call
_last_db_name = ""

def _preprocess_sql_tracking(record: Dict) -> str:
    """Preprocess that also tracks the db name for the wrapper."""
    global _last_db_name
    extra = record.get("extra", {})
    _last_db_name = extra.get("db", "")
    if _last_db_name:
        return f"USE {_last_db_name}"
    return ""


class InterCodeSQLEnv(BaseEnv):
    """BaseEnv wrapper for InterCode SQL environment."""

    # Shared SqlEnv instance across all tasks (avoids Docker container churn)
    _shared_env = None
    _shared_env_data_path = None

    def __init__(
        self,
        task_index: int,
        data_path: str = None,
        image_name: str = "docker-env-sql-spider",
        traj_dir: str = None,
        verbose: bool = False,
    ):
        """
        Args:
            task_index: Index into the InterCode SQL dataset (CSV row index).
            data_path: Path to the InterCode SQL data CSV file.
                       If None, uses InterCode's default test data.
            image_name: Docker image name for the SQL environment.
            traj_dir: Optional directory for InterCode's internal trajectory logs.
            verbose: Whether to enable InterCode's verbose output.
        """
        self.task_index = task_index
        self.data_path = data_path
        self.image_name = image_name
        self.traj_dir = traj_dir
        self.verbose = verbose
        self.env = None
        self._task_query = ""
        self._db_name = ""
        self.last_reward = 0.0
        self._init_env()

    def _init_env(self):
        """Initialize or reuse the InterCode SQL environment."""
        try:
            from intercode.envs import SqlEnv
            from intercode.assets import sql_test_data
        except ImportError:
            raise ImportError(
                "Could not import InterCode. Install with: pip install intercode-bench\n"
                "Also ensure Docker is running and the SQL image is built."
            )

        # Default to built-in test data if no path given
        if not self.data_path:
            self.data_path = sql_test_data

        # Reuse existing env if same data_path (avoids restarting Docker)
        if (InterCodeSQLEnv._shared_env is not None
                and InterCodeSQLEnv._shared_env_data_path == self.data_path):
            self.env = InterCodeSQLEnv._shared_env
            return

        kwargs = {
            "image_name": self.image_name,
            "data_path": self.data_path,
            "preprocess": _preprocess_sql_tracking,
            "verbose": self.verbose,
        }
        if self.traj_dir:
            kwargs["traj_dir"] = self.traj_dir

        self.env = SqlEnv(**kwargs)
        InterCodeSQLEnv._shared_env = self.env
        InterCodeSQLEnv._shared_env_data_path = self.data_path

    # Maximum retry attempts and base backoff (seconds) for the preprocess race.
    _RESET_MAX_RETRIES: int = 5
    _RESET_BACKOFF_BASE: float = 1.0

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """
        Reset the environment to the task at self.task_index.

        Wraps the underlying env.reset() with exponential-backoff retries to
        absorb the transient ``USE <db>`` race that fires when multiple SQL
        processes share a single Docker MySQL container.  The underlying error
        is a plain RuntimeError whose message starts with
        "Preprocess command failed to execute successfully".

        Returns:
            observation: The natural-language question the agent must answer
                         with SQL, plus database and schema context.
            info: Dictionary with task metadata.
        """
        global _last_db_name

        last_exc: Optional[RuntimeError] = None
        for attempt in range(self._RESET_MAX_RETRIES):
            try:
                self.env.reset(self.task_index)
                break  # success
            except RuntimeError as exc:
                if "Preprocess command failed" not in str(exc):
                    raise  # unrelated error — don't swallow
                last_exc = exc
                wait = self._RESET_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "[InterCodeSQLEnv] Retrying preprocess (attempt %d/%d, "
                    "task_index=%d, backoff=%.1fs): %s",
                    attempt + 1, self._RESET_MAX_RETRIES,
                    self.task_index, wait, exc,
                )
                time.sleep(wait)
        else:
            # All retries exhausted — re-raise so the runner records FAIL.
            raise last_exc  # type: ignore[misc]

        self._task_query = self.env.query if hasattr(self.env, "query") else ""
        self._db_name = _last_db_name

        # After reset + preprocess, env.observation may be overwritten by the
        # USE db command. Use env.query which holds the NL question.
        observation = self._task_query or str(self.env.observation or "")

        # Build a richer observation with database context
        obs_parts = []
        if self._db_name:
            obs_parts.append(f"Database: {self._db_name}")
        obs_parts.append(f"Question: {observation}")

        # Get table listing for schema context
        schema_info = self._get_schema_info()
        if schema_info:
            obs_parts.append(f"Tables:\n{schema_info}")

        full_observation = "\n".join(obs_parts)

        info = {
            "task_index": self.task_index,
            "query": self._task_query,
            "db": self._db_name,
        }
        return full_observation, info

    def _get_schema_info(self) -> str:
        """Retrieve table listing for the current database."""
        if not self.env or not self._db_name:
            return ""
        try:
            self.env.exec_action("SHOW TABLES")
            tables_obs = self.env.observation
            if tables_obs is None:
                return ""
            if isinstance(tables_obs, list):
                # Each row is a tuple like ('table_name',)
                return "\n".join(f"  {row[0]}" for row in tables_obs if row)
            return str(tables_obs)
        except Exception:
            return ""

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute a SQL action in the environment.

        Args:
            action: SQL query string, or "submit" to end the episode.

        Returns:
            observation: Query results or error message.
            reward: 0.0 during interaction, IoU score on "submit".
            done: True if "submit" was issued.
            info: Additional metadata.
        """
        # Intercept think: actions — don't send them to the SQL environment
        # as they would be executed as SQL and corrupt the observation state.
        if action.startswith("think:"):
            return "OK.", 0.0, False, {}

        observation, reward, done, info = self.env.step(action)

        # Normalize observation to string
        if observation is None:
            observation = "(no output)"
        elif isinstance(observation, list):
            # Format table results readably
            if len(observation) == 0:
                observation = "(empty result set)"
            else:
                rows = [str(row) for row in observation]
                observation = "\n".join(rows)
        elif not isinstance(observation, str):
            observation = str(observation)

        reward = float(reward) if reward else 0.0
        self.last_reward = reward

        return observation, reward, done, info

    def close(self):
        """
        Clean up. Note: we don't stop the shared Docker container here
        since other tasks may reuse it. Call close_shared() when done.
        """
        # Don't close shared env — it will be reused
        self.env = None

    @classmethod
    def close_shared(cls):
        """Stop the shared Docker container. Call after all tasks are done."""
        if cls._shared_env is not None:
            try:
                cls._shared_env.close()
            except Exception:
                pass
            cls._shared_env = None
            cls._shared_env_data_path = None

    @staticmethod
    def get_task_description(observation: str) -> str:
        """
        Extract the task question from an initial observation.

        Args:
            observation: The initial observation from reset().

        Returns:
            The question string.
        """
        for line in observation.split("\n"):
            if line.startswith("Question: "):
                return line[len("Question: "):].strip()
        return observation.strip()

    @staticmethod
    def get_dataset_size(data_path: str) -> int:
        """Return the number of tasks in a given InterCode SQL data CSV."""
        try:
            import pandas as pd
            data = pd.read_csv(data_path)
            return len(data)
        except Exception:
            return 0
