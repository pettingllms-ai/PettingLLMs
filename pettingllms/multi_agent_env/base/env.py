from abc import abstractmethod
from typing import List


class Env:
    """
    An environment for multi-turn interactions with LLMs.
    The environment provides a series of questions/prompts and evaluates responses using a custom reward function.
    The interaction terminates after reaching the maximum number of turns.
    """

    def __init__(self, env_idx: int, rollout_idx: int, config: dict | None = None):
        """
        Initialize the multi-agents environment.

        Args:
            env_idx: Environment index
            rollout_idx: Rollout index
            config: Configuration for the system
        """
        # Save configuration
        self.config = config

        # Initialize variables required by step method
        self.done = False
        self.state = None
        self.success = False


        
        
    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Response string from the LLM

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        return NotImplementedError("Subclasses must implement this method")


class EnvBatch:
    def __init__(self, env_idx_list: List[int], rollout_idx_list: List[int]):
        self.env_list = []
        for env_idx, rollout_idx in zip(env_idx_list, rollout_idx_list):
            env = Env(env_idx, rollout_idx)
            self.env_list.append(env)
