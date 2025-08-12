from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from pettingllms.multi_agent_env.base.env import Env

@dataclass
class AgentData:
    history: Optional[Any] = None
    current_prompt: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"text": None, "image": None}
    )
    current_action: Optional[Any] = None
    current_observation: Optional[Any] = None
    info: Optional[Dict[str, Any]] = None
    agent_reward: Optional[float] = None
    done: bool = False

class Agent(AgentData):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def update_from_env(self, env_data: Env, **kwargs)-> Env:
        """
        Updates the agent's internal state after an environment step.

        Args:
            env_data (EnvData): The environment data after stepping through environment.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def update_from_model(self, env_data: Env, **kwargs) -> Env:
        """
        Updates the agent's internal state after the model generates a response.

        Args:
            response (str): The response from the model.

        Returns:
            None
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def reset(self):
        """
        Resets the agent's internal state, typically called at the beginning of a new episode.

        This function should clear any stored history or state information necessary
        for a fresh interaction.

        Returns:
            None
        """
        return

    def get_current_state(self) -> Optional[Any]:
        """
        Returns the agent's current state as a dictionary.

        This method provides access to the agent's internal state at the current step,
        which can be useful for debugging, logging, or state management.

        Returns:
            Step: The agent's current state.
        """
        if not self.trajectory.steps:
            return None
        return self.trajectory.steps[-1]
