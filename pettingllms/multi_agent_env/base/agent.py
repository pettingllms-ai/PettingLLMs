from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, List

from pettingllms.multi_agent_env.base.env import Env

@dataclass
class AgentData:
    current_prompt: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"text": None, "image": None}
    )
    current_action: Optional[Any] = None
    agent_reward: Optional[float] = 0.0
    success: bool = False
    done: bool = False
    answer_history: Optional[List[Any]] = field(default_factory=list)
    action_history: Optional[List[Any]] = field(default_factory=list)
    reward_history: Optional[List[float]] = field(default_factory=list)
    if_trained: bool = True
    skip_current_turn: bool = False
    
    

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
        history: Optional[Any] = None
        current_prompt: Optional[Dict[str, Any]] = field(
            default_factory=lambda: {"text": None, "image": None}
        )
        current_action: Optional[Any] = None
        current_observation: Optional[Any] = None
        info: Optional[Dict[str, Any]] = None
        agent_reward: Optional[float] = None
        done: bool = False
        reward_history: Optional[List[float]] = field(default_factory=list)
        is_pass: bool = False

        Returns:
            None
        """
        self.current_action = None
        self.current_prompt = None
        self.current_observation = None
        self.info = None
        self.agent_reward = None
        self.reward_history = []
        self.success = False


