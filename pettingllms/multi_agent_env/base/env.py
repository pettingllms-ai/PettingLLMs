from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # For type checking only to avoid runtime circular imports
    from pettingllms.multi_agent_env.base.agent import AgentData




@dataclass
class Env:
    env_idx: int
    rollout_idx: int
    max_turns: int
    turn: int = 0
    global_observations: Optional[Any] = None
    global_infos: Optional[Any] = None
    agent_infos: Optional[Dict[str, Any]] = None
    agent_observations: Optional[Dict[str, Any]] = None
    state: Optional[Any] = None
    termination_reason: Optional[str] = None #TODO:
    is_pass: bool = False
    
class MultiAgentsEnvironment(Env):
    """
    An environment for multi-turn interactions with LLMs.
    The environment provides a series of questions/prompts and evaluates responses using a custom reward function.
    The interaction terminates after reaching the maximum number of turns.
    """

    def __init__(self, env_idx: int, rollout_idx: int, max_turns: int,  config: dict | None = None):
        """
        Initialize the multi-agents environment using the dataclass EnvData.

        Args:
            env_idx: Environment index
            rollout_idx: Rollout index  
            max_turns: Maximum number of turns before terminating
            task: Dictionary containing the task information
            config: Configuration for the system
        """
        super().__init__(env_idx, rollout_idx, max_turns)
        
        # Save configuration
        self.config = config
        
        # Initialize variables required by step method
        self.history = []
        self.task = None
        self.current_turn = 0
        self.done = False
        

    def reset(self):
        """
        Reset the environment to initial state.
        Keeps env_idx, rollout_idx, max_turns unchanged, resets all other values to defaults.
        """
        # Reset default values from Env base class
        self.turn = 0
        self.global_observations = None
        self.global_infos = None
        self.agent_infos = None
        self.agent_observations = None
        self.state = None
        self.termination_reason = None
        self.is_pass = False
        self.agentdata = None
        
        # Reset variables needed by step method
        self.history = []
        self.task = None
        self.current_turn = 0
        self.done = False
        
        # Re-initialize agent_data_dict
        agent_data_dict = {}
        if hasattr(self, 'config') and self.config is not None and 'multi_agent_interaction' in self.config:
            multi_agent_config = self.config['multi_agent_interaction']
            
            # Get agent names from turn_order
            if 'turn_order' in multi_agent_config:
                agent_names = multi_agent_config['turn_order']
                
                # Re-initialize AgentData for each agent
                for agent_name in agent_names:
                    # Lazy import to avoid circular import with Agent -> Env
                    from pettingllms.multi_agent_env.base.agent import AgentData  # noqa: WPS433
                    agent_data_dict[agent_name] = AgentData()
        
        # Update agentdata
        self.agentdata = agent_data_dict
        
        return self

    def step(self, action):
        """
        Take a step in the environment based on the action.

        Args:
            action: Response string from the LLM

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        # Store the action in history
        self.history.append(action)

        # Calculate reward for the current turn using the abstract method
        assert self.task is not None, "Task is not set"
        reward, next_obs = self.get_reward_and_next_obs(self.task, action)

        # Increment turn counter
        self.current_turn += 1

        # Check if we've reached the maximum number of turns
        if self.current_turn >= self.max_turns:
            self.done = True
            return {}, reward, self.done, self.task

        return next_obs, reward, self.done, self.task

    @abstractmethod
    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        """
        Abstract method to compute the reward based on the task and action.

        Args:
            task: The task dictionary containing relevant information
            action: The action taken by the agent

        Returns:
            Tuple of (reward: float, metadata: Dict)
        """
        pass

    @staticmethod
    def from_dict(env_args: dict) -> "MultiAgentsEnvironment":
        raise NotImplementedError("MultiAgentsEnvironment is abstract and cannot be instantiated directly. Use a concrete subclass.")

class EnvBatch:
    def __init__(self, env_idx_list: List[int], rollout_idx: List[int], max_turns: int):
        self.env_list=[]
        for env_idx in env_idx_list:
            env=Env(env_idx, max_turns)
            self.env_list.append(env)

    def reset(self):
        for env in self.env_list:
            env.reset()

    def step(self, action):
        pass