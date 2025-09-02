import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pettingllms.multi_agent_env.base.env import MultiAgentsEnvironment
from pettingllms.utils.logger_config import get_multi_logger
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from .utils import load_config_file
import ray
logger = logging.getLogger(__name__)

def extract_task(text_obs: str):
        
    task_start = text_obs.find('Your task is to: ')
    task_description = text_obs[task_start + len('Your task is to: '):].strip()
    return task_description
            


@ray.remote(num_cpus=0.002)
class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
        obs, infos=self.env.reset()
        self.task_description=extract_task(obs)
        self.observation=obs
        self.admissible_actions=infos['admissible_commands']
        self.action_history=[]
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        self.action_history.append(action)
        
        obs, scores, dones, infos = self.env.step(actions)
        self.observation=obs
        self.admissible_actions=infos['admissible_commands']
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    


from typing import List

class AlfWorldEnvBatch:
    """
   
    """
    def __init__(
        self,
        env_idx_list: List[int],
        rollout_idx_list: List[int],
        samples: int,
        max_turns: int,
        config: dict,
        mode: str = "train",
        *,
        env_workers: List = None,
    ):
        self.mode = mode
        self.env_list=[]
        valid_env_list_num=150
        alf_config_path = "pettingllms/multi_agent_env/alfworld/configs/config_tw.yaml"
        config = load_config_file(alf_config_path)
        #env_type = config['env']['type']
        base_env = AlfredTWEnv(config, train_eval='train' if mode == 'train' else 'valid')
        if mode == "train":
            for rollout_idx in rollout_idx_list:
                env_worker=AlfworldWorker.remote(config, 0+rollout_idx//samples, base_env)
                self.env_list.append(env_worker)

        else:
            for rollout_idx in range(valid_env_list_num):
                env_worker=AlfworldWorker.remote(config, 0+rollout_idx, base_env)
                self.env_list.append(env_worker)
