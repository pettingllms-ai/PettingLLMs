import asyncio
import concurrent.futures
import logging
import time
import json
import traceback
import uuid
try:
    from verl.protocol import DataProto
except Exception:  # fallback when verl is a src tree: verl/verl/protocol.py
    from verl.verl.protocol import DataProto
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import openai
import torch
from openai.types import Completion
from pettingllms.trainer.multiagentssys_register import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING, ENV_BATCH_CLASSES
from functools import partial
import multiprocessing

from pettingllms.multi_agent_env.base.env import Env, EnvBatch
from pettingllms.misc import colorful_print
from pettingllms.parser.chat_template.parser import ChatTemplateParser
from pettingllms.router.router import Router
from pettingllms.trainer.utils import convert_prompt_to_dpr, convert_dpr_to_response
from threading import Thread


logger = logging.getLogger(__name__)




class MultiAgentsExecutionEngine:
    def _load_config_parameters(self):
        """Load parameters from config with fallback to defaults"""
        
        
        # Data configuration - direct access with fallbacks
        if hasattr(self.config, 'data') and self.config.data is not None:
            self.max_prompt_length = getattr(self.config.data, 'max_prompt_length', 10240)
            self.max_response_length = getattr(self.config.data, 'max_response_length', 2048)
        else:
            self.max_prompt_length = 10240
            self.max_response_length = 2048
        # Multi-agent interaction configuration - direct access with fallbacks
        if hasattr(self.config, 'multi_agent_interaction') and self.config.multi_agent_interaction is not None:
            self.turn_order = getattr(self.config.multi_agent_interaction, 'turn_order', ['code_generator', 'test_generator'])
            self.num_interacting_agents = getattr(self.config.multi_agent_interaction, 'num_interacting_agents', 2)
            self.shared_observation = getattr(self.config.multi_agent_interaction, 'shared_observation', True)
        else:
            self.turn_order = ['code_generator', 'test_generator']
            self.num_interacting_agents = 2
            self.shared_observation = True
        
        # Rollout configuration - direct access with fallbacks
        if hasattr(self.config, 'data') and self.config.data is not None:
            self.sample_temperature = getattr(self.config.data, 'sample_temperature', 0.7)
            self.gen_batch_size = getattr(self.config.data, 'gen_batch_size', 64)
            self.gen_n_samples = getattr(self.config.data, 'gen_n_samples', 8)
        else:
            self.sample_temperature = 0.7
            self.gen_batch_size = 64
            self.gen_n_samples = 8
    def __init__(
        self,
        config,
        tokenizer_dict=None,
        processor_dict=None,
        router_dict=None,
        agent_policy_mapping=None,
        env_args=None,
        max_workers=64,
        **kwargs,
    ):
        

        self.config = config
        self.tokenizer_dict = tokenizer_dict
        self.processor_dict = processor_dict or {}
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.env_args = env_args or {}
        self.max_workers = max_workers
        # Read parameters from config with fallback to defaults
        self._load_config_parameters()
        self.n_cpu = multiprocessing.cpu_count()

        # Environment configuration - direct access
        if hasattr(self.config, 'env') and self.config.env is not None:
            self.max_turns = getattr(self.config.env, 'max_turns', 8)
            env_name = getattr(self.config.env, 'name', None)
            if env_name is None:
                raise ValueError("env.name is not set in the config.env")
        else:
            raise ValueError("env is not set in the config")
            
        print(f"env_name: {env_name}")
        
        # Store env_name for later use
        self.env_name = env_name
        self.env_class = ENV_CLASS_MAPPING[env_name]
        self.agent_class_list = [AGENT_CLASS_MAPPING[agent_name] for agent_name in self.turn_order]
        self._init_agents_and_envs()
        #self._init_agents()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        # rollout_engine_dict is not maintained in this class to avoid referencing a non-existent attribute
        self.server_addresses_dict = {}
        self.router_dict = router_dict or {}
        self.chat_parser_dict={}
        #for key,value in self.router_dict.items():
        #    self.chat_parser_dict[key]=ChatTemplateParser.get_parser(self.tokenizer_dict[key], disable_thinking=False)
        

    def _init_agents_and_envs(self):
       
        # Check for batched_init in config.env
        if hasattr(self.config, 'env') and self.config.env is not None:
            batched_init = getattr(self.config.env, 'batched_init', True)
        else:
            batched_init = getattr(self.config, 'batched_init', True)
        if batched_init == False:
            with multiprocessing.Pool(self.n_cpu // 4) as pool: # Only use 1/4 of the cores to avoid conflicts
                func=partial(self.__init_one_env_instance, env_args=self.env_args)
                self.envs = pool.map(func, range(self.gen_batch_size*self.gen_n_samples))
            self.env_list = self.envs
            
        else:
            self.env_batch_class=ENV_BATCH_CLASSES[self.env_name]
            self.envs_batch=self.env_batch_class(env_idx_list=range(self.gen_batch_size*self.gen_n_samples), rollout_idx_list=range(self.gen_batch_size*self.gen_n_samples), max_turns=self.max_turns, config=self.config)
            self.envs=self.envs_batch.env_list
            self.env_list = getattr(self.envs, "env_list", [])

        # Initialize the agent group for each rollout
        self.agent_groups = [
            [agent_cls(rollout_idx=rollout_idx) for agent_cls in self.agent_class_list]
            for rollout_idx in range(len(self.env_list))
        ]
        
        
        
    
    def __init_one_env_instance(self, rollout_idx, env_args):
        env = self.env_class( env_idx=rollout_idx//self.gen_batch_size,rollout_idx=rollout_idx,max_turns=self.max_turns, **env_args)
        
        return env
    

        
    async def generate_single_rollout(self, rollout_idx, timing_raw, meta_info):
        """
        Generate a single rollout, adapted for multi-agent interaction in the code testing environment.
        
        Args:
            env: Code testing environment instance
            timing_raw: Timing record dictionary
            meta_info: Meta information
            
        Returns:
            DataProto: DataProto object containing trajectory data
        """
        rollout_id = str(uuid.uuid4())
        trajectory_per_task_dict = {}
        for policy_name in self.router_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()


        env = self.env_list[rollout_idx] if hasattr(self, "env_list") else self.envs[rollout_idx]
        agent_group = self.agent_groups[rollout_idx]
        

        
        for turn_idx in range(self.max_turns):
            for agent_idx, agent_name in enumerate(self.turn_order):
                current_agent = agent_group[agent_idx]
                current_agent.update_from_env(env)
                prompt = current_agent.current_prompt
                # Select the policy name; if not provided, fall back to any available policy
                policy_name = self.agent_policy_mapping.get(agent_name) if self.agent_policy_mapping else None
                if policy_name is None:
                    policy_name = next(iter(self.router_dict.keys())) if self.router_dict else next(iter(self.tokenizer_dict.keys()))
                # Convert to DataProto format
                dpr_prompt = convert_prompt_to_dpr(
                    self.tokenizer_dict[policy_name], 
                    self.chat_parser_dict.get(policy_name), 
                    self.processor_dict.get(policy_name) if isinstance(self.processor_dict, dict) else None,
                    prompt, 
                    self.max_prompt_length,
                    multi_modal=False
                )
                
                # Generate responses
                output_dpr = await self.router_dict[policy_name].generate_sequences(
                    dpr_prompt, 
                    application_id=rollout_id
                )
                
                # Convert response format
                response = convert_dpr_to_response(
                    self.tokenizer_dict[policy_name], 
                    self.chat_parser_dict.get(policy_name), 
                    output_dpr, 
                    self.max_prompt_length
                )
                
                current_agent.update_from_model(response)
                
                env.step(agent_name, current_agent.action)
                
                current_agent.calculate_reward(env,mode="sum")

                output_dpr.non_tensor_batch["reward"] = [current_agent.reward]
                # union returns a new DataProto, so reassign back
                trajectory_per_task_dict[policy_name] = trajectory_per_task_dict[policy_name].union(output_dpr)
        return trajectory_per_task_dict
        
       
    
class AsyncMultiAgentsExecutionEngine(MultiAgentsExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


