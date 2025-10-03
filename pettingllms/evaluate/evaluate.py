import argparse
import asyncio
import json
import os
import sys
import time
import uuid
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np
import sys
import asyncio


from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


# ------------------------------
# Internal imports for multi-agent validate path
# ------------------------------
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from pettingllms.trainer.multi_agents_execution_engine import MultiAgentsExecutionEngine
import pettingllms.trainer.multi_agents_execution_engine as mae_engine
from pettingllms.trainer import utils as trainer_utils
from verl.utils import hf_tokenizer, hf_processor
from pettingllms.trainer.utils import convert_prompt_to_dpr, convert_dpr_to_response, llm_async_generate
from pettingllms.trainer.multi_agents_execution_engine import MultiAgentsExecutionEngine
import asyncio
import json
import math
import os
import uuid
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
import time
from tqdm import tqdm
import numpy as np
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import load_reward_manager
from pettingllms.trainer.multi_agents_execution_engine import MultiAgentsExecutionEngine
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.trainer.ppo.ray_trainer import (
    
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    reduce_metrics,
)

from pettingllms.verl.ray_trainer import RayPPOTrainer
from verl.utils.torch_functional import pad_sequence_to_length
from typing import Dict
from pettingllms.utils.profiler.performance import simple_timer
import ray
from omegaconf import DictConfig
import hydra

from pprint import pprint

from omegaconf import OmegaConf

from verl.utils.fs import copy_local_path_from_hdfs
# Initialize tokenizer dictionary for multiple models
from verl.utils import hf_tokenizer, hf_processor
import subprocess
import socket
import os
from pathlib import Path
from typing import Optional


def init_agent_execution_engine(config: DictConfig, address: str):
    ppo_trainer_config_dict = {}
    tokenizer_dict = {}
    processor_dict = {}
    server_address_dict = {}
    agent_policy_mapping = {}
    for agent_key, agent_config in config.agent_policy_configs.agent_configs.items():
                agent_name = agent_config.name
                policy_name = agent_config.policy_name
                agent_policy_mapping[agent_name] = policy_name
               
   
    address_map = getattr(config, 'address_map', {}) if hasattr(config, 'address_map') else {}

    for i, (model_key, model_config) in enumerate(config.models.items()):
        model_name = model_config.name
        model_path = model_config.path
        
        if hasattr(model_config, 'ppo_trainer_config'):
            ppo_trainer_config = model_config.ppo_trainer_config
            ppo_trainer_config_dict[model_name] = ppo_trainer_config
            local_path = copy_local_path_from_hdfs(model_path)
            
           
            trust_remote_code = getattr(model_config, 'trust_remote_code', False)
            if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
                trust_remote_code = config.resource.trust_remote_code
            # Create tokenizer for this model
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
            tokenizer_dict[model_name] = tokenizer
            processor_dict[model_name] = processor
            ppo_trainer_config = model_config.ppo_trainer_config
            ppo_trainer_config_dict[model_name] = ppo_trainer_config
            policy_addr = address_map.get(model_name, address)
            server_address_dict[model_name] = [policy_addr]
            

    agent_execution_engine = MultiAgentsExecutionEngine(config=config, ppo_trainer_config_dict=ppo_trainer_config_dict, tokenizer_dict=tokenizer_dict, processor_dict=processor_dict, server_address_dict=server_address_dict, agent_policy_mapping=agent_policy_mapping)
    return agent_execution_engine

def validate(config: DictConfig, address: str):
    agent_execution_engine = init_agent_execution_engine(config, address)
    agent_execution_engine.init_agents_and_envs(mode="validate")
    batch_per_trainer: Dict[str,DataProto]={}
    gen_batch_output_per_policy =asyncio.run( agent_execution_engine.generate_multiple_rollouts_concurrent(agent_execution_engine.env_idx_list))
    for model_name in agent_execution_engine.ppo_trainer_config_dict.keys():
        if model_name not in batch_per_trainer or batch_per_trainer[model_name].batch is None:
        # If empty, assign directly
            batch_per_trainer[model_name] = gen_batch_output_per_policy[model_name]
        else:
            # Use concat instead of union, because each response content is different
            batch_per_trainer[model_name] = DataProto.concat([
                batch_per_trainer[model_name], 
                gen_batch_output_per_policy[model_name]
            ])

    total_rollout_num = len(agent_execution_engine.rollout_idx_list)
    success_rollout_rate_dict: Dict[str, float] = {}
    for agent_name in agent_execution_engine.turn_order:
        success_rollout_num = len(
            agent_execution_engine.success_rollout_idx_list_dict.get(agent_name, [])
        )
        success_rollout_rate_dict[agent_name] = (
            success_rollout_num / total_rollout_num if total_rollout_num > 0 else 0.0
        )
    return agent_execution_engine.success_rollout_idx_list_dict,success_rollout_rate_dict


def test(config: DictConfig, address: str):
    prompt = "Hello, who are you?"
    model_path = config.models.model_0.path
    local_path = copy_local_path_from_hdfs(model_path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=False)
    prompt_dpr = convert_prompt_to_dpr(
        tokenizer=tokenizer,
        processor=None,
        prompts={"text": prompt, "image": None},
        max_prompt_length=config.data.max_prompt_length,
        multi_modal=False,
    )
    print("prompt_dpr")
    print(prompt_dpr)
    response = asyncio.run(llm_async_generate(
        rollout_idx=0,
        turn_idx=0,
        agent_idx=0,
        enable_thinking=False,
        prompt_dpr=prompt_dpr,
        address=address,
        model_name=model_path,
        tokenizer=tokenizer,
        ppo_trainer_config=config.models.model_0.ppo_trainer_config,
    ))
    print(response)


@hydra.main(config_path="../../pettingllms/config/math", config_name="math_single_policy", version_base=None)
def main(config: DictConfig):
    address = getattr(config, 'vllm_address', '127.0.0.1:8220')
    print(f"🚀 Using vLLM service address: {address}")
   
    success_rollout_idx_list_dict,success_rollout_rate_dict = validate(config, address)
    with open("success_rollout_idx_list_dict.json", "a") as f:
        json.dump(success_rollout_idx_list_dict, f)
    with open("success_rollout_rate_dict.json", "a") as f:
        json.dump(success_rollout_rate_dict, f)
    with open("success_rollout_idx_list_dict.txt", "a") as f:
        for agent_name, idx_list in success_rollout_idx_list_dict.items():
            f.write(f"{agent_name}: {idx_list}\n")
    with open("success_rollout_rate_dict.txt", "a") as f:
        text=f"Model path: {config.models.model_0.path}\n"
        text+=f"Enable thinking: {config.enable_thinking}\n"
        text+=f"Max turns: {config.env.max_turns}\n"
        text+=f"Benchmark: {config.benchmark}\n"
        for agent_name, rate in success_rollout_rate_dict.items():
            f.write(f"{agent_name}: {rate}\n")

if __name__ == "__main__":
    main()