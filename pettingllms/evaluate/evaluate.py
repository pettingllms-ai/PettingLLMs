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
# MultiAgentsExecutionEngineGraph will be conditionally imported based on workflow_type
import pettingllms.trainer.multi_agents_execution_engine as mae_engine
from pettingllms.trainer import async_generate as trainer_utils
from verl.utils import hf_tokenizer, hf_processor
from pettingllms.trainer.async_generate import convert_prompt_to_dpr, convert_dpr_to_response, llm_async_generate
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

from pettingllms.verl.ray_trainer import RayPPOTrainer
from verl.utils.torch_functional import pad_sequence_to_length
from typing import Dict
from pettingllms.utils.performance import simple_timer
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
    """Initialize agent execution engine based on workflow_type in config."""
    # Initialize basic dictionaries
    ppo_trainer_config_dict = {}
    tokenizer_dict = {}
    processor_dict = {}
    server_address_dict = {}
    agent_policy_mapping = {}
    
    # Build agent_policy_mapping
    for agent_key, agent_config in config.agent_policy_configs.agent_configs.items():
        agent_name = agent_config.name
        policy_name = agent_config.policy_name
        agent_policy_mapping[agent_name] = policy_name
    
    # Get address mapping
    address_map = getattr(config, 'address_map', {}) if hasattr(config, 'address_map') else {}

    # Initialize models
    for i, (model_key, model_config) in enumerate(config.models.items()):
        # Skip models without name or path (they may only have ppo_trainer_config)
        if not hasattr(model_config, 'name') or not hasattr(model_config, 'path'):
            continue
            
        model_name = model_config.name
        model_path = model_config.path
        
        if hasattr(model_config, 'ppo_trainer_config'):
            ppo_trainer_config = model_config.ppo_trainer_config
            ppo_trainer_config_dict[model_name] = ppo_trainer_config
            local_path = copy_local_path_from_hdfs(model_path)
            
            trust_remote_code = getattr(model_config, 'trust_remote_code', False)
            if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
                trust_remote_code = config.resource.trust_remote_code
            
            # Create tokenizer and processor
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
            tokenizer_dict[model_name] = tokenizer
            processor_dict[model_name] = processor
            
            policy_addr = address_map.get(model_name, address)
            server_address_dict[model_name] = [policy_addr]
    
    # Detect LoRA differ mode for multi-agent LoRA evaluation
    lora_differ_mode = False
    agent_lora_mapping = {}

    if hasattr(config, 'specialization') and config.specialization == "lora":
        print("=" * 60)
        print("LoRA Specialization Detected")
        print("=" * 60)

        # Check if lora_paths is provided (from command line)
        if hasattr(config, 'lora_paths') and config.lora_paths:
            # Parse lora_paths (comma-separated string)
            lora_paths_list = config.lora_paths.split(',')
            num_agents = len(agent_policy_mapping)

            # Validate that number of LoRA paths matches number of agents
            if len(lora_paths_list) != num_agents:
                raise ValueError(
                    f"Number of LoRA paths ({len(lora_paths_list)}) does not match "
                    f"number of agents ({num_agents}). "
                    f"LoRA paths: {lora_paths_list}, "
                    f"Agents: {list(agent_policy_mapping.keys())}"
                )

            lora_differ_mode = True
            print("LoRA Differ Mode ENABLED for Evaluation")
            print("Each agent will use a different LoRA adapter")
            print(f"Number of agents: {num_agents}")
            print(f"Number of LoRA adapters: {len(lora_paths_list)}")

            # Map agents to LoRA adapters based on agent_configs order (agent_0, agent_1, ...)
            # This ensures consistent mapping regardless of dictionary iteration order
            agent_config_items = sorted(
                config.agent_policy_configs.agent_configs.items(),
                key=lambda x: int(x[0].split('_')[1])  # Sort by agent number: agent_0, agent_1, ...
            )
            print(f"Agent config order: {[item[0] for item in agent_config_items]}")

            for agent_idx, (agent_key, agent_config) in enumerate(agent_config_items):
                agent_name = agent_config.name
                lora_id = agent_idx+1
                agent_lora_mapping[agent_name] = lora_id
                print(f"  Agent '{agent_name}' (from {agent_key}) -> LoRA adapter 'lora_{lora_id}' (ID: {lora_id})")

            print(f"Total {len(agent_lora_mapping)} agent-specific LoRA adapters")
            print("=" * 60)
        else:
            raise ValueError(
                "LoRA speciaslization is set, but no lora_paths provided in config. "
                "Please provide --config.lora_paths with comma-separated LoRA adapter paths."
            )
    # Get workflow_type from config, default to "turn"
    workflow_type = getattr(config, 'workflow_type', 'turn')
    print(f"Using workflow_type: {workflow_type}")

    # In evaluation mode with LoRA, we want to use the LoRA adapters for generation
    use_lora_for_generation = lora_differ_mode

    # Select the appropriate execution engine based on workflow_type
    if workflow_type == "graph":
        print("Initializing MultiAgentsExecutionEngineGraph")
        from pettingllms.trainer.multi_agents_execution_engine_graph import MultiAgentsExecutionEngineGraph
        agent_execution_engine = MultiAgentsExecutionEngineGraph(
            config=config,
            ppo_trainer_config_dict=ppo_trainer_config_dict,
            tokenizer_dict=tokenizer_dict,
            processor_dict=processor_dict,
            server_address_dict=server_address_dict,
            agent_policy_mapping=agent_policy_mapping,
            lora_differ_mode=lora_differ_mode,
            agent_lora_mapping=agent_lora_mapping,
            use_lora_for_generation=use_lora_for_generation,
        )
    elif workflow_type == "autoevol":
        print("Initializing MultiAgentsExecutionEngineAutoEvol")
        from pettingllms.trainer.multi_agents_execution_engine_autoevol import MultiAgentsExecutionEngineAutoEvol
        agent_execution_engine = MultiAgentsExecutionEngineAutoEvol(
            config=config,
            ppo_trainer_config_dict=ppo_trainer_config_dict,
            tokenizer_dict=tokenizer_dict,
            processor_dict=processor_dict,
            server_address_dict=server_address_dict,
            agent_policy_mapping=agent_policy_mapping,
            lora_differ_mode=lora_differ_mode,
            agent_lora_mapping=agent_lora_mapping,
            use_lora_for_generation=use_lora_for_generation,
        )
    else:
        # Default to "turn" workflow
        print("Initializing MultiAgentsExecutionEngine (turn-based)")
        agent_execution_engine = MultiAgentsExecutionEngine(
            config=config,
            ppo_trainer_config_dict=ppo_trainer_config_dict,
            tokenizer_dict=tokenizer_dict,
            processor_dict=processor_dict,
            server_address_dict=server_address_dict,
            agent_policy_mapping=agent_policy_mapping,
            lora_differ_mode=lora_differ_mode,
            agent_lora_mapping=agent_lora_mapping,
            use_lora_for_generation=use_lora_for_generation,
        )

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
    env_success_rollout_idxs = [
        rollout_idx
        for rollout_idx, env in zip(agent_execution_engine.rollout_idx_list, agent_execution_engine.envs)
        if hasattr(env, "success") and env.success
    ]
    env_success_rate = (
        len(env_success_rollout_idxs) / total_rollout_num if total_rollout_num > 0 else 0.0
    )
    return agent_execution_engine, env_success_rollout_idxs, env_success_rate



@hydra.main(config_path="../config/math", config_name="math_L1_prompt", version_base=None)
def main(config: DictConfig):
    address = getattr(config, 'vllm_address', '127.0.0.1:8220')
    print(f"Using vLLM service address: {address}")
   
    agent_execution_engine, env_success_rollout_idxs, env_success_rate = validate(config, address)
    
    # Log success_rollout information to summary logger
    evaluation_summary = {
        "model_path": config.models.model_0.path,
        "benchmark": config.env.benchmark,
        "env_success_rollout_idxs": env_success_rollout_idxs,
        "env_success_rate": env_success_rate,
        "agent_enable_thinking": {}
    }
    
    # Collect enable_thinking configuration for each agent
    for agent_key, agent_config in config.agent_policy_configs.agent_configs.items():
        agent_name = agent_config.name
        enable_thinking = getattr(agent_config, 'enable_thinking', False)
        evaluation_summary["agent_enable_thinking"][agent_name] = enable_thinking
    
    # Log to summary via multi_logger
    agent_execution_engine.multi_logger.log_evaluation_summary(
        mode="validate",
        evaluation_summary=evaluation_summary
    )
    
    print("Evaluation Summary:")
    print(f"  Model path: {evaluation_summary['model_path']}")
    print(f"  Benchmark: {evaluation_summary['benchmark']}")
    print(
        f"    env: {env_success_rate:.4f} "
        f"({len(env_success_rollout_idxs)}/{len(agent_execution_engine.rollout_idx_list)})"
    )

if __name__ == "__main__":
    main()
