import asyncio
import concurrent.futures
import logging
import time
import json
import traceback
import uuid
from tqdm.asyncio import tqdm
try:
    from verl.protocol import DataProto
except Exception:  # fallback when verl is a src tree: verl/verl/protocol.py
    from verl import DataProto
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
from pettingllms.trainer.utils import convert_prompt_to_dpr, convert_dpr_to_response
from pettingllms.utils.logger_config import get_multi_logger
from threading import Thread


logger = logging.getLogger(__name__)




class MultiAgentsExecutionEngine:
    def _load_config_parameters(self):
        """Load parameters from config with fallback to defaults"""
        
        
        # Data configuration - direct access with fallbacks
        if hasattr(self.config, 'data') and self.config.data is not None:
            self.max_prompt_length = getattr(self.config.data, 'max_prompt_length', 1024)
            self.max_response_length = getattr(self.config.data, 'max_response_length', 1024)
        else:
            self.max_prompt_length = 1024
            self.max_response_length = 1024
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
            
        # Timeout configuration - direct access with fallbacks
        if hasattr(self.config, 'timeout') and self.config.timeout is not None:
            self.generate_timeout = getattr(self.config.timeout, 'generate_timeout', 60.0)
            self.step_timeout = getattr(self.config.timeout, 'step_timeout', 30.0)
        else:
            self.generate_timeout = 120.0  # 60 seconds for generation
            self.step_timeout = 20.0      # 30 seconds for environment step
    def __init__(
        self,
        config,
        tokenizer_dict=None,
        processor_dict=None,
        server_manager_dict=None,
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
        
        # 初始化多日志系统
        self.multi_logger = get_multi_logger()
        
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
        self.mode=self.config.mode
        self._init_agents_and_envs()
        
        #self._init_agents()

        #self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        # rollout_engine_dict is not maintained in this class to avoid referencing a non-existent attribute
        self.server_manager_dict = server_manager_dict or {}
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
           
            
        else:
            self.env_batch_class=ENV_BATCH_CLASSES[self.env_name]
            self.rollout_idx_list=range(self.gen_batch_size*self.gen_n_samples)
            self.envs_batch=self.env_batch_class(env_idx_list=range(self.gen_batch_size), rollout_idx_list=range(self.gen_batch_size*self.gen_n_samples), samples=self.gen_n_samples, max_turns=self.max_turns, config=self.config, mode=self.mode)
            self.envs=self.envs_batch.env_list
           

        # Initialize the agent group for each rollout
        self.agent_groups = [
            [agent_cls(rollout_idx=rollout_idx) for agent_cls in self.agent_class_list]
            for rollout_idx in range(len(self.envs))
        ]
        
        
        
    
    def __init_one_env_instance(self, rollout_idx, env_args):
        env = self.env_class( env_idx=rollout_idx % self.gen_batch_size,rollout_idx=rollout_idx,max_turns=self.max_turns, **env_args)
        
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
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()


        env = self.env_list[rollout_idx] if hasattr(self, "env_list") else self.envs[rollout_idx]
        agent_group = self.agent_groups[rollout_idx]
        
        

        
        self.multi_logger.log_async_event(
            rollout_idx, "rollout_start", 
            f"Starting multi-turn conversation, max turns: {self.max_turns}",
            {
                "turn_order": self.turn_order,
                "available_tokenizers": list(self.tokenizer_dict.keys()),
                "available_server_managers": list(self.server_manager_dict.keys())
            }
        )
        reward_history_dict={}
        for turn_idx in range(self.max_turns):
            self.multi_logger.log_async_event(
                rollout_idx, "turn_start",
                f"Starting turn {turn_idx + 1}",
                {"turn_idx": turn_idx + 1}
            )
            
            for agent_idx, agent_name in enumerate(self.turn_order):
                current_agent = agent_group[agent_idx]
                current_agent.update_from_env(env)
                prompt = current_agent.current_prompt
                
                # Select the policy name; if not provided, fall back to any available policy
                policy_name = self.agent_policy_mapping.get(agent_name) if self.agent_policy_mapping else None
                if policy_name is None:
                    policy_name = next(iter(self.server_manager_dict.keys())) if self.server_manager_dict else next(iter(self.tokenizer_dict.keys()))
                

                # Convert to DataProto format
                dpr_prompt = convert_prompt_to_dpr(self.tokenizer_dict[policy_name], 
                        self.processor_dict.get(policy_name) if isinstance(self.processor_dict, dict) else None,
                        prompt, 
                        self.max_prompt_length,
                       multi_modal=False
                   )
                
                # Generate responses
                generation_success = True
                output_dpr = None
                response_str = None
                try:
                    output_dpr,response_str = await self.server_manager_dict[policy_name].generate(
                            dpr_prompt, 
                            application_id=rollout_id,
                            tokenizer=self.tokenizer_dict[policy_name],
                            rollout_idx=rollout_idx,
                            policy_name=policy_name,
                            timeout=self.generate_timeout
                        )
                      
                    
                except asyncio.TimeoutError:
                    self.multi_logger.log_env_agent_info(
                        rollout_idx, turn_idx + 1, agent_name,
                        f"❌ Generation timed out after {self.generate_timeout}s",
                        {"error": "timeout", "timeout_seconds": self.generate_timeout}
                    )
                    generation_success = False
                except Exception as e:
                    self.multi_logger.log_env_agent_info(
                        rollout_idx, turn_idx + 1, agent_name,
                        f"Failed to generate response: {e}",
                        {"error": str(e), "traceback": traceback.format_exc()}
                    )
                    raise
               
                # Skip processing if generation failed
                if not generation_success:
                    continue
                
                current_agent.update_from_model(response_str)
                
                step_success = True
                try:
                    await asyncio.wait_for(
                        env.step(agent_name, current_agent.current_action),
                        timeout=self.step_timeout
                    )
                except asyncio.TimeoutError:
                    self.multi_logger.log_env_agent_info(
                        rollout_idx, turn_idx + 1, agent_name,
                        f"❌ Environment step timed out after {self.step_timeout}s",
                        {"error": "timeout", "timeout_seconds": self.step_timeout}
                    )
                    step_success = False
                except Exception as e:
                    self.multi_logger.log_env_agent_info(
                        rollout_idx, turn_idx + 1, agent_name,
                        f"Failed to execute environment step: {e}",
                        {"error": str(e), "traceback": traceback.format_exc()}
                    )
                    step_success = False
                
                # Skip processing if step failed
                if not step_success:
                    continue
                
                current_agent.calculate_reward(env,mode="sum")

                reward_history_dict[agent_name] = current_agent.agent_reward if agent_name not in reward_history_dict else reward_history_dict[agent_name].append(current_agent.agent_reward)

                # Only process trajectory if both generation and step succeeded
                if output_dpr is not None:
                    output_dpr.non_tensor_batch["reward"] = [current_agent.agent_reward]
                    output_dpr.non_tensor_batch["agent_name"] = [agent_name]  # Add agent name for metrics tracking
               
                    if trajectory_per_task_dict[policy_name].batch is None:
                        # If empty, assign directly
                        trajectory_per_task_dict[policy_name] = output_dpr
                    else:
                        # Use concat instead of union, because each response content is different
                        trajectory_per_task_dict[policy_name] = DataProto.concat([
                            trajectory_per_task_dict[policy_name], 
                            output_dpr
                        ])
                

                self.multi_logger.log_env_agent_info(
                    rollout_idx, turn_idx + 1, agent_name,
                    "Trajectory information updated",
                    {
                        "agent_name": agent_name,
                        #"agent_prompt": prompt,
                        "agent_response": response_str,
                        "agent_reward": current_agent.agent_reward,
                        "agent_action": str(current_agent.current_action),
                        "reward_history_dict": reward_history_dict
                       
                    }
                )
                if agent_name == self.turn_order[-1]:
                    self.multi_logger.log_env_agent_info(
                        rollout_idx, turn_idx + 1, agent_name,
                        "Trajectory information updated",
                        {
                            "env_state": env.state,
                        }
                    )
                
                # Check if environment is done (e.g., all tests passed)
                if hasattr(env, 'done') and env.done:
                    termination_reason = getattr(env, 'termination_reason', 'environment_done')
                    self.multi_logger.log_async_event(
                        rollout_idx, "early_termination",
                        f"Environment completed early due to: {termination_reason}",
                        {"termination_reason": termination_reason, "completed_turn": turn_idx + 1, "completed_agent": agent_name}
                    )
                    
              
                    agent_rewards = {}
                    for agent_idx, agent_name_summary in enumerate(self.turn_order):
                        agent_rewards[agent_name_summary] = agent_group[agent_idx].agent_reward
                    
                    self.multi_logger.log_rollout_summary(
                        rollout_idx=rollout_idx,
                        agent_rewards=agent_rewards,
                        termination_reason=termination_reason,
                        extra_data={
                            "completed_turn": turn_idx + 1,
                            "completed_agent": agent_name,
                            "env_state": env.state,
                            "reward_history_dict": reward_history_dict
                        }
                    )
                    
                    # Break out of both agent loop and turn loop
                    return trajectory_per_task_dict
        
        extra_data={}
        agent_rewards = {}
        for agent_idx, agent_name in enumerate(self.turn_order):
            extra_data[f"{agent_name}_final_result"]={
                "agent_action": str(agent_group[agent_idx].current_action),
                "agent_reward": agent_group[agent_idx].agent_reward,
            }
            agent_rewards[agent_name] = agent_group[agent_idx].agent_reward
        extra_data["env_state"]=env.state
        

        self.multi_logger.log_rollout_summary(
            rollout_idx=rollout_idx,
            agent_rewards=agent_rewards,
            termination_reason="max_turns_reached",
            extra_data={
                "max_turns": self.max_turns,
                "env_state": env.state,
                "final_actions": {agent_name: str(agent_group[agent_idx].current_action) 
                                for agent_idx, agent_name in enumerate(self.turn_order)}
            }
        )
        
        self.multi_logger.log_async_event(
            rollout_idx, "rollout_complete",
            f"Rollout {rollout_idx} completed successfully",
            extra_data=extra_data
        )
        
        return trajectory_per_task_dict

    async def generate_multiple_rollouts_concurrent(self, rollout_indices):
        max_concurrent_tasks=self.max_workers
        
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def run_single_rollout_with_semaphore(rollout_idx):
           
            async with semaphore:
                self.multi_logger.log_async_event(
                    rollout_idx, "rollout_concurrent_start",
                    f"Starting concurrent rollout {rollout_idx}"
                )
                try:
                    result = await self.generate_single_rollout(rollout_idx, None, None)
                    self.multi_logger.log_async_event(
                        rollout_idx, "rollout_complete",
                        f"Rollout {rollout_idx} completed successfully"
                    )
                    return rollout_idx, result
                except Exception as e:
                    self.multi_logger.log_async_event(
                        rollout_idx, "rollout_error",
                        f"Rollout {rollout_idx} failed",
                        {"error": str(e), "traceback": traceback.format_exc()}
                    )
                    
                    raise Exception(f"Rollout {rollout_idx} failed: {e}") from e
        
        tasks = [
            asyncio.create_task(
                run_single_rollout_with_semaphore(rollout_idx), 
                name=f"rollout_{rollout_idx}"
            )
            for rollout_idx in rollout_indices
        ]
        
        self.multi_logger.log_async_event(
            -1, "concurrent_batch_start", 
            f"Created {len(tasks)} concurrent tasks",
            {"task_count": len(tasks), "rollout_indices": list(rollout_indices)}
        )
        
        # 初始化按policy_name分组的结果字典
        aggregated_results = {}
        for policy_name in self.tokenizer_dict.keys():
            aggregated_results[policy_name] = DataProto()
        
        completed_count = 0
        failed_count = 0
        
  
        task_pbar = tqdm(total=len(tasks), desc="Rollouts", position=1, leave=False)
        
        try:
        
            for completed_task in asyncio.as_completed(tasks):
                try:
                    
                    rollout_idx, rollout_result = await completed_task
                    
        
                    for policy_name, policy_data in rollout_result.items():
                        if policy_data.batch is not None:  
                            if aggregated_results[policy_name].batch is None:
                                aggregated_results[policy_name] = policy_data
                            else:
                                aggregated_results[policy_name] = DataProto.concat([
                                    aggregated_results[policy_name], 
                                    policy_data
                                ])
                    
                    completed_count += 1
                    
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)})")
                    
                    self.multi_logger.log_async_event(
                        rollout_idx, "task_complete",
                        f"Task {rollout_idx} completed",
                        {
                            "completed_count": completed_count,
                            "total_tasks": len(tasks),
                            "progress": f"{completed_count}/{len(tasks)}"
                        }
                    )
                except Exception as e:
                    failed_count += 1
                    # 更新进度条（失败的任务）
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)}, {failed_count} failed)")
                    
                    self.multi_logger.log_async_event(
                        -1, "task_error",
                        f"Task failed with error: {e}",
                        {
                            "failed_count": failed_count,
                            "error": str(e)
                        }
                    )
                    
                    continue
                    
        except Exception as e:
            self.multi_logger.log_async_event(
                -1, "concurrent_batch_error",
                f"Concurrent execution encountered error: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            # 取消所有未完成的任务
            for task in tasks:
                if not task.done():
                    task_name = task.get_name()
                    self.multi_logger.log_async_event(
                        -1, "task_cancel",
                        f"Cancelling task {task_name}"
                    )
                    task.cancel()
            raise
        
        # 关闭进度条
        task_pbar.close()
        
        self.multi_logger.log_async_event(
            -1, "concurrent_batch_complete",
            "Concurrent execution completed",
            {
                "successfully_processed": completed_count,
                "total_rollouts": len(rollout_indices),
                "failed": failed_count,
                "success_rate": f"{completed_count}/{len(rollout_indices)}",
                "aggregated_policies": list(aggregated_results.keys())
            }
        )
        
        return aggregated_results
        
       
    
class AsyncMultiAgentsExecutionEngine(MultiAgentsExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def test_server_manager_simple(trainer_config,config):
    import ray
    import os
    from verl.utils import hf_tokenizer
    from verl.experimental.agent_loop import AgentLoopManager
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
    from verl.utils import hf_tokenizer


    os.environ["VERL_VLLM_DISTRIBUTED_BACKEND"] = "none"

    os.environ["VLLM_USE_V1"] = "1"

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    test_multi_logger = get_multi_logger()
    test_multi_logger.log_async_event(
        -1, "test_start",
        "Starting test_server_manager_simple",
        {
            "trainer_config_type": str(type(trainer_config)),
            "config_type": str(type(config)),
            "model_path": trainer_config.actor_rollout_ref.model.path
        }
    )

    
    if not ray.is_initialized():
        ray.init(num_cpus=4)
    server_list=[]
    print(f"begin to init server list")
    from pettingllms.trainer.utils import initialize_llm_servers
    server_list,server_addresses=initialize_llm_servers(None,AsyncvLLMServer,trainer_config)    

    test_multi_logger.log_async_event(
        -1, "server_manager_init_start", 
        f"Starting to init server manager for server"
    )
    from pettingllms.trainer.utils import AsyncLLMServerManager
    from verl.utils import hf_tokenizer



    model_path_local =trainer_config.actor_rollout_ref.model.path
    tokenizer_local = hf_tokenizer(model_path_local, trust_remote_code=True)
    server_manager = AsyncLLMServerManager(config=trainer_config, server_handles=server_list)
    
    
    tokenizer_dict = {"code_generator": tokenizer_local}
    

    prompt_text = "Hello"
    prompt={"text":prompt_text, "image":None}
    prompt_dpr = convert_prompt_to_dpr(tokenizer_local, processor=None, prompts=prompt, max_prompt_length=trainer_config.actor_rollout_ref.rollout.prompt_length, multi_modal=False)
    output_server_manager = asyncio.run(server_manager.generate(prompt_dpr, tokenizer=tokenizer_local, application_id="test", sampling_params={}))
    test_multi_logger.log_async_event(
        -1, "server_manager_test_complete", 
        "Server manager test completed",
        {"output_type": str(type(output_server_manager))}
    )
    server_manager_dict={}
    server_manager_dict["code_generator"]=server_manager
    

    test_multi_logger.log_async_event(
        -1, "multi_agent_engine_init_start",
        "Initializing Multi-Agent Execution Engine",
        {
            "tokenizer_dict_keys": list(tokenizer_dict.keys()),
            "server_manager_dict_keys": list(server_manager_dict.keys())
        }
    )
    
    # Fix: Pass correct tokenizer_dict
    multi_agent_execution_engine = MultiAgentsExecutionEngine(
        config=config, 
        tokenizer_dict=tokenizer_dict, 
        server_manager_dict=server_manager_dict
    )
    
    test_rollout_indices = range(len(multi_agent_execution_engine.envs))   
    test_multi_logger.log_async_event(
        -1, "concurrent_test_start",
        "Testing concurrent rollout execution with class method",
        {"test_rollout_count": len(list(test_rollout_indices))}
    )
    
    try:
        concurrent_results = asyncio.run(
            multi_agent_execution_engine.generate_multiple_rollouts_concurrent(
                test_rollout_indices
            )
        )
        
    except Exception as e:
        test_multi_logger.log_async_event(
            -1, "concurrent_test_error",
            "Class method concurrent execution failed",
            {"error": str(e), "traceback": traceback.format_exc()}
        )



       

    return None



def test_rollout_engine_simple():
    from omegaconf import OmegaConf
    trainer_config = OmegaConf.load("pettingllms/config/code/ppo_trainer/eval.yaml")
    config=OmegaConf.load("pettingllms/config/code/code_eval.yaml")
    _ = test_server_manager_simple(trainer_config,config)

if __name__ == "__main__":
    test_rollout_engine_simple()