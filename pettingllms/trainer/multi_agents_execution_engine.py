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
from pettingllms.utils.simpler_timer import create_timer, timer_checkpoint
import copy
from pettingllms.multi_agent_env.base.env import Env, EnvBatch
from pettingllms.misc import colorful_print
from pettingllms.parser.chat_template.parser import ChatTemplateParser
from pettingllms.trainer.utils import convert_prompt_to_dpr, convert_dpr_to_response, convert_prompt_to_format,llm_async_generate
from pettingllms.utils.logger_config import get_multi_logger
from threading import Thread
from pettingllms.trainer.utils import build_reverse_mapping,poll_completions_openai
from pettingllms.multi_agent_env.code.code_utils import get_ray_docker_worker_cls


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
            self.step_timeout = 60     # 30 seconds for environment step
        
    def __init__(
        self,
        config,
        ppo_trainer_config_dict=None,
        tokenizer_dict=None,
        processor_dict=None,
        server_address_dict=None,
        agent_policy_mapping=None,
        env_args=None,
        max_workers=1000
    ):
        
        # Initialize timer for this engine
        self.timer = create_timer("MultiAgentsExecutionEngine")
        self.timer.start("Initializing MultiAgentsExecutionEngine")

        self.config = config
        self.ppo_trainer_config_dict = ppo_trainer_config_dict or {}
        self.tokenizer_dict = tokenizer_dict
        self.processor_dict = processor_dict or {}
        self.agent_policy_mapping = agent_policy_mapping or {}
        self.env_args = env_args or {}
        self.max_workers = max_workers
 
        self.multi_logger = get_multi_logger()
        
        # Read parameters from config with fallback to defaults
        self.timer.checkpoint("Loading config parameters")
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
        self.agent_configs_raw = self.config.agent_policy_configs.agent_configs
        
        # Create a mapping from agent name to agent config
        self.agent_config_dict = {}
        for agent_key, agent_config in self.agent_configs_raw.items():
            agent_name = agent_config.name
            self.agent_config_dict[agent_name] = agent_config
        self.sample_mode=self.config.sample_mode
        # Calculate total sample_num and sample_num_list based on turn_order
        self.sample_num=self.gen_n_samples
        self.filter_ratio=self.config.data.filter_ratio
        self.filter_method=getattr(self.config.data, 'filter_method', 'uid')
        print(f"sample_num: {self.sample_num}")
        print(f"agent_config_dict keys: {list(self.agent_config_dict.keys())}")
        # rollout_engine_dict is not maintained in this class to avoid referencing a non-existent attribute
        self.server_address_dict = server_address_dict or {}
        self.chat_parser_dict={}
        self.rollout_latency_dict = {}
        self.timer.checkpoint("MultiAgentsExecutionEngine initialization completed")
        

    def init_agents_and_envs(self,mode="train",step_idx=0,resample=True):
        self.timer.checkpoint("Starting init_agents_and_envs")
        self.mode=mode
        if mode=="validate":
            self.success_rollout_idx_list_dict={}
            self.sample_num=1
            self.gen_batch_size=1
            for agent_name in self.turn_order:
                self.success_rollout_idx_list_dict[agent_name]=[]
        else:
            self.sample_num=self.gen_n_samples
            self.gen_batch_size=self.config.data.gen_batch_size
        
        # Check for batched_init in config.env
        if hasattr(self.config, 'env') and self.config.env is not None:
            batched_init = getattr(self.config.env, 'batched_init', True)
        else:
            batched_init = getattr(self.config, 'batched_init', True)
        if batched_init == False:
            self.timer.checkpoint("Starting non-batched env initialization")
            with multiprocessing.Pool(self.n_cpu // 4) as pool: # Only use 1/4 of the cores to avoid conflicts
                func=partial(self.__init_one_env_instance, env_args=self.env_args)
                self.envs = pool.map(func, range(self.gen_batch_size*self.sample_num))
            self.timer.checkpoint("Non-batched env initialization completed")   
        else:
            self.env_batch_class=ENV_BATCH_CLASSES[self.env_name]
            self.timer.checkpoint("Creating environment batch")  
            if resample==False and mode=="train":
                for env in self.envs:
                    env.reset()
            else:
                epoch_size=getattr(self.config.data, "epoch_size", 20)
                step_in_epoch_idx=step_idx%epoch_size
                env_indices=range(step_in_epoch_idx*self.gen_batch_size, (step_in_epoch_idx+1)*self.gen_batch_size)
                self.envs_batch=self.env_batch_class(
                    env_idx_list=range(self.gen_batch_size),
                    rollout_idx_list=range(self.gen_batch_size*self.sample_num),
                    env_indices=env_indices,
                    samples=self.sample_num,
                    max_turns=self.max_turns,
                    config=self.config,
                    mode=self.mode
                )
                self.envs=self.envs_batch.env_list
                self.gen_batch_size=len(self.envs)//self.sample_num
                self.env_idx_list=range(len(self.envs)//self.sample_num)
                self.rollout_idx_list=range(len(self.envs))
                self.env_rollout_mapping={}
                for env_idx in range(len(self.env_idx_list)):
                    self.env_rollout_mapping[env_idx] = [_ for _ in range(env_idx*self.sample_num, (env_idx+1)*self.sample_num)]
            self.timer.checkpoint("Starting batched env initialization")
            self.env_workers = None
            RayDockerWorker = get_ray_docker_worker_cls()
            print("begin to create Ray docker workers")
            if RayDockerWorker is not None:
                num_workers = len(self.envs)
                self.timer.checkpoint(f"Creating {num_workers} Ray docker workers")
                self.env_workers = [RayDockerWorker.remote(_) for _ in range(num_workers)]
                self.multi_logger.log_rollout_summary(
                    -1, -1, "env_workers",
                    f"init {num_workers} env workers"
                )
            self.agent_groups_list = []
            for rollout_idx in range(len(self.envs)):
                self.agent_groups_list_per_rollout = []
                for agent_idx, agent_name in enumerate(self.turn_order):
                    agent_class = self.agent_class_list[agent_idx]
                    self.agent_groups_list_per_rollout.append(agent_class(env_idx=rollout_idx, agent_sample_idx=rollout_idx))
                self.agent_groups_list.append(self.agent_groups_list_per_rollout)
            
            
            self.timer.checkpoint("Environment batch created") 
        self.timer.checkpoint("Agent groups initialization completed")
    def __init_one_env_instance(self, rollout_idx, env_args):
        env = self.env_class( env_idx=rollout_idx % self.gen_batch_size,rollout_idx=rollout_idx,max_turns=self.max_turns, **env_args)
        
        return env
                   
            
    async def generate_single_rollout(self, rollout_idx):
        """
        Generate a single rollout, adapted for multi-agent interaction in the code testing environment.
        
        Args:
            env: Code testing environment instance
            timing_raw: Timing record dictionary
            meta_info: Meta information
            
        Returns:
            DataProto: DataProto object containing trajectory data
        """
        trajectory_per_task_dict = {}
        env_idx = rollout_idx// self.sample_num
        start_time = time.perf_counter()
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()

        reward=0.0
        env = self.envs[rollout_idx]
        agent_group = self.agent_groups_list[rollout_idx]
        
        for turn_idx in range(self.max_turns):
            self.multi_logger.log_async_event(
                env_idx, rollout_idx, "turn_start",
                f"Starting turn {turn_idx + 1}",
                {"turn_idx": turn_idx + 1}
            )
            
            for agent_idx, agent_name in enumerate(self.turn_order):
                current_agent = agent_group[agent_idx]
                current_agent.update_from_env(env)
                prompt = current_agent.current_prompt
                policy_name = self.agent_policy_mapping.get(agent_name) if self.agent_policy_mapping else None
                # Convert to DataProto format
                format_prompt =convert_prompt_to_dpr( self.tokenizer_dict[policy_name], 
                        self.processor_dict.get(policy_name), 
                        prompt, 
                        self.max_prompt_length,
                        multi_modal=False
                   )
                if format_prompt is None:
                    return None
                ppo_trainer_config = self.ppo_trainer_config_dict.get(policy_name, None)
                model_path=ppo_trainer_config.actor_rollout_ref.model.path
                model_name = "/".join(model_path.split("/")[-2:])
                # Generate responses
                output_dpr = None
                response_str = None
                print(f"DEBUG: begin tp generate response for {agent_name} with model {model_name} using llm_async_generate")
                
                try:
                    output_dpr,response_str = await llm_async_generate(
                        rollout_idx=rollout_idx, 
                        turn_idx=turn_idx, 
                        agent_idx=agent_idx,
                        prompt_dpr=format_prompt, 
                        ppo_trainer_config=ppo_trainer_config,
                        address=self.server_address_dict[policy_name],
                        model_name=model_name,
                        tokenizer=self.tokenizer_dict[policy_name],
                        image_data=None,
                        application_id=str(uuid.uuid4()),
                        env_idx=rollout_idx // self.sample_num,
                        policy_name=policy_name,
                        timeout=self.generate_timeout,
                        mode=self.mode
                    )
                except Exception as e:
                    self.multi_logger.log_env_agent_info(
                        env_idx, rollout_idx, turn_idx + 1, agent_name,
                        f"Failed to generate response: {e}",
                        {"error": str(e), "traceback": traceback.format_exc()}
                    )
               
                    output_dpr = None
                    response_str = None
                    
                current_agent.update_from_model(response_str)
                
              
                try:
                    await asyncio.wait_for(
                        current_agent.step(self.envs[rollout_idx],env_worker=self.env_workers[rollout_idx]),
                        timeout=self.step_timeout
                    )
                except asyncio.TimeoutError:
                    self.multi_logger.log_env_agent_info(
                        env_idx, rollout_idx, turn_idx + 1, agent_name,
                        f"❌ Environment step timed out after {self.step_timeout}s",
                        {"error": "timeout", "timeout_seconds": self.step_timeout}
                    )
                    if len(current_agent.reward_history)>0:
                        current_agent.agent_reward = 0.0-current_agent.reward_history[-1]
                    else:
                        current_agent.agent_reward = 0.0
                    current_agent.reward_history.append(0.0)
                if output_dpr is not None:
                    output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
                    output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name], dtype=object)  # Add agent name for metrics tracking
               
                    if trajectory_per_task_dict[policy_name].batch is None:
                        # If empty, assign directly
                        trajectory_per_task_dict[policy_name] = output_dpr
                    else:
                        # Use concat instead of union, because each response content is different
                        trajectory_per_task_dict[policy_name] = DataProto.concat([
                            trajectory_per_task_dict[policy_name], 
                            output_dpr
                        ])
                        print(f"The length of concatenated trajectory_per_task_dict[policy_name]: {len(trajectory_per_task_dict[policy_name])}")
                
                if agent_name == self.turn_order[-1]:
                    self.multi_logger.log_env_agent_info(
                        env_idx, rollout_idx, turn_idx + 1, agent_name,
                        "Trajectory information updated",
                        {
                            "env_state": env.state,
                       
                        }
                    )
                    reward=env.state.ground_truth_test_vs_generated_code_match_ratio
        
        
            finish=False
            if agent_group[-1].done:
                finish=True
            if finish:
                agent_rewards={agent_name: agent.reward_history for agent_name, agent in zip(self.turn_order, agent_group)}
                
                self.multi_logger.log_rollout_summary(
                    env_idx, rollout_idx, "success!!",
                    f"Rollout {rollout_idx} completed successfully",
                    extra_data={
                        "turn_idx": turn_idx,
                        "agent_rewards": agent_rewards,
                        "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                        "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                    }
                )
                break
        agent_rewards={agent_name: agent.reward_history for agent_name, agent in zip(self.turn_order, agent_group)}
        self.multi_logger.log_rollout_summary(
                env_idx, rollout_idx, "rollout_complete",
                f"Rollout {rollout_idx} completed successfully",
                extra_data={
                    "turn_idx": 4,
                    "agent_rewards": agent_rewards,
                    "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                    "env_state.ground_truth_test_vs_generated_code_match_ratio": env.state.ground_truth_test_vs_generated_code_match_ratio,
                }
            )

        if self.mode=="validate":
            for i,agent_name in enumerate(self.turn_order):
                current_agent=self.agent_groups_list[rollout_idx][i]
                if current_agent.is_pass:
                    self.success_rollout_idx_list_dict[agent_name].append(rollout_idx)
        
       
        #trajectory_per_task_dict = self._assign_consistent_uids(trajectory_per_task_dict)
        
        # record latency for this rollout
        try:
            
            latency_s = time.perf_counter() - start_time
            self.rollout_latency_dict[rollout_idx] = {"latency_s": latency_s, "reward": reward}
            self.multi_logger.log_async_event(
                env_idx, rollout_idx, "rollout_latency",
                f"Rollout {rollout_idx} latency: {latency_s:.3f}s",
                {"latency_s": float(latency_s)}
            )
        except Exception:
            pass

        return trajectory_per_task_dict

        
    async def generate_env_idx_rollout(self, env_idx):
        """
        Generate a single rollout, adapted for multi-agent interaction in the code testing environment.
        
        Args:
            env: Code testing environment instance
            timing_raw: Timing record dictionary
            meta_info: Meta information
            
        Returns:
            DataProto: DataProto object containing trajectory data
        """
        trajectory_per_task_dict = {}
        rollout_idx_list=self.env_rollout_mapping[env_idx]
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()

        envs_list = [self.envs[rollout_idx] for rollout_idx in rollout_idx_list]
        agent_groups = [self.agent_groups_list[rollout_idx] for rollout_idx in rollout_idx_list]
        finish=False
        for turn_idx in range(self.max_turns):
            for agent_idx, agent_name in enumerate(self.turn_order):
                async def single_generate_response(idx):
                    rollout_idx=rollout_idx_list[idx]
                    env=envs_list[idx]
                    current_agent = agent_groups[idx][agent_idx]
                    current_agent.update_from_env(env)
                    prompt = current_agent.current_prompt
                    policy_name = self.agent_policy_mapping.get(agent_name) 
                    format_prompt =convert_prompt_to_dpr( self.tokenizer_dict[policy_name], 
                            self.processor_dict.get(policy_name), 
                            prompt, 
                            self.max_prompt_length,
                            multi_modal=False
                    )
                    if format_prompt.batch is None:
                        return None
                    ppo_trainer_config = self.ppo_trainer_config_dict.get(policy_name, None)
                    model_path=ppo_trainer_config.actor_rollout_ref.model.path
                    model_name = "/".join(model_path.split("/")[-2:])
                    output_dpr = None
                    response_str = None
                    print(f"DEBUG: begin tp generate response for {agent_name} with model {model_name} using llm_async_generate")
                    
                    try:
                        output_dpr,response_str = await llm_async_generate(
                            rollout_idx=rollout_idx, 
                            turn_idx=turn_idx, 
                            agent_idx=agent_idx,
                            prompt_dpr=format_prompt, 
                            ppo_trainer_config=ppo_trainer_config,
                            address=self.server_address_dict[policy_name],
                            model_name=model_name,
                            tokenizer=self.tokenizer_dict[policy_name],
                            image_data=None,
                            application_id=str(uuid.uuid4()),
                            env_idx=rollout_idx // self.sample_num,
                            policy_name=policy_name,
                            timeout=self.generate_timeout,
                            mode=self.mode
                        )
                    except Exception:
                        output_dpr = None
                        response_str = None
                        
                    current_agent.update_from_model(response_str)
                    try:
                        await asyncio.wait_for(
                            current_agent.step(self.envs[rollout_idx],env_worker=self.env_workers[rollout_idx]),
                            timeout=self.step_timeout
                        )
                    except asyncio.TimeoutError:
                        if len(current_agent.reward_history)>0:
                            current_agent.agent_reward = 0.0-current_agent.reward_history[-1]
                        else:
                            current_agent.agent_reward = 0.0
                        current_agent.reward_history.append(0.0)
                    if output_dpr is not None:
                        output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
                        output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name], dtype=object)  # Add agent name for metrics trackin
                    return {
                        'current_agent': current_agent,
                        'output_dpr': output_dpr,
                        'policy_name': policy_name
                    }
                
                tasks=[single_generate_response(idx) for idx in range(len(rollout_idx_list))]
                results=await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if result is not None and not isinstance(result, Exception):
                        current_agent = result['current_agent']
                        output_dpr = result['output_dpr']
                        policy_name = result['policy_name']
                        if output_dpr is not None:
                            if trajectory_per_task_dict[policy_name].batch is None:
                                trajectory_per_task_dict[policy_name] = output_dpr
                            else:
                                # Use concat instead of union, because each response content is different
                                trajectory_per_task_dict[policy_name] = DataProto.concat([
                                    trajectory_per_task_dict[policy_name], 
                                    output_dpr
                                ])
                                print(f"The length of concatenated trajectory_per_task_dict[policy_name]: {len(trajectory_per_task_dict[policy_name])}")
                values = []
                for i in range(len(rollout_idx_list)):
                    agent_i = agent_groups[i][agent_idx]
                    val = getattr(agent_i, 'value', None)
                    if agent_i.done:
                        finish=True
                    if val is None:
                        val = -1e9
                    values.append(val)
                try:
                    best_i = int(np.argmax(np.asarray(values)))
                except Exception:
                    best_i = 0
                selected_env = envs_list[best_i]
                envs_list = [selected_env for _ in envs_list]
                agent_groups = [copy.deepcopy(agent_groups[best_i]) for _ in agent_groups]
                if finish:
                    break
            if finish:
                break
        return trajectory_per_task_dict


    async def generate_multiple_rollouts_concurrent(self, env_idx_list, rollout_mode="tree"):
        rollout_indices=[]
        for env_idx in env_idx_list:
            rollout_indices.extend(self.env_rollout_mapping[env_idx])
        concurrent_timer = create_timer("ConcurrentRollouts")
        concurrent_timer.start(f"Starting concurrent rollouts for {len(rollout_indices)} rollouts")
        
        concurrent_timer.checkpoint("Creating async tasks")
        
        if rollout_mode == "tree" and self.mode=="train":
            tasks = [
                asyncio.create_task(
                    self.generate_env_idx_rollout(env_idx), 
                    name=f"env_{env_idx}_rollouts"
                )
                for env_idx in env_idx_list
            ]
        else:
            tasks = [
            asyncio.create_task(
            self.generate_single_rollout(rollout_idx), 
                name=f"env_{rollout_idx}_rollouts"
            )
            for rollout_idx in range(self.gen_batch_size*self.sample_num)
        ]
        
        
        concurrent_timer.checkpoint(f"Created {len(tasks)} async tasks")
        
        aggregated_results = {}
        for policy_name in self.tokenizer_dict.keys():
            aggregated_results[policy_name] = DataProto()
        
        completed_count = 0
        failed_count = 0
  
        task_pbar = tqdm(total=len(tasks), desc="Rollouts", position=1, leave=False)
        
        try:
            concurrent_timer.checkpoint("Starting task execution")
            for completed_task in asyncio.as_completed(tasks):
                try:
                    
                    rollout_result = await completed_task
        
                    for policy_name, policy_data in rollout_result.items():
                        print(policy_data)
                        if policy_data.batch is not None:  
                            if aggregated_results[policy_name].batch is None:
                                aggregated_results[policy_name] = policy_data
                            else:
                                aggregated_results[policy_name] = DataProto.concat([
                                    aggregated_results[policy_name], 
                                    policy_data
                                ])
                            print(f"The length of concatenated aggregated_results[policy_name]: {len(aggregated_results[policy_name])}")
                    
                    completed_count += 1
                    
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)})")
                except Exception as e:
                    failed_count += 1
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)}, {failed_count} failed)")
                    
                    self.multi_logger.log_async_event(
                        -1, -1, "task_error",
                        f"Task failed with error: {e}",
                        {
                            "failed_count": failed_count,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "traceback": traceback.format_exc()
                        }
                    )
                    
                    continue
                    
        except Exception as e:
            # Log Ray status when encountering errors
            self.multi_logger.log_ray_status(context="during_error")
            
            self.multi_logger.log_async_event(
                -1, -1, "concurrent_batch_error",
                f"Concurrent execution encountered error: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            for task in tasks:
                if not task.done():
                    task_name = task.get_name()
                    self.multi_logger.log_async_event(
                        -1, -1, "task_cancel",
                        f"Cancelling task {task_name}"
                    )
                    task.cancel()
            raise

        task_pbar.close()
        
        concurrent_timer.checkpoint("All tasks completed")
        with open("rollout_data.json", "w", encoding="utf-8") as f:
            json.dump(self.rollout_latency_dict, f, ensure_ascii=False, indent=4)
        
        
        self.multi_logger.log_async_event(
            -1, -1, "concurrent_batch_complete",
            "Concurrent execution completed",
            {
                "successfully_processed": completed_count,
                "total_env_groups": len(tasks),
                "total_rollouts": len(rollout_indices),
                "failed": failed_count,
                "success_rate": f"{completed_count}/{len(tasks)}",
                "aggregated_policies": list(aggregated_results.keys()),
            }
        )
        
        # Log Ray status after concurrent execution
        self.multi_logger.log_ray_status(context="after_concurrent_batch")
        
        concurrent_timer.checkpoint("Starting UID assignment and filtering")
        # Assign uid and filter the rollouts
        if self.mode == "train":    
            aggregated_results = self._assign_consistent_uids(aggregated_results,filter_ratio=self.filter_ratio,mode=self.filter_method)
        
        concurrent_timer.end("Concurrent rollouts completed successfully")
        return aggregated_results
    
    def _assign_consistent_uids(self, aggregated_results,filter_ratio=0.0, mode="mean"):
        import uuid
        import numpy as np
        from collections import defaultdict
        
        uid_mapping = {}
        all_rewards = []
        uid_reward_groups = defaultdict(list)  
        for policy_name, data_proto in aggregated_results.items():
            if data_proto.batch is None or len(data_proto) == 0:
                continue

            non_tensor_batch = data_proto.non_tensor_batch
            
            if not all(key in non_tensor_batch for key in ["rollout_idx", "turn_idx", "agent_idx"]):
                data_proto.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(data_proto))], dtype=object
                )
                continue
            
            rollout_indices = non_tensor_batch["rollout_idx"]
            turn_indices = non_tensor_batch["turn_idx"] 
            agent_indices = non_tensor_batch["agent_idx"]
            rewards = non_tensor_batch.get("reward", [])
            
            uids = []
            for i in range(len(data_proto)):
                key = (rollout_indices[i]//self.sample_num, turn_indices[i], agent_indices[i])
                if key not in uid_mapping:
                    uid_mapping[key] = str(uuid.uuid4())
                uid = uid_mapping[key]
                uids.append(uid)
                
                if len(rewards) > 0 and filter_ratio > 0:
                    reward_val = float(rewards[i]) if rewards[i] is not None else 0.0
                    uid_reward_groups[uid].append((i, policy_name, reward_val))
                
                if len(rewards) > 0:
                    reward_val = float(rewards[i]) if rewards[i] is not None else 0.0
                    all_rewards.append(reward_val)
            
            data_proto.non_tensor_batch["uid"] = np.array(uids, dtype=object)
        def range_normalized_variance(rewards_in_group):
            rewards_in_group = np.asarray(rewards_in_group, dtype=float)
            rng = np.max(rewards_in_group) - np.min(rewards_in_group)
            if rng == 0:   # 避免除零
                return 0.0
            return np.var(rewards_in_group, ddof=0) / (rng ** 2)
        
        sample_to_remove = set()
        if filter_ratio > 0:
            # calculate the variance of each uid group
            if mode == "dapo":
                uids_to_remove = []
                for uid, samples in uid_reward_groups.items():
                    rewards_in_group = [s[2] for s in samples]
                    variance = range_normalized_variance(rewards_in_group)
                    if variance==0:
                        uids_to_remove.append(uid)
                for uid in uids_to_remove:
                    if uid in uid_reward_groups:
                        for sample_idx, policy_name, reward_val in uid_reward_groups[uid]:
                            sample_to_remove.add((policy_name, sample_idx))

            if mode == "std":
                uid_variances = {}
                for uid, samples in uid_reward_groups.items():
                    if len(samples) > 1:
                        rewards_in_group = [s[2] for s in samples]
                        variance = range_normalized_variance(rewards_in_group)
                        uid_variances[uid] = variance
                    else:
                        uid_variances[uid] = 0.0
                
                if uid_variances:
                    total_uids = len(uid_variances)
                    num_to_remove = int(total_uids * filter_ratio)
                    
                    if num_to_remove > 0:
                        sorted_uids = sorted(uid_variances.items(), key=lambda x: x[1])
                        uids_to_remove = [uid for uid, variance in sorted_uids[:num_to_remove]]
                        
                        for uid in uids_to_remove:
                            if uid in uid_reward_groups:
                                for sample_idx, policy_name, reward_val in uid_reward_groups[uid]:
                                    sample_to_remove.add((policy_name, sample_idx))
            elif mode == "mean":
                uid_means = {}
                for uid, samples in uid_reward_groups.items():
                    if len(samples) > 1:
                        rewards_in_group = [s[2] for s in samples]
                        mean = np.mean(rewards_in_group)
                        uid_means[uid] = mean
                    else:
                        uid_means[uid] = 0.0
                        
                if uid_means:
                    total_uids = len(uid_means)
                    num_to_remove = int(total_uids * filter_ratio)
                    
                    if num_to_remove > 0:
                        sorted_uids = sorted(uid_means.items(), key=lambda x: x[1])
                        uids_to_remove = [uid for uid, mean in sorted_uids[:num_to_remove]]
                        
                        for uid in uids_to_remove:
                            if uid in uid_reward_groups:
                                for sample_idx, policy_name, reward_val in uid_reward_groups[uid]:
                                    sample_to_remove.add((policy_name, sample_idx))
            elif mode=="uid":
                sample_to_remove = set()
                if filter_ratio > 0:
                    for uid, samples in uid_reward_groups.items():
                        if len(samples) > 1:
                            rewards_in_group = [s[2] for s in samples]
                            group_mean = np.mean(rewards_in_group)
                            samples_with_deviation = [(s[0], s[1], abs(s[2] - group_mean)) for s in samples]
                            samples_with_deviation.sort(key=lambda x: x[2], reverse=True)
                            num_to_remove = int(len(samples_with_deviation) * filter_ratio)
                            for i in range(num_to_remove):
                                sample_idx, policy_name, _ = samples_with_deviation[i]
                                sample_to_remove.add((policy_name, sample_idx))
        
        if sample_to_remove:
            for policy_name, data_proto in aggregated_results.items():
                if data_proto.batch is None:
                    continue
                
                keep_indices = [i for i in range(len(data_proto)) 
                               if (policy_name, i) not in sample_to_remove]
                
                if len(keep_indices) < len(data_proto):
                    if data_proto.batch is not None:
                        index_tensor = torch.as_tensor(keep_indices, dtype=torch.long)
                        data_proto.batch = data_proto.batch[index_tensor]
                    
                    if data_proto.non_tensor_batch is not None:
                        for key, value in data_proto.non_tensor_batch.items():
                            if isinstance(value, np.ndarray):
                                data_proto.non_tensor_batch[key] = value[keep_indices]
                            elif hasattr(value, '__getitem__'):
                                data_proto.non_tensor_batch[key] = np.array([value[i] for i in keep_indices], dtype=object)
        
        if all_rewards:
            summary = {
                "total_samples": len(all_rewards),
                "mean_reward": float(np.mean(all_rewards)),
                "std_reward": float(np.std(all_rewards)),
                "filtered_samples": len(sample_to_remove) if filter_ratio > 0 else 0,
                "remain_samples": len(aggregated_results)
            }
            
            self.multi_logger.log_async_event(
                -1, -1, "reward_summary",
                f"UID assignment and reward statistics complete",
                summary
            )
        
        return aggregated_results
        
       
    
class AsyncMultiAgentsExecutionEngine(MultiAgentsExecutionEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)





import hydra
from omegaconf import DictConfig

def test_rollout_engine_simple(config_path=None):
    from omegaconf import OmegaConf
    import sys
    import argparse
    
    if config_path is None:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Multi-Agent Execution Engine')
        parser.add_argument('--config', '-c', type=str, 
                          default="pettingllms/config/code/code_eval.yaml",
                          help='Path to config file')
        parser.add_argument('--trainer_config', '-t', type=str,
                          default="pettingllms/config/code/ppo_trainer/eval.yaml", 
                          help='Path to trainer config file')
        args = parser.parse_args()
        config_path = args.config
        trainer_config_path = args.trainer_config
    else:
        trainer_config_path = "pettingllms/config/code/ppo_trainer/eval.yaml"
    
    trainer_config = OmegaConf.load(trainer_config_path)
    config = OmegaConf.load(config_path)
    
@hydra.main(config_path="../config/code", config_name="code_eval", version_base=None)
def run_benchmark_with_hydra(config: DictConfig):
    """使用 Hydra 运行 benchmark，可以通过命令行覆盖任何配置参数"""
    from omegaconf import OmegaConf
    
    # 加载 trainer 配置
    trainer_config = OmegaConf.load("pettingllms/config/code/ppo_trainer/eval.yaml")
    
    print(f"运行 benchmark: {config.env.benchmark}")
    print(f"实验名称: {config.get('experiment_name', 'code_test')}")
    print("="*50)
    
   
if __name__ == "__main__":
    # 如果直接运行该文件，使用 Hydra 版本
    run_benchmark_with_hydra()