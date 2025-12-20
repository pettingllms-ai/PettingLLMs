import asyncio
import logging
import time
import json
import traceback
import uuid
from tqdm.asyncio import tqdm
import random
try:
    from verl.protocol import DataProto
except Exception:  # fallback when verl is a src tree: verl/verl/protocol.py
    from verl import DataProto
import torch
import numpy as np
from pettingllms.trainer.multiagentssys_register import     ENV_CLASS_MAPPING,ENV_BATCH_CLASS_MAPPING
# Backward compatibility
from pettingllms.multi_agent_env.autoevol.gen_agent import MASGenerator
from functools import partial
import multiprocessing
from pettingllms.utils.performance import create_timer
import copy
from pettingllms.trainer.async_generate import convert_prompt_to_dpr, llm_async_generate
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.multi_agent_env.math.math_worker import get_ray_docker_worker_cls


logger = logging.getLogger(__name__)

_DEBUG_ENGINE = False


def set_debug_engine(enabled: bool):
    """Enable or disable debug output for execution engine"""
    global _DEBUG_ENGINE
    _DEBUG_ENGINE = enabled


class MultiAgentsExecutionEngineAutoEvol:
    def _load_config_parameters(self):
        self.max_prompt_length = getattr(self.config.training, 'max_prompt_length', 1024)
        self.max_response_length = getattr(self.config.training, 'max_response_length', 1024)
        self.turn_order = self.config.multi_agent_interaction.turn_order
        self.num_interacting_agents = len(self.turn_order)  # Computed from turn_order
        self.parallel = getattr(self.config.multi_agent_interaction, 'parallel', False)
        self.generate_timeout = getattr(self.config.training, 'generate_timeout', 300.0)
        # Multi-modal support configuration
        self.enable_multimodal = getattr(self.config.training, 'enable_multimodal', False)
          
        
    def __init__(
        self,
        config,
        ppo_trainer_config_dict=None,
        tokenizer_dict=None,
        processor_dict=None,
        server_address_dict=None,
        agent_policy_mapping=None,
        env_args=None,
        max_workers=1000,
        lora_differ_mode=False,
        agent_lora_mapping=None,
        use_lora_for_generation=False,
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
        self.lora_differ_mode = lora_differ_mode
        self.agent_lora_mapping = agent_lora_mapping or {}
        # Control whether to use LoRA adapters for generation
        self.use_lora_for_generation = use_lora_for_generation
        # Read parameters from config with fallback to defaults
        self.timer.checkpoint("Loading config parameters")
        self._load_config_parameters()
        self.n_cpu = multiprocessing.cpu_count()

        env_name = getattr(self.config.env, 'name', None)
        if env_name is None:
            raise ValueError("env.name is not set in the config.env")

            
        print(f"env_name: {env_name}")
        self.experiment_name = self.config.training.experiment_name
        self.env_name = env_name
        self.env_class = ENV_CLASS_MAPPING[env_name]
        self.agent_class_list = [MASGenerator(task_type=getattr(self.config, 'task_type', "math"))]
        self.agent_configs_raw = self.config.agent_policy_configs.agent_configs
        self.agent_config_dict = {}
        for agent_key, agent_config in self.agent_configs_raw.items():
            agent_name = agent_config.name
            self.agent_config_dict[agent_name] = agent_config
        self.step_timeout = getattr(self.config.training, 'step_timeout', 150.0)
        print(f"agent_config_dict keys: {list(self.agent_config_dict.keys())}")
        self.server_address_dict = server_address_dict or {}
        self.chat_parser_dict={}
        self.rollout_latency_dict = {}
        self.timer.checkpoint("MultiAgentsExecutionEngine initialization completed")
        
        num_workers = self.config.training.get("num_workers", 180)
        RayDockerWorker = get_ray_docker_worker_cls(num_workers=num_workers)
        print("begin to create Ray docker workers")
        if RayDockerWorker is not None and hasattr(RayDockerWorker, "remote"):
            num_workers = self.config.training.get("num_workers", 32)
            self.num_workers = num_workers

            # Get GPU group ID for worker pool isolation
            import os
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                gpu_ids = sorted([g.strip() for g in cuda_visible.split(",") if g.strip()])
                self.gpu_group_id = f"gpu_{'_'.join(gpu_ids)}"
            else:
                self.gpu_group_id = "gpu_default"

            print(f"GPU group ID: {self.gpu_group_id}")
            self.timer.checkpoint(f"Creating {num_workers} Ray docker workers for GPU group {self.gpu_group_id}")
            self.env_workers = [RayDockerWorker.remote(idx) for idx in range(num_workers)]
        else:
            self.gpu_group_id = "gpu_default"
            print(f"RayDockerWorker is not available or invalid for env '{self.env_name}'. Skipping env workers initialization.")
        

    def init_agents_and_envs(self,mode="train",step_idx=0):
        self.multi_logger = get_multi_logger(experiment_name=self.experiment_name)
        self.timer.checkpoint("Starting init_agents_and_envs")
        self.mode=mode
        self.success_rollout_idx_list_dict={}
        self.success_ave_turn_dict={}
        
        # Initialize enable_thinking mapping for each agent
        self.agent_enable_thinking = {}
        # Initialize enable_multimodal mapping for each agent
        self.agent_enable_multimodal = {}
        for agent_name in self.turn_order:
            agent_config = self.agent_config_dict.get(agent_name, None)
            # Read enable_thinking from agent config, default to False
            enable_thinking = False
            if agent_config:
                # Read from train_llm_config (enable_thinking is same for train and val)
                train_llm_config = getattr(agent_config, 'train_llm_config', None)
                if train_llm_config:
                    enable_thinking = train_llm_config.get('enable_thinking', False)
                else:
                    # Fallback to old format
                    enable_thinking = getattr(agent_config, 'enable_thinking', False)
                    self.agent_enable_thinking[agent_name] = enable_thinking
                    # Read enable_multimodal from agent config, fallback to global setting
                    enable_multimodal = getattr(agent_config, 'enable_multimodal', self.enable_multimodal) if agent_config else self.enable_multimodal
                    self.agent_enable_multimodal[agent_name] = enable_multimodal
                 
        
        if mode=="validate":
            self.sample_num=self.config.training.validate_sample_num
            self.gen_batch_size=1
            for agent_name in self.turn_order:
                self.success_rollout_idx_list_dict[agent_name]=[]
                self.success_ave_turn_dict[agent_name]=0
        else:
            self.sample_num=self.config.training.train_sample_num
            self.gen_batch_size=self.config.training.train_batch_size

        self.env_batch_class=ENV_BATCH_CLASS_MAPPING[self.env_name]
        env_indices=range(step_idx*self.gen_batch_size, (step_idx+1)*self.gen_batch_size)
        # Convert to list for safety
        env_indices_list = list(env_indices)
        self.envs_batch=self.env_batch_class(
            env_idx_list=range(self.gen_batch_size),
            rollout_idx_list=range(self.gen_batch_size*self.sample_num),
            env_indices=env_indices_list,
            samples=self.sample_num,
            max_turns=1,
            config=self.config,
            mode=self.mode
        )
        self.envs=self.envs_batch.env_list
        print(f"[DEBUG] envs_batch created {len(self.envs)} environments")
        print(f"[DEBUG] sample_num={self.sample_num}, gen_batch_size before={self.gen_batch_size}")
        self.gen_batch_size=len(self.envs)//self.sample_num
        print(f"[DEBUG] gen_batch_size after={self.gen_batch_size}")
        self.env_idx_list=range(len(self.envs)//self.sample_num)
        self.rollout_idx_list=range(len(self.envs))
        print(f"[DEBUG] rollout_idx_list has {len(self.rollout_idx_list)} elements")
        self.env_rollout_mapping={}
        for env_idx in range(len(self.env_idx_list)):
            self.env_rollout_mapping[env_idx] = [_ for _ in range(env_idx*self.sample_num, (env_idx+1)*self.sample_num)]
        self.timer.checkpoint("Starting batched env initialization")
            
        # For autoevol, each rollout only needs one MASGenerator
        # No need for multiple agents or turns - just one generation per rollout
        self.agent_groups_list = []
        for rollout_idx in range(len(self.envs)):
            # Create a single MASGenerator for this rollout
            agent_init_params = {
                'env_idx': rollout_idx,
                'agent_sample_idx': rollout_idx,
                'rollout_idx': rollout_idx,
                'task_type': getattr(self.config.env, 'task_type', 'math')
            }
            agent_init_params['benchmark'] = getattr(self.config.env, 'benchmark', 'AIME24') if hasattr(self.config, 'env') else 'AIME24'

            # Single MASGenerator instance per rollout
            mas_generator = MASGenerator(**agent_init_params)
            self.agent_groups_list.append(mas_generator)
        
    
                   
            
    async def generate_single_rollout(self, rollout_idx):
        """
        Generate a single rollout for autoevol - simplified to single generation per rollout.
        MASGenerator only needs to generate once, then step() handles MAS execution.

        Args:
            rollout_idx: Index of the rollout

        Returns:
            DataProto: DataProto object containing trajectory data
        """
        trajectory_per_task_dict = {}
        env_idx = rollout_idx // self.sample_num
        start_time = time.perf_counter()
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()

        reward = 0.0
        env = self.envs[rollout_idx]
        mas_generator = self.agent_groups_list[rollout_idx]  # Single MASGenerator instance

        # Use the first (and only) agent name from turn_order
        agent_name = self.turn_order[0] if self.turn_order else "mas_generator"
        policy_name = self.agent_policy_mapping.get(agent_name)

        self.multi_logger.log_async_event(
            self.mode, env_idx, rollout_idx, "generation_start",
            f"Starting MAS generation for rollout {rollout_idx}",
            {"rollout_idx": rollout_idx}
        )

        # Step 1: Update agent from environment to get prompt
        mas_generator.update_from_env(env)
        prompt = mas_generator.current_prompt

        agent_enable_thinking = self.agent_enable_thinking.get(agent_name, False)
        agent_enable_multimodal = self.agent_enable_multimodal.get(agent_name, False)

        # Step 2: Format prompt for model
        format_prompt = convert_prompt_to_dpr(
            self.tokenizer_dict[policy_name],
            self.processor_dict.get(policy_name),
            prompt,
            self.max_prompt_length,
            multi_modal=agent_enable_multimodal,
            enable_thinking=agent_enable_thinking
        )

        if format_prompt is None:
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, agent_name,
                "Failed to format prompt",
                {"error": "format_prompt is None"}
            )
            return trajectory_per_task_dict

        # Step 3: Generate MAS code using LLM
        ppo_trainer_config = self.ppo_trainer_config_dict.get(policy_name, None)
        model_path = ppo_trainer_config.actor_rollout_ref.model.path
        if "checkpoint" in str(model_path):
            model_name = str(model_path)
        else:
            model_name = "/".join(str(model_path).split("/")[-2:])

        output_dpr = None
        response = None

        try:
            _addresses = self.server_address_dict.get(policy_name)
            if isinstance(_addresses, (list, tuple)):
                _address = random.choice(_addresses) if len(_addresses) > 0 else _addresses[0]
            else:
                _address = _addresses

            lora_id = None
            if self.lora_differ_mode and self.use_lora_for_generation and agent_name in self.agent_lora_mapping:
                lora_id = self.agent_lora_mapping[agent_name]

            agent_config = self.agent_config_dict.get(agent_name, None)
            agent_sample_num = getattr(agent_config, 'sample_num', 1) if agent_config else 1

            if _DEBUG_ENGINE:
                lora_status = f"LoRA={lora_id}" if lora_id is not None else "base_model"
                print(f"[Engine][AUTOEVOL] rollout_idx={rollout_idx} generating MAS code ({lora_status}, multimodal={agent_enable_multimodal}, sample_num={agent_sample_num})")

            output_dpr, response = await llm_async_generate(
                rollout_idx=rollout_idx,
                turn_idx=0,  # Single turn
                agent_idx=0,  # Single agent
                prompt_dpr=format_prompt,
                ppo_trainer_config=ppo_trainer_config,
                address=_address,
                model_name=model_name,
                tokenizer=self.tokenizer_dict[policy_name],
                enable_thinking=agent_enable_thinking,
                application_id=str(uuid.uuid4()),
                env_idx=env_idx,
                policy_name=policy_name,
                timeout=self.generate_timeout,
                mode=self.mode,
                lora_id=lora_id,
                agent_config=agent_config,
                sample_num=agent_sample_num,
            )
        except Exception as e:
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, agent_name,
                f"Failed to generate response: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            output_dpr = None
            response = ""

        if response is None:
            response = ""

        # Step 4: Update agent with model response (extract code)
        mas_generator.update_from_model(response)
        
        # Log the generated MAS code

        # Step 5: Execute MAS code via step() method and get tokenized trajectories + final reward
        tokenized_trajectories = []
        final_reward = 0.0
        try:
            env_worker_id = rollout_idx % self.num_workers
            env_worker = self.env_workers[env_worker_id]

            if hasattr(env, 'state'):
                env.state.assigned_worker_id = env_worker_id
                env.state.gpu_group_id = self.gpu_group_id

            # Prepare output directory for MAS execution
            import os
            output_base_dir = './tmp_auto_mas'
            output_dir = os.path.abspath(os.path.join(
                output_base_dir,
                f'rollout_{rollout_idx}'
            ))
            try:
                os.makedirs(output_dir, exist_ok=True)
                if not os.path.exists(output_dir):
                    raise RuntimeError(f"Failed to create output directory: {output_dir}")
            except Exception as e:
                self.multi_logger.log_env_agent_info(
                    self.mode, env_idx, rollout_idx, 1, agent_name,
                    f"Failed to create output directory: {e}",
                    {"output_dir": output_dir, "error": str(e)}
                )
                raise
            
            # Get LLM config parameters from train/val config
            if agent_config:
                if self.mode == "train":
                    llm_config = getattr(agent_config, 'train_llm_config', {})
                else:
                    llm_config = getattr(agent_config, 'val_llm_config', {})
                
                temperature = llm_config.get('temperature', 0.2) if llm_config else 0.2
                top_p = llm_config.get('top_p', 0.95) if llm_config else 0.95
                top_k = llm_config.get('top_k', 20) if llm_config else 20
            else:
                temperature = 0.2
                top_p = 0.95
                top_k = 20

            # Prepare LLM config for MAS execution
            llm_config_for_mas = {
                "server_address": _address,
                "model_name": model_name,
                "api_key": getattr(self.config.training, 'openai_api_key', ''),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
            
            # Log MAS execution configuration
            print(f"[AUTOEVOL] Rollout {rollout_idx}: output_dir={output_dir}")
            print(f"[AUTOEVOL] Rollout {rollout_idx}: LLM config: temp={temperature}, top_p={top_p}, top_k={top_k}")
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, agent_name,
                "MAS execution configuration",
                {
                    "output_dir": output_dir,
                    "llm_config": {k: v for k, v in llm_config_for_mas.items() if k != 'api_key'},
                    "mode": self.mode
                }
            )

            # Call step and get tokenized trajectories, final reward, and MAS execution success
            tokenized_trajectories, final_reward, mas_execution_success = await asyncio.wait_for(
                mas_generator.step(
                    env_data=env,
                    env_worker=env_worker,
                    output_dir=output_dir,
                    server_address=_address,
                    model_name=model_name,
                    tokenizer=self.tokenizer_dict[policy_name],
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length,
                    llm_config_for_mas=llm_config_for_mas
                ),
                timeout=self.step_timeout
            )
            
            # Log step execution results
            print(f"[AUTOEVOL] Rollout {rollout_idx}: MAS execution success={mas_execution_success}, final_reward={final_reward}, trajectories={len(tokenized_trajectories) if tokenized_trajectories else 0}")
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, agent_name,
                "MAS step completed",
                {
                    "mas_execution_success": mas_execution_success,
                    "final_reward": final_reward,
                    "num_trajectories": len(tokenized_trajectories) if tokenized_trajectories else 0,
                    "output_dir": output_dir
                }
            )
        except asyncio.TimeoutError:
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, agent_name,
                f"MAS step timed out after {self.step_timeout}s",
                {"error": "timeout", "timeout_seconds": self.step_timeout}
            )
            tokenized_trajectories = []
            mas_execution_success = False
        except Exception as e:
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, agent_name,
                f"MAS step failed: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            tokenized_trajectories = []
            mas_execution_success = False
            

        # Step 7: Merge MAS generation DataProto with tokenized trajectories DataProtos
        all_dataprotos = []

        # First batch: MAS generation DataProto (reward based on task correctness)
        if output_dpr is not None:
            batch_size = output_dpr.batch.shape[0]
            output_dpr.non_tensor_batch["reward"] = np.array([final_reward] * batch_size)
            output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name] * batch_size, dtype=object)
            output_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward] * batch_size)
            print(f"[DEBUG REWARD] Rollout {rollout_idx}: Set reward array with final_reward={final_reward}, batch_size={batch_size}")

            if self.lora_differ_mode:
                lora_ids = [self.agent_lora_mapping[agent_name]] * batch_size
                output_dpr.non_tensor_batch["lora_ids"] = np.array(lora_ids, dtype=object)

            all_dataprotos.append(output_dpr)

        # Second batch: tokenized trajectory DataProtos from MAS execution (reward = final_reward)
        if tokenized_trajectories:
            for traj_dpr, traj_response in tokenized_trajectories:
                # Add metadata to each trajectory DataProto
                batch_size = traj_dpr.batch.shape[0]
                traj_dpr.non_tensor_batch["reward"] = np.array([final_reward] * batch_size)
                traj_dpr.non_tensor_batch["agent_name"] = np.array([agent_name] * batch_size, dtype=object)
                traj_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward] * batch_size)
                traj_dpr.non_tensor_batch["turn_idx"] = np.array([1] * batch_size)
                traj_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * batch_size)
                traj_dpr.non_tensor_batch["rollout_idx"] = np.array([rollout_idx] * batch_size)
                traj_dpr.non_tensor_batch["agent_idx"] = np.array([0] * batch_size)

                if self.lora_differ_mode:
                    lora_ids = [self.agent_lora_mapping[agent_name]] * batch_size
                    traj_dpr.non_tensor_batch["lora_ids"] = np.array(lora_ids, dtype=object)

                all_dataprotos.append(traj_dpr)

        # Concatenate all DataProtos if we have any
        if all_dataprotos:
            if len(all_dataprotos) == 1:
                trajectory_per_task_dict[policy_name] = all_dataprotos[0]
            else:
                trajectory_per_task_dict[policy_name] = DataProto.concat(all_dataprotos)

        # Step 8: Log results
        env_state_compact = env.state.to_dict_compact(agent_name=agent_name) if hasattr(env.state, 'to_dict_compact') else env.state

        self.multi_logger.log_env_agent_info(
            self.mode, env_idx, rollout_idx, 1, agent_name,
            "MAS generation and execution completed",
            {
                "agent_prompt": {"text": prompt.get("text", "") if isinstance(prompt, dict) else str(prompt), "image": None},
                "agent_response": response,
                "env_state": env_state_compact,
                "reward": float(reward)
            }
        )

        # Step 9: Log rollout summary
        agent_rewards = {agent_name: mas_generator.agent_reward}
        self.multi_logger.log_rollout_summary(
            self.mode, env_idx, rollout_idx, agent_rewards,
            "rollout_complete",
            extra_data={
                "turn_idx": 1,  # Single turn
                "message": f"Rollout {rollout_idx} completed",
                "reward": float(reward)
            }
        )

        if self.mode == "validate":
            if env.success:
                self.success_rollout_idx_list_dict[agent_name].append(rollout_idx)
                self.success_ave_turn_dict[agent_name] += 1  # Single turn
        
       
        #trajectory_per_task_dict = self._assign_consistent_uids(trajectory_per_task_dict)
        
        # record latency for this rollout
        try:
            
            latency_s = time.perf_counter() - start_time
            self.rollout_latency_dict[rollout_idx] = {"latency_s": latency_s, "reward": reward}
            self.multi_logger.log_async_event(
                self.mode, env_idx, rollout_idx, "rollout_latency",
                f"Rollout {rollout_idx} latency: {latency_s:.3f}s",
                {"latency_s": float(latency_s)}
            )
        except Exception:
            pass

        return trajectory_per_task_dict
            


    async def generate_multiple_rollouts_concurrent(self, env_idx_list, rollout_mode="tree"):
        rollout_indices = []
        for env_idx in env_idx_list:
            rollout_indices.extend(self.env_rollout_mapping[env_idx])
        
        concurrent_timer = create_timer("ConcurrentRollouts")
        concurrent_timer.start(f"Starting concurrent rollouts for {len(rollout_indices)} rollouts")
        
        concurrent_timer.checkpoint("Creating async tasks")

        tasks = [
                asyncio.create_task(
                    self.generate_single_rollout(rollout_idx=rollout_idx)
                )
                for rollout_idx in rollout_indices
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
                        #print(policy_data)
                        if policy_data.batch is not None:  
                            if aggregated_results[policy_name].batch is None:
                                aggregated_results[policy_name] = policy_data
                            else:
                                aggregated_results[policy_name] = DataProto.concat([
                                    aggregated_results[policy_name], 
                                    policy_data
                                ])
                            #print(f"The length of concatenated aggregated_results[policy_name]: {len(aggregated_results[policy_name])}")
                    
                    completed_count += 1
                    
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)})")
                except Exception as e:
                    failed_count += 1
                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)}, {failed_count} failed)")
                    
                    self.multi_logger.log_async_event(
                        self.mode, -1, -1, "task_error",
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
            self.multi_logger.log_ray_status(mode=self.mode, context="during_error")
            
            self.multi_logger.log_async_event(
                self.mode, -1, -1, "concurrent_batch_error",
                f"Concurrent execution encountered error: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            for task in tasks:
                if not task.done():
                    task_name = task.get_name()
                    self.multi_logger.log_async_event(
                        self.mode, -1, -1, "task_cancel",
                        f"Cancelling task {task_name}"
                    )
                    task.cancel()
            raise

        task_pbar.close()
        
        concurrent_timer.checkpoint("All tasks completed")
        if self.mode=="validate":
            for agent_name in self.turn_order:
                success_rate = len(self.success_rollout_idx_list_dict.get(agent_name, [])) / len(tasks)
                self.multi_logger.log_rollout_summary(
                    self.mode, -1, -1, 
                    {agent_name: success_rate}, 
                    "validate_finished",
                    extra_data={"success_rate": success_rate}
                )
            
        self.multi_logger.log_async_event(
            self.mode, -1, -1, "concurrent_batch_complete",
            "Concurrent execution completed",
            {
                "successfully_processed": completed_count,
                "total_env_groups": len(tasks),
                "failed": failed_count,
                "success_rate": f"{completed_count}/{len(tasks)}",
                "aggregated_policies": list(aggregated_results.keys()),
            }
        )
        
        # Log Ray status after concurrent execution
        self.multi_logger.log_ray_status(mode=self.mode, context="after_concurrent_batch")
        
        import sys
        sys.stdout.flush()
        concurrent_timer.end("Concurrent rollouts completed successfully")

        sys.stdout.flush()
        return aggregated_results



