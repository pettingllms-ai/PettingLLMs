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

def colorful_print(text, color="white"):
    """Simple colorful print function for debugging"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m", 
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


class MultiAgentsPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer_dict,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor_dict=None,
    ):
        self.config = config
        self.processor_dict = processor_dict or {}
        
        
        
        # Initialize agent_policy_mapping from agent_policy_configs
        self.agent_policy_mapping = {}
        if hasattr(config, 'agent_policy_configs') and hasattr(config.agent_policy_configs, 'agent_configs'):
            for agent_key, agent_config in config.agent_policy_configs.agent_configs.items():
                agent_name = agent_config.name
                policy_name = agent_config.policy_name
                self.agent_policy_mapping[agent_name] = policy_name
                colorful_print(f"Agent mapping: {agent_name} -> {policy_name}", "green")
        
        # Initialize ppo_trainer_dict from models configuration
        self.ppo_trainer_config_dict = {}
        self.rollout_sample_dict = {}
        self.tokenizer_dict = tokenizer_dict
        self.ppo_trainer_dict = {}
        
        
        if hasattr(config, 'models'):
            for i, (model_key, model_config) in enumerate(config.models.items()):
                model_name = model_config.name
                print(f"model_config: {model_config}")
                if hasattr(model_config, 'ppo_trainer_config'):
                    ppo_config = model_config.ppo_trainer_config
                    self.ppo_trainer_config_dict[model_name] = ppo_config
                    ppo_config.data["train_batch_size"]=self.config.data.train_batch_size
                    ppo_config.data["val_batch_size"]=self.config.data.val_batch_size
                    # 确保experiment_name传递给子trainer，并为每个policy创建独立的experiment name - 使用OmegaConf.set_struct临时允许添加新键
                    if hasattr(self.config, 'experiment_name'):
                        from omegaconf import OmegaConf
                        OmegaConf.set_struct(ppo_config, False)  # 临时禁用结构化模式
                        # 为每个policy创建独立的experiment name，避免checkpoint覆盖
                        base_experiment_name = self.config.experiment_name
                        policy_experiment_name = f"{base_experiment_name}_policy_{model_name}"
                        ppo_config.experiment_name = policy_experiment_name
                        colorful_print(f"Set experiment name for model {model_name}: {policy_experiment_name}", "cyan")
                        OmegaConf.set_struct(ppo_config, True)   # 重新启用结构化模式
                    print(f"ppo_config: {ppo_config}")
                    model_tokenizer = self.tokenizer_dict[model_name]
                    #reward_fn = load_reward_manager(ppo_config,model_tokenizer, num_examine=0)
                    #val_reward_fn = load_reward_manager(ppo_config,model_tokenizer, num_examine=1)
                    
                    print(f'ppo_config (partial): {ppo_config}')

                    # Compose full PPO config by merging the base config with the per-model overrides.
                    # This explicitly expands nested defaults like `- /ppo_trainer` which are not
                    # automatically composed by Hydra when placed inside nested nodes.
                    # Prefer VERL's canonical PPO trainer config as the base to ensure required `_target_` fields
                    # sandbox_fusion = {"url": None, "max_concurrent": 64, "memory_limit_mb": 1024}

                    ppo_trainer = RayPPOTrainer(
                        config=ppo_config,
                        tokenizer=model_tokenizer,
                        role_worker_mapping=role_worker_mapping,
                        resource_pool_manager=resource_pool_manager[i],
                        ray_worker_group_cls=ray_worker_group_cls,
                        
                    )
                    ppo_trainer.global_steps = 0
                    
                    self.ppo_trainer_dict[model_name] = ppo_trainer
                    self.tokenizer_dict[model_name] = model_tokenizer
                    colorful_print(f"PPO trainer created for model: {model_name}", "green")
        colorful_print(f"the number of ppo_trainer_dict: {len(self.ppo_trainer_dict)}", "green")
        
        colorful_print(f"Number of PPO trainers: {len(self.ppo_trainer_dict)}", "cyan")
        colorful_print(f"Number of agent mappings: {len(self.agent_policy_mapping)}", "cyan")
        
        
        self.llm_servers = []





    def init_multi_agent_sys_execution_engine(self):
        from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
        # Get the rollout engines and tokenizers from the trainers
        self.rollout_engine_dict = {}
        self.tokenizer_dict = {}
        self.server_address_dict = {}
        
        for model_name, trainer in self.ppo_trainer_dict.items():
            self.rollout_engine_dict[model_name] = trainer.async_rollout_manager
            self.tokenizer_dict[model_name] = trainer.tokenizer
            rollout_engine = trainer.async_rollout_manager
            server_address_list = getattr(rollout_engine, "server_addresses", [])
            self.server_address_dict[model_name] = server_address_list
 
            # Construct an independent Router for each model
            
        
        self.agent_execution_engine = MultiAgentsExecutionEngine(
            config=self.config,
            ppo_trainer_config_dict=self.ppo_trainer_config_dict,
            tokenizer_dict=self.tokenizer_dict,
            processor_dict=self.processor_dict,
            server_address_dict=self.server_address_dict,
            agent_policy_mapping=self.agent_policy_mapping,
        )

    def init_workers(self):
  
     
        colorful_print("Initializing workers for all PPO trainers...", "cyan")
        if not self.ppo_trainer_dict:
            colorful_print("No PPO trainers to initialize", "yellow")
            return

        colorful_print(f"Initializing {len(self.ppo_trainer_dict)} trainers sequentially (each trainer spawns workers in parallel)...", "blue")
        
        for idx, (model_name, trainer) in enumerate(self.ppo_trainer_dict.items(), 1):
            colorful_print(f"[{idx}/{len(self.ppo_trainer_dict)}] Initializing workers for: {model_name}", "blue")
            try:
                trainer.init_workers()  # 内部会并行创建 Ray workers
                colorful_print(f"✓ [{idx}/{len(self.ppo_trainer_dict)}] Successfully initialized: {model_name}", "green")
            except Exception as e:
                colorful_print(f"✗ Failed to initialize {model_name}: {str(e)}", "red")
                raise RuntimeError(f"Failed to initialize trainer {model_name}") from e
        
        colorful_print(f"All {len(self.ppo_trainer_dict)} trainers initialized successfully!", "green")

    def _update_parameters(self, batch, ppo_trainer, timing_raw):
        #TODO: uid
        ppo_trainer.global_steps += 1
        
        
        # Initialize metrics dictionary if not exists
        if not hasattr(batch, 'meta_info'):
            batch.meta_info = {}
        if 'metrics' not in batch.meta_info:
            batch.meta_info['metrics'] = {}
        #TODO: repeat to align with repeated responses in rollout
        #batch = batch.repeat(repeat_times=ppo_trainer.config.actor_rollout_ref.rollout.n, interleave=True)
        #batch = batch.union(gen_batch_output)

        # padding the batch to the same length
        # prompts: left padding
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in batch.batch["prompts"]],
            batch_first=True,
            padding_value=ppo_trainer.tokenizer.pad_token_id,
        ).flip(dims=[1])
        # responses: right padding
        responses_batch = torch.nn.utils.rnn.pad_sequence(
            [i for i in batch.batch["responses"]],
            batch_first=True,
            padding_value=ppo_trainer.tokenizer.pad_token_id,
        )
        # response_mask may be absent; safely compute it if missing, otherwise keep padding
        if "response_mask" in batch.batch:
            response_mask_batch = torch.nn.utils.rnn.pad_sequence(
                [i for i in batch.batch["response_mask"]],
                batch_first=True,
                padding_value=0,
            )
        else:
            response_mask_batch = None
        #TODO: try if not pad to the max length, the performance is better
        # prompts: left padding
        prompts_batch = pad_sequence_to_length(prompts_batch, ppo_trainer.config.data.max_prompt_length, ppo_trainer.tokenizer.pad_token_id, left_pad=True)
        # responses: right padding  
        responses_batch = pad_sequence_to_length(responses_batch, ppo_trainer.config.data.max_response_length, ppo_trainer.tokenizer.pad_token_id, left_pad=False)
        if response_mask_batch is not None:
            # response_mask: right padding (same as responses)
            response_mask_batch = pad_sequence_to_length(
                response_mask_batch,
                ppo_trainer.config.data.max_response_length,
                0,
                left_pad=False,
            )
        input_ids_batch=torch.cat([prompts_batch, responses_batch], dim=1)
        attention_mask_batch = torch.where(input_ids_batch != ppo_trainer.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask_batch, dim=1) - 1) * attention_mask_batch


        batch.batch["prompts"] = prompts_batch
        batch.batch["responses"] = responses_batch
        batch.batch["input_ids"] = input_ids_batch
        batch.batch["attention_mask"] = attention_mask_batch
        batch.batch["position_ids"] = position_ids
        # If response_mask is absent, generate mask based on non-padding tokens in responses
        # Since responses use right padding, valid tokens are on the left side
        if response_mask_batch is None:
            # Valid tokens in responses are 1; padding tokens are 0
            response_mask_batch = (responses_batch != ppo_trainer.tokenizer.pad_token_id).to(attention_mask_batch.dtype)
        batch.batch["response_mask"] = response_mask_batch
        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # Add reward tensor calculation
        reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        
        # Since responses_batch now uses right padding, valid tokens are on the left
        # We need to find the last valid token position for each sequence
        response_attention_mask = (responses_batch != ppo_trainer.tokenizer.pad_token_id)
        
        # Calculate valid token counts for each sequence
        valid_token_counts = response_attention_mask.sum(dim=-1)
        valid_sequences_mask = valid_token_counts > 0
        
        if valid_sequences_mask.any():
            # For right-padded sequences, find the last valid token position
            # This is much simpler: last_valid_position = valid_token_count - 1
            valid_batch_indices = torch.where(valid_sequences_mask)[0]
            last_valid_positions = valid_token_counts[valid_batch_indices] - 1
            
            # Get rewards for valid sequences
            rewards_tensor = torch.tensor([batch.non_tensor_batch["reward"][i] for i in valid_batch_indices.tolist()], 
                                        dtype=torch.float32, device=reward_tensor.device)
            
            # Place rewards at the last valid token position for each sequence
            reward_tensor[valid_batch_indices, last_valid_positions] = rewards_tensor

        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]



        # recompute old_log_probs
        with simple_timer("old_log_prob", timing_raw):
            # 防御性填充，确保 DataProto 可被 actor_rollout_wg 等分
            try:
                dp_world_size = ppo_trainer.actor_rollout_wg.world_size
            except Exception:
                dp_world_size = 1
            if dp_world_size > 1:
                batch, _ = pad_dataproto_to_divisor(batch, dp_world_size)
            old_log_prob = ppo_trainer.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)


        if ppo_trainer.use_reference_policy:
            # compute reference log_prob
            with simple_timer("ref", timing_raw):
                if not ppo_trainer.ref_in_actor:
                    ref_log_prob = ppo_trainer.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = ppo_trainer.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

            # compute values
        if ppo_trainer.use_critic:
            with simple_timer("values", timing_raw):
                values = ppo_trainer.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with simple_timer("adv", timing_raw):

            # compute advantages, executed on the driver process

            norm_adv_by_std_in_grpo = ppo_trainer.config.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            )  # GRPO adv normalization factor

            batch = compute_advantage(
                batch,
                adv_estimator=ppo_trainer.config.algorithm.adv_estimator,
                gamma=ppo_trainer.config.algorithm.gamma,
                lam=ppo_trainer.config.algorithm.lam,
                num_repeat=ppo_trainer.config.actor_rollout_ref.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=ppo_trainer.config.algorithm,
            )

        # update critic
        if ppo_trainer.use_critic:
            with simple_timer("update_critic", timing_raw):
                critic_output = ppo_trainer.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            batch.meta_info["metrics"].update(critic_output_metrics)

        # implement critic warmup
        if ppo_trainer.config.trainer.critic_warmup <= ppo_trainer.global_steps:
            # update actor
            with simple_timer("update_actor", timing_raw):
                batch.meta_info["multi_turn"] = ppo_trainer.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = ppo_trainer.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            batch.meta_info["metrics"].update(actor_output_metrics)

        # Log rollout generations if enabled
        rollout_data_dir = ppo_trainer.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with simple_timer("dump_rollout_generations", timing_raw):
                reward_extra_infos_dict: dict[str, list] = {}
                inputs = ppo_trainer.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                outputs = ppo_trainer.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                if "request_id" in batch.non_tensor_batch:
                    reward_extra_infos_dict.setdefault(
                        "request_id",
                        batch.non_tensor_batch["request_id"].tolist(),
                    )
                ppo_trainer._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )

    

    def _initialize_logger_safely(self):
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.project_name,
            experiment_name=self.config.experiment_name,
            default_backend=self.config.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        return logger

    def fit(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        logger = self._initialize_logger_safely()
        

        self.global_steps = 0
        
        # Use the minimum total training steps among sub-trainers as the global total
        self.total_training_steps = getattr(self.config, "trainer.total_training_steps", 200)
        progress_bar = tqdm(range(self.total_training_steps), desc="Training Progress", position=0, leave=True)
       
        # we start from step 1
        last_val_metrics = None
        self.max_steps_duration = 0

        #epoch_pbar = tqdm(range(self.config.trainer.total_epochs), desc="Epochs", position=0, leave=True)
        
        for i,step in enumerate(progress_bar):
            
            progress_bar.set_description(f"Step {self.global_steps}")
            pprint(f"step {self.global_steps} started")
            #for batch_dict in self.train_dataloader:
                #batch: DataProto = DataProto.from_single_dict(batch_dict)
            meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since 
                    "global_step": self.global_steps,
                }
            batch_size = self.config.data.gen_batch_size
            # Get batch_num from traindataset
            
            #init dataproto with batch_size
            batch_per_trainer: Dict[str,DataProto]={}
            
            # Load data for each trainer
            for model_name in self.ppo_trainer_dict.keys():
                # For now, create a placeholder batch
                batch_per_trainer[model_name] = DataProto.from_dict({})  # Placeholder
                
            metrics = {}
            timing_raw = {}
            validation_summary={}
            last_resample_mode="train"

            # load checkpoint before doing anything
            if i==0:
                
                colorful_print(f"Starting initial validation for multi-agent system", "cyan")
                start_time = time.time()
                # validation before training
                val_metrics = self._validate()
                last_resample_mode = "validate"
         
                metrics.update(val_metrics)
                
                current_avg_success_rate = val_metrics.get('validation/average/success_rate', 0.0)
                pprint(f"Initial validation metrics logged")
                print(f"Time taken to validate: {time.time() - start_time}")
                
                # 打印验证摘要
                agent_summary = {}
                for key, value in val_metrics.items():
                    if "/success_rate" in key and "/agent_" in key:
                        agent_name = key.split("/agent_")[1].split("/")[0]
                        agent_summary[agent_name] = value
                
             

                # Process each trainer's batch

            with simple_timer("step", timing_raw):
                
                with simple_timer("collect_trajectory", timing_raw):
                    resample_freq=self.config.data.get("resample_freq",10)
                    resample_=False
                    if self.global_steps%resample_freq==0 or i==0 or last_resample_mode=="validate":
                        resample_=True
                    
                    self.agent_execution_engine.init_agents_and_envs(mode="train",step_idx=i,resample=resample_)
                    last_resample_mode="train"
                    for model_name,rollout_engine in self.rollout_engine_dict.items():
                        rollout_engine.wake_up()
                    gen_batch_output_per_policy =asyncio.run( self.agent_execution_engine.generate_multiple_rollouts_concurrent(self.agent_execution_engine.env_idx_list,rollout_mode=self.config.get("sample_mode","no_tree")))
                    for model_name, trainer in self.ppo_trainer_dict.items():
                        dp_world_size = trainer.actor_rollout_wg.world_size
                        batch_per_trainer_temp = self._pad_dataproto_to_world_size(
                            gen_batch_output_per_policy[model_name], dp_world_size
                        )
                        if batch_per_trainer[model_name].batch is None:
                        # If empty, assi`gn directly
                            
                            batch_per_trainer[model_name] = batch_per_trainer_temp
                        else:
                            # Use concat instead of union, because each response content is different
                            
                            batch_per_trainer[model_name] = DataProto.concat([
                                batch_per_trainer[model_name], 
                                batch_per_trainer_temp
                            ])
                    for model_name,rollout_engine in self.rollout_engine_dict.items():
                        rollout_engine.sleep()
                
                # Sort trajectories by their idx, to ensure they are in order.
                timing_raw = {}
                with simple_timer("update_parameters", timing_raw):
                    # Track metrics from all trainers
                    all_trainer_metrics = {}
                    
                    # 使用 ThreadPoolExecutor 进行并行更新（避免 Ray remote 的 self 引用问题）
                    def update_single_trainer(model_name, batch, trainer):
                        """更新单个 trainer 的参数"""
                        try:
                            # 为每个任务创建独立的 timing 字典
                            local_timing_raw = {}
                            
                            # 调用原有的更新逻辑
                            self._update_parameters(batch, trainer, local_timing_raw)
                            
                            # 提取 metrics
                            trainer_metrics = {}
                            if hasattr(batch, 'meta_info') and 'metrics' in batch.meta_info:
                                trainer_metrics = batch.meta_info['metrics']
                            
                            # 提取 agent names 用于后续处理
                            agent_names = None
                            if hasattr(batch, 'non_tensor_batch') and 'agent_name' in batch.non_tensor_batch:
                                agent_names = batch.non_tensor_batch['agent_name']
                            
                            return {
                                "status": "success",
                                "model_name": model_name,
                                "timing": local_timing_raw,
                                "metrics": trainer_metrics,
                                "agent_names": agent_names
                            }
                        except Exception as e:
                            import traceback
                            return {
                                "status": "error",
                                "model_name": model_name,
                                "error": str(e),
                                "traceback": traceback.format_exc()
                            }
                    
                    # 收集需要更新的 trainers
                    tasks_to_submit = []
                    for model_name, trainer in self.ppo_trainer_dict.items():
                        if model_name in gen_batch_output_per_policy:
                            tasks_to_submit.append((model_name, batch_per_trainer[model_name], trainer))
                    
                    if not tasks_to_submit:
                        colorful_print("No trainers to update", "yellow")
                    else:
                        colorful_print(f"Starting parallel parameter updates for {len(tasks_to_submit)} trainers...", "cyan")
                        
                        # 使用线程池并行执行（Ray 的 PPO trainer 内部已经使用 Ray 做分布式）
                        with ThreadPoolExecutor(max_workers=len(tasks_to_submit)) as executor:
                            # 提交所有任务
                            futures = {}
                            for model_name, batch, trainer in tasks_to_submit:
                                future = executor.submit(update_single_trainer, model_name, batch, trainer)
                                futures[future] = model_name
                                colorful_print(f"  Submitted update task for: {model_name}", "blue")
                            
                            # 使用 tqdm 显示进度
                            update_pbar = tqdm(total=len(futures), desc="Updating Parameters", position=2, leave=False)
                            
                            # 等待所有任务完成
                            results = []
                            for future in as_completed(futures):
                                result = future.result()
                                results.append(result)
                                update_pbar.update(1)
                                update_pbar.set_description(f"Updated {result.get('model_name', 'unknown')}")
                            
                            update_pbar.close()
                        
                        # 处理结果：合并 timing 和 metrics
                        success_count = 0
                        for result in results:
                            if result["status"] == "success":
                                model_name = result["model_name"]
                                colorful_print(f"✓ Updated parameters for: {model_name}", "green")
                                success_count += 1
                                
                                # 合并 timing 信息（取最大值，因为是并行执行）
                                for key, value in result["timing"].items():
                                    if key in timing_raw:
                                        timing_raw[key] = max(timing_raw[key], value)
                                    else:
                                        timing_raw[key] = value
                                
                                # 处理 metrics
                                trainer_metrics = result["metrics"]
                                agent_names = result["agent_names"]
                                
                                # Check if we have agent_name information to split metrics by agent
                                if agent_names is not None:
                                    unique_agents = list(set(agent_names))
                                    # Split metrics by agent
                                    for agent_name in unique_agents:
                                        for key, value in trainer_metrics.items():
                                            prefixed_key = f"agent_{agent_name}/{key}"
                                            all_trainer_metrics[prefixed_key] = value
                                else:
                                    # Fallback: use model name prefix (for backward compatibility)
                                    for key, value in trainer_metrics.items():
                                        prefixed_key = f"model_{model_name}/{key}"
                                        all_trainer_metrics[prefixed_key] = value
                            else:
                                colorful_print(f"✗ Failed to update {result['model_name']}: {result['error']}", "red")
                                colorful_print(f"Traceback:\n{result['traceback']}", "red")
                                raise RuntimeError(f"Failed to update trainer {result['model_name']}")
                        
                        colorful_print(f"All {success_count} trainers updated successfully!", "green")
                    
                    # Add trainer metrics to main metrics
                    metrics.update(all_trainer_metrics)
                

                #save checkpoint done
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0 and self.global_steps != 1:
                    with simple_timer("save_checkpoint", timing_raw):
                        for model_name, trainer in self.ppo_trainer_dict.items():
                            experiment_name = getattr(trainer.config, 'experiment_name', 'default_experiment')
                            colorful_print(f"Saving checkpoint for trainer {model_name} to experiment: {experiment_name}", "cyan")
                            trainer._save_checkpoint()

            # TODO: collect metrics
            # Use the first trainer's batch for metrics calculation
    
            for model_name, batch in batch_per_trainer.items():
                for metric_name, metric_value in compute_data_metrics(batch=batch, use_critic=any(trainer.use_critic for trainer in self.ppo_trainer_dict.values())).items():
                    metric_name_policy= model_name + "_" + metric_name
                    metrics[metric_name_policy] = metric_value
                
                for metric_name, metric_value in compute_timing_metrics(batch=batch, timing_raw=timing_raw).items():
                    metric_name_policy= model_name + "_" + metric_name
                    metrics[metric_name_policy] = metric_value
            
            # Standard data and timing metrics
            #metrics.update(compute_data_metrics(batch=first_batch, use_critic=any(trainer.use_critic for trainer in self.ppo_trainer_dict.values())))
            #metrics.update(compute_timing_metrics(batch=first_batch, timing_raw=timing_raw))
                    
            # Add training step metrics
            metrics.update({
                "training/global_step": self.global_steps,
                
            })

            self.global_steps += 1

            if self.global_steps % self.config.data.val_freq == 0:
                val_metrics = self._validate()
                last_resample_mode = "validate"
                
                # 将验证指标添加到主metrics字典中
                metrics.update(val_metrics)
        
                # 打印验证摘要
                agent_summary = {}
                for key, value in val_metrics.items():
                    if "/success_rate" in key and "/agent_" in key:
                        agent_name = key.split("/agent_")[1].split("/")[0]
                        agent_summary[agent_name] = value
         
            # TODO: make a canonical logger that supports various backend
            try:
                logger.log(data=metrics, step=self.global_steps)
            except Exception as e:
                pprint(f"Warning: Failed to log metrics to logger: {type(e).__name__}: {e}")
                pprint(f"Metrics that failed to log: {list(metrics.keys())}")
            # Check if any trainer has reached its total training steps
            if self.global_steps >= self.total_training_steps:
                progress_bar.close()
                
                # perform final validation and print summary
               
                return
        
        progress_bar.close()

    def _validate(self):
        self.agent_execution_engine.init_agents_and_envs(mode="validate")
        batch_per_trainer: Dict[str,DataProto]={}
        for model_name in self.ppo_trainer_dict.keys():
            batch_per_trainer[model_name] = DataProto.from_dict({})
            
        for _, rollout_engine in self.rollout_engine_dict.items():
            rollout_engine.wake_up()
            
        gen_batch_output_per_policy =asyncio.run( self.agent_execution_engine.generate_multiple_rollouts_concurrent(self.agent_execution_engine.env_idx_list))
        for model_name,rollout_engine in self.rollout_engine_dict.items():
            rollout_engine.sleep()
        for model_name in self.ppo_trainer_dict.keys():
            if batch_per_trainer[model_name].batch is None:
            # If empty, assi`gn directly
                batch_per_trainer[model_name] = gen_batch_output_per_policy[model_name]
            else:
                # Use concat instead of union, because each response content is different
                batch_per_trainer[model_name] = DataProto.concat([
                    batch_per_trainer[model_name], 
                    gen_batch_output_per_policy[model_name]
                ])
        for model_name,rollout_engine in self.rollout_engine_dict.items():
            rollout_engine.sleep()
                

        total_rollout_num = len(self.agent_execution_engine.rollout_idx_list)
        success_rollout_rate_dict: Dict[str, float] = {}
        success_turn_ave_dict: Dict[str, float] = {}
        for agent_name in self.agent_execution_engine.turn_order:
            success_rollout_num = len(
                set(self.agent_execution_engine.success_rollout_idx_list_dict.get(agent_name, []))
            )
            if success_rollout_num > 0:
                success_ave_turn = self.agent_execution_engine.success_ave_turn_dict.get(agent_name, 0)/success_rollout_num
            else:
                success_ave_turn = self.agent_execution_engine.config.env.max_turns
            success_rollout_rate_dict[agent_name] = (
                success_rollout_num / total_rollout_num if total_rollout_num > 0 else 0.0
            )
            success_turn_ave_dict[agent_name] = success_ave_turn
        
        save_dir = getattr(self.config.trainer, "validation_data_dir", None)
        
        # 构建验证指标字典 - 只包含每个agent的成功率和平均turn数，以及总体平均值
        validation_metrics = {}
        
        # 添加每个agent的success rate和avg_turns
        for agent_name in self.agent_execution_engine.turn_order:
            success_rate = success_rollout_rate_dict.get(agent_name, 0.0)
            avg_turns = success_turn_ave_dict.get(agent_name, 0.0)
            
            validation_metrics[f"validation/agent_{agent_name}/success_rate"] = success_rate
            validation_metrics[f"validation/agent_{agent_name}/avg_turns"] = avg_turns
        
        # 计算总体平均值
        if success_rollout_rate_dict:
            success_rates = list(success_rollout_rate_dict.values())
            avg_turns_list = list(success_turn_ave_dict.values())
            
            validation_metrics["validation/average/success_rate"] = sum(success_rates) / len(success_rates)
            validation_metrics["validation/average/avg_turns"] = sum(avg_turns_list) / len(avg_turns_list)
            
        return validation_metrics
    


    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        

        # Get the relevant tensors
        prompts = tensor_batch.batch["prompts"]
        responses = tensor_batch.batch["responses"]
        traj_mask = tensor_batch.batch[mask_key]
        token_level_scores = tensor_batch.batch["token_level_scores"]

        batch_size = prompts.shape[0]
        end_idx = min(sample_idx + max_samples, batch_size)

        for i in range(sample_idx, end_idx):
            colorful_print(f"\n===== Sample {i} =====", fg="cyan", bold=True)

            # Detokenize prompt
            prompt_tokens = prompts[i]
            prompt_mask = prompt_tokens != self.tokenizer.pad_token_id
            valid_prompt_tokens = prompt_tokens[prompt_mask]
            prompt_text = self.tokenizer.decode(valid_prompt_tokens)

            colorful_print("Prompt:", fg="green", bold=True)
            colorful_print(f"{prompt_text}\n", fg="green")

            # Detokenize response with color highlighting for masked tokens
            response_tokens = responses[i]
            response_mask = traj_mask[i]

            # Get non-padding tokens
            valid_indices = response_tokens != self.tokenizer.pad_token_id
            valid_response_tokens = response_tokens[valid_indices]
            valid_response_mask = response_mask[valid_indices]

            # Then show token-by-token with masking
            colorful_print("Response with masking:", fg="yellow", bold=True)

            for j, (token, mask) in enumerate(zip(valid_response_tokens, valid_response_mask, strict=False)):
                token_text = self.tokenizer.decode(token)

                # Check if this token has a reward
                has_reward = token_level_scores[i, j] != 0

                # Apply different colors based on mask and rewards
                if mask == 0:
                    # Masked token (not used in training)
                    colorful_print(token_text, fg="red", end="")
                elif has_reward:
                    # Token with reward
                    colorful_print(token_text, bg="green", end="")

                    reward_info = ""
                    if has_reward:
                        reward_info += f" R:{token_level_scores[i, j].item():.2f}"

                    colorful_print(reward_info, fg="magenta", end="")
                else:
                    # Normal token used in training
                    colorful_print(token_text, fg="blue", end="")

            print()  # New line after all tokens

            # Print reward summary
            total_reward = token_level_scores[i].sum().item()
            colorful_print("Rewards:", fg="green", bold=True)
            print(f" Trajectory Reward={total_reward:.2f}")

  
    def _pad_dataproto_to_world_size(self, batch, world_sizes):
        #world_sizes = self.config.data.tensor_model_parallel_size
        batch, pad_size = pad_dataproto_to_divisor(batch, world_sizes)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        return batch
    
    def _cleanup_llm_servers(self, servers):
        """清理 LLM servers"""
        for server in servers:
            try:
                ray.kill(server)
                colorful_print(f"Killed LLM server: {server}", "yellow")
            except Exception as e:
                colorful_print(f"Error killing LLM server {server}: {e}", "red")
    
    def cleanup(self):
        """清理所有资源"""
        try:
            # 清理 LLM servers
            if hasattr(self, 'llm_servers') and self.llm_servers:
                colorful_print("Cleaning up LLM servers...", "yellow")
                self._cleanup_llm_servers(self.llm_servers)
                self.llm_servers.clear()
            
            # 清理 PPO trainers
            if hasattr(self, 'ppo_trainer_dict'):
                for model_name, trainer in self.ppo_trainer_dict.items():
                    try:
                        # 如果 trainer 有清理方法，调用它
                        if hasattr(trainer, 'cleanup'):
                            trainer.cleanup()
                        colorful_print(f"Cleaned up trainer for model: {model_name}", "yellow")
                    except Exception as e:
                        colorful_print(f"Error cleaning up trainer for {model_name}: {e}", "red")
            
            colorful_print("Multi-agent trainer cleanup completed", "green")
        except Exception as e:
            colorful_print(f"Error during cleanup: {e}", "red")
