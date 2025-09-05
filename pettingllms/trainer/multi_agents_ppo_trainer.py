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
            # 保存全部可用的 server addresses，以便下游进行随机选择/负载均衡
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
        """Initialize workers for all PPO trainers."""
        colorful_print("Initializing workers for all PPO trainers...", "cyan")
        if not self.ppo_trainer_dict:
            colorful_print("No PPO trainers to initialize", "yellow")
            return

        
        for model_name, trainer in self.ppo_trainer_dict.items():
            trainer.init_workers()
            colorful_print(f"Initialized workers for trainer: {model_name}", "green")
        colorful_print("All workers initialized successfully", "green")

    def _update_parameters(self, batch, ppo_trainer, timing_raw):
        #TODO: uid
        ppo_trainer.global_steps += 1
        batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
        
        # Initialize metrics dictionary if not exists
        if not hasattr(batch, 'meta_info'):
            batch.meta_info = {}
        if 'metrics' not in batch.meta_info:
            batch.meta_info['metrics'] = {}
        #TODO: repeat to align with repeated responses in rollout
        #batch = batch.repeat(repeat_times=ppo_trainer.config.actor_rollout_ref.rollout.n, interleave=True)
        #batch = batch.union(gen_batch_output)

        # padding the batch to the same length
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in batch.batch["prompts"]],
            batch_first=True,
            padding_value=ppo_trainer.tokenizer.pad_token_id,
        ).flip(dims=[1])
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
        prompts_batch = pad_sequence_to_length(prompts_batch, ppo_trainer.config.data.max_prompt_length, ppo_trainer.tokenizer.pad_token_id, left_pad=True)
        responses_batch = pad_sequence_to_length(responses_batch, ppo_trainer.config.data.max_response_length, ppo_trainer.tokenizer.pad_token_id, left_pad=True)
        if response_mask_batch is not None:
            response_mask_batch = pad_sequence_to_length(
                response_mask_batch,
                ppo_trainer.config.data.max_response_length,
                0,
                left_pad=True,
            )
        input_ids_batch=torch.cat([prompts_batch, responses_batch], dim=1)
        attention_mask_batch = torch.where(input_ids_batch != ppo_trainer.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask_batch, dim=1) - 1) * attention_mask_batch


        batch.batch["prompts"] = prompts_batch
        batch.batch["responses"] = responses_batch
        batch.batch["input_ids"] = input_ids_batch
        batch.batch["attention_mask"] = attention_mask_batch
        batch.batch["position_ids"] = position_ids
        # If response_mask is absent, generate a right-side mask based on non-padding tokens in responses
        if response_mask_batch is None:
            # Valid tokens in responses are 1; padding tokens are 0
            response_mask_batch = (responses_batch != ppo_trainer.tokenizer.pad_token_id).to(attention_mask_batch.dtype)
        batch.batch["response_mask"] = response_mask_batch
        


        

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        #add reward tensor calculation
        reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        
        # 正确计算response mask，基于responses_batch本身而不是拼接后的input_ids
        # responses_batch 使用了左填充，所以需要找到每个序列的最后一个有效token位置
        response_attention_mask = (responses_batch != ppo_trainer.tokenizer.pad_token_id)
        
        # 找到每行的最后一个有效token位置（从右往左第一个非填充token）
        # 由于是左填充，有效内容在右侧
        batch_size = response_attention_mask.shape[0]
        batch_indices = torch.arange(batch_size)
        
        # 计算每行有效token数量
        valid_token_counts = response_attention_mask.sum(dim=-1)
        
        # 使用更高效的向量化方式找到最后一个有效token位置
        # 对于左填充的序列，我们需要找到每行最右边的有效token
        valid_sequences_mask = valid_token_counts > 0
        
        if valid_sequences_mask.any():
            # 找到每行最后一个有效token的位置
            # 翻转mask，找到第一个True的位置，然后转换回原始索引
            flipped_mask = response_attention_mask.flip(dims=[1]).float()  # 转换为float以支持argmax
            last_valid_positions_from_right = flipped_mask.argmax(dim=1)  # 从右边开始的位置
            last_valid_positions = response_attention_mask.shape[1] - 1 - last_valid_positions_from_right
            
            # 只对有效序列设置奖励
            valid_batch_indices = torch.where(valid_sequences_mask)[0]
            rewards_tensor = torch.tensor([batch.non_tensor_batch["reward"][i] for i in valid_batch_indices.tolist()], 
                                        dtype=torch.float32, device=reward_tensor.device)
            
            reward_tensor[valid_batch_indices, last_valid_positions[valid_batch_indices]] = rewards_tensor

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

    

    def fit(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.project_name,
            experiment_name=self.config.experiment_name,
            default_backend=self.config.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

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
                
                colorful_print(f"Loading checkpoint for trainer {model_name}", "cyan")
                start_time = time.time()
                # validation before training
                val_metrics = self._validate()
                last_resample_mode="validate"
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
                print(f"Time taken to validate agent_{model_name}: {time.time() - start_time}")
                for agent_name, success_rate in val_metrics.items():
                    metrics[f"validation/agent_{agent_name}/success_rate"] = success_rate
                    validation_summary[agent_name] = success_rate
                if val_metrics:
                    success_rates = list(val_metrics.values())
                    metrics["validation/overall/avg_success_rate"] = sum(success_rates) / len(success_rates)



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
                    
                    update_pbar = tqdm(self.ppo_trainer_dict.items(), desc="Updating Parameters", position=2, leave=False)
                    
                    for model_name, trainer in update_pbar:
                        update_pbar.set_description(f"Updating {model_name}")
                        
                        # Update parameters for the corresponding policy/model
                        if model_name not in gen_batch_output_per_policy:
                            # Skip if this model has not generated data
                            continue
                        self._update_parameters(
                            batch_per_trainer[model_name],
                            trainer,
                            timing_raw,
                        )
                                                # Collect metrics from each trainer's batch
                        if hasattr(batch_per_trainer[model_name], 'meta_info') and 'metrics' in batch_per_trainer[model_name].meta_info:
                            trainer_metrics = batch_per_trainer[model_name].meta_info['metrics']
                            
                            # Check if we have agent_name information to split metrics by agent
                            if hasattr(batch_per_trainer[model_name], 'non_tensor_batch') and 'agent_name' in batch_per_trainer[model_name].non_tensor_batch:
                                agent_names = batch_per_trainer[model_name].non_tensor_batch['agent_name']
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
                    update_pbar.close()
                    
                    # Add trainer metrics to main metrics
                    metrics.update(all_trainer_metrics)
                

                #save checkpoint done
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with simple_timer("save_checkpoint", timing_raw):
                        for model_name, trainer in self.ppo_trainer_dict.items():
                            colorful_print(f"Saving checkpoint for trainer {model_name}", "cyan")
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
                last_resample_mode="validate"
                # val_metrics是success_rollout_rate_dict，包含每个agent的成功率
                validation_summary = {}
                
                for agent_name, success_rate in val_metrics.items():
                    metrics[f"validation/agent_{agent_name}/success_rate"] = success_rate
                    validation_summary[agent_name] = success_rate
                
                # 计算总体统计信息
                if val_metrics:
                    success_rates = list(val_metrics.values())
                    metrics["validation/overall/avg_success_rate"] = sum(success_rates) / len(success_rates)
                
                pprint(f"Validation Summary - Step {self.global_steps}: {validation_summary}")
                pprint(f"Overall avg success rate: {metrics.get('validation/overall/avg_success_rate', 0.0):.4f}")
            
            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)
            # Check if any trainer has reached its total training steps
            if self.global_steps >= self.total_training_steps:
                progress_bar.close()
                
                # perform validation after training
                first_trainer = next(iter(self.ppo_trainer_dict.values()))
                if first_trainer.val_reward_fn is not None:
                    #val_metrics = first_trainer._validate()
                    pprint(f"Final validation metrics: skip")
                    #logger.log(data=val_metrics, step=self.global_steps)
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
                
        # 统计成功率
        total_rollout_num = len(self.agent_execution_engine.rollout_idx_list)
        success_rollout_rate_dict: Dict[str, float] = {}
        for agent_name in self.agent_execution_engine.turn_order:
            success_rollout_num = len(
                self.agent_execution_engine.success_rollout_idx_list_dict.get(agent_name, [])
            )
            success_rollout_rate_dict[agent_name] = (
                success_rollout_num / total_rollout_num if total_rollout_num > 0 else 0.0
            )
        # 可选：保存验证样本到本地目录
        save_dir = getattr(self.config.trainer, "validation_data_dir", None)
        return success_rollout_rate_dict

    def visualize_trajectory(self, tensor_batch, sample_idx=0, max_samples=1, mask_key="traj_mask"):
        """
        Visualize the trajectory from tensor_batch by detokenizing prompts and responses,
        and highlighting the masked parts with color.

        Args:
            tensor_batch: The tensor batch containing trajectory data
            sample_idx: Starting index of samples to visualize
            max_samples: Maximum number of samples to visualize
        """
        from pettingllms.misc import colorful_print

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
