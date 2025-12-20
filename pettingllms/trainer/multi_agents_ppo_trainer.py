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
from pettingllms.trainer.multi_agents_execution_engine import MultiAgentsExecutionEngine
from verl import DataProto
#from pettingllms.trainer.multi_agents_execution_engine_graph import MultiAgentsExecutionEngineGraph
from verl.protocol import pad_dataproto_to_divisor
from concurrent.futures import ThreadPoolExecutor, as_completed
from verl.trainer.ppo.ray_trainer import (
    
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    compute_advantage,
    compute_data_metrics,
    compute_timing_metrics,
    reduce_metrics,
)

from pettingllms.verl.ray_trainer import RayPPOTrainer
from verl.utils.torch_functional import pad_sequence_to_length
from typing import Dict
from pettingllms.utils.performance import simple_timer,colorful_print
from pettingllms.utils.clean_up import cleanup_old_image_folders
import ray



class MultiAgentsPPOTrainer:
    def __init__(
        self,
        config,
        tokenizer_dict,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        agent_policy_mapping: dict = None,
        processor_dict=None,
    ):
        self.config = config
        self.processor_dict = processor_dict or {}
        self.tokenizer_dict = tokenizer_dict
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        
        # Initialize basic attributes
        self.best_success_rate = -1.0
        self.env_success_rate = 0.0
        self.llm_servers = []
        self.ppo_trainer_config_dict = {}
        self.rollout_sample_dict = {}
        self.ppo_trainer_dict = {}
        self.agent_policy_mapping = agent_policy_mapping
        self.agent_lora_mapping = {}
        self.lora_differ_mode = False
        self.lora_num = 1
        # Control variable: whether to use LoRA for generation (False initially for base model)
        self.use_lora_for_generation = False

        # Read agent_untrained configuration
        self.agent_untrained = []
        if hasattr(config, 'multi_agent_interaction') and hasattr(config.multi_agent_interaction, 'agent_untrained'):
            self.agent_untrained = config.multi_agent_interaction.agent_untrained
            colorful_print(f"Agents excluded from training: {self.agent_untrained}", "yellow")

        if config.specialization =="lora":
            self.lora_num = len(self.agent_policy_mapping)
            self.lora_differ_mode = True
            for agent_idx, agent_name in enumerate(self.agent_policy_mapping.keys()):
                    lora_id = agent_idx+1  # Use integer ID directly ( 1, 2, ...)
                    self.agent_lora_mapping[agent_name] = lora_id
                  
        
        # Step 2: Initialize PPO trainers based on specialization
        self._initialize_ppo_trainers()

   

    def _initialize_ppo_trainers(self):
        """Initialize PPO trainers based on specialization mode"""
        config = self.config
        specialization = config.specialization
        
        
        if specialization in ["prompt", "lora"]:
            # Single PPO trainer for prompt/lora specialization
            self._create_single_ppo_trainer()
        else:
            # Multiple PPO trainers for full/other specialization
            self._create_multiple_ppo_trainers()
        
    

    def _create_single_ppo_trainer(self):
        """Create a single PPO trainer for prompt/lora specialization"""
        config = self.config
        model_key = list(config.models.keys())[0]
        model_config = config.models[model_key]
        model_name = model_config.name
        
        if not hasattr(model_config, 'ppo_trainer_config'):
            raise ValueError(f"Model '{model_name}' missing ppo_trainer_config")
        
        ppo_config = model_config.ppo_trainer_config
        ppo_config.actor_rollout_ref.model.lora_rank = config.get("lora_rank", 0)
        ppo_config.actor_rollout_ref.model.lora_alpha = config.get("lora_alpha", 16)
        if ppo_config.actor_rollout_ref.model.lora_rank > 0:
            print("Enabling LoRA in single PPO trainer")
            ppo_config.actor_rollout_ref.rollout.enable_lora = True
            ppo_config.actor_rollout_ref.rollout.max_loras = self.lora_num
            ppo_config.trainer.experiment_name = config.training.experiment_name
            ppo_config.actor_rollout_ref.rollout.max_lora_rank = config.get("lora_rank", 0)
        else:
            ppo_config.actor_rollout_ref.rollout.enable_lora = False
            ppo_config.actor_rollout_ref.rollout.max_loras = 0
            ppo_config.actor_rollout_ref.rollout.max_lora_rank = 0
        self.ppo_trainer_config_dict[model_name] = ppo_config
        ppo_config.data["train_batch_size"] = config.training.train_batch_size
        
        ppo_trainer = RayPPOTrainer(
            config=ppo_config,
            tokenizer=self.tokenizer_dict[model_name],
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager[0],
            ray_worker_group_cls=self.ray_worker_group_cls,
        )
        ppo_trainer.lora_num = self.lora_num
        ppo_trainer.agent_lora_mapping = self.agent_lora_mapping
        ppo_trainer.global_steps = 0
        
        self.ppo_trainer_dict[model_name] = ppo_trainer

    def _create_multiple_ppo_trainers(self):
        """Create multiple PPO trainers for full/other specialization modes"""
        config = self.config
        
        for i, (model_key, model_config) in enumerate(config.models.items()):
            model_name = model_config.name
            
            if not hasattr(model_config, 'ppo_trainer_config'):
                continue
            
            ppo_config = model_config.ppo_trainer_config
            self.ppo_trainer_config_dict[model_name] = ppo_config
            ppo_config.data["train_batch_size"] = config.training.train_batch_size
            ppo_config.actor_rollout_ref.model.lora_rank = config.get("lora_rank", 0)
            ppo_config.actor_rollout_ref.model.lora_alpha = config.get("lora_alpha", 16)
            ppo_config.trainer.experiment_name = config.training.experiment_name
            
            if ppo_config.actor_rollout_ref.model.lora_rank > 0:
                ppo_config.actor_rollout_ref.rollout.enable_lora = True
                ppo_config.actor_rollout_ref.rollout.max_loras = self.lora_num if hasattr(self, 'lora_num') else 1
                ppo_config.actor_rollout_ref.rollout.max_lora_rank = config.get("lora_rank", 0)
            else:
                ppo_config.actor_rollout_ref.rollout.enable_lora = False
                ppo_config.actor_rollout_ref.rollout.max_loras = 0
                ppo_config.actor_rollout_ref.rollout.max_lora_rank = 0
            
            ppo_trainer = RayPPOTrainer(
                config=ppo_config,
                tokenizer=self.tokenizer_dict[model_name],
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager[i],
                ray_worker_group_cls=self.ray_worker_group_cls,
            )
            ppo_trainer.global_steps = 0
            
            self.ppo_trainer_dict[model_name] = ppo_trainer





    def init_multi_agent_sys_execution_engine(self):
        self.rollout_engine_dict = {}
        self.tokenizer_dict = {}
        self.server_address_dict = {}
        
        for model_name, trainer in self.ppo_trainer_dict.items():
            self.rollout_engine_dict[model_name] = trainer.async_rollout_manager
            self.tokenizer_dict[model_name] = trainer.tokenizer
            rollout_engine = trainer.async_rollout_manager
            server_address_list = getattr(rollout_engine, "server_addresses", [])
            self.server_address_dict[model_name] = server_address_list
 
        workflow_type = getattr(self.config, 'workflow_type', 'turn')
        colorful_print(f"Initializing execution engine with workflow_type: {workflow_type}", "cyan")
        
        if workflow_type == "graph":
            colorful_print("Initializing MultiAgentsExecutionEngineGraph", "cyan")
            from pettingllms.trainer.multi_agents_execution_engine_graph import MultiAgentsExecutionEngineGraph
            self.agent_execution_engine = MultiAgentsExecutionEngineGraph(
                config=self.config,
                ppo_trainer_config_dict=self.ppo_trainer_config_dict,
                tokenizer_dict=self.tokenizer_dict,
                processor_dict=self.processor_dict,
                server_address_dict=self.server_address_dict,
                agent_policy_mapping=self.agent_policy_mapping,
                lora_differ_mode=self.lora_differ_mode,
                agent_lora_mapping=self.agent_lora_mapping,
                use_lora_for_generation=self.use_lora_for_generation,
            )
        elif workflow_type == "autoevol":
            colorful_print("Initializing MultiAgentsExecutionEngineAutoEvol", "cyan")
            from pettingllms.trainer.multi_agents_execution_engine_autoevol import MultiAgentsExecutionEngineAutoEvol
            self.agent_execution_engine = MultiAgentsExecutionEngineAutoEvol(
                config=self.config,
                ppo_trainer_config_dict=self.ppo_trainer_config_dict,
                tokenizer_dict=self.tokenizer_dict,
                processor_dict=self.processor_dict,
                server_address_dict=self.server_address_dict,
                agent_policy_mapping=self.agent_policy_mapping,
                lora_differ_mode=self.lora_differ_mode,
                agent_lora_mapping=self.agent_lora_mapping,
                use_lora_for_generation=self.use_lora_for_generation,
            )
        else:
            colorful_print("Initializing MultiAgentsExecutionEngine (turn-based)", "cyan")
            self.agent_execution_engine = MultiAgentsExecutionEngine(
                config=self.config,
                ppo_trainer_config_dict=self.ppo_trainer_config_dict,
                tokenizer_dict=self.tokenizer_dict,
                processor_dict=self.processor_dict,
                server_address_dict=self.server_address_dict,
                agent_policy_mapping=self.agent_policy_mapping,
                lora_differ_mode=self.lora_differ_mode,
                agent_lora_mapping=self.agent_lora_mapping,
                use_lora_for_generation=self.use_lora_for_generation,
            )

    def init_workers(self):
  
     
        colorful_print("Initializing workers for all PPO trainers...", "cyan")
        if not self.ppo_trainer_dict:
            colorful_print("No PPO trainers to initialize", "yellow")
            return

        colorful_print(f"Initializing {len(self.ppo_trainer_dict)} trainers sequentially (each trainer spawns workers in parallel)...", "blue")
        
        for idx, (model_name, trainer) in enumerate(self.ppo_trainer_dict.items(), 1):
            colorful_print(f"[{idx}/{len(self.ppo_trainer_dict)}] Initializing workers for: {model_name}", "blue")
            if self.lora_differ_mode:
                    trainer.init_workers(lora_num=self.lora_num, agent_lora_mapping=self.agent_lora_mapping)
                    colorful_print(f"  Initialized with {self.lora_num} LoRA adapters for multi-agent training", "cyan")
            else:
                trainer.init_workers(lora_num=self.lora_num)
            colorful_print(f"✓ [{idx}/{len(self.ppo_trainer_dict)}] Successfully initialized: {model_name}", "green")
        
        colorful_print(f"All {len(self.ppo_trainer_dict)} trainers initialized successfully!", "green")
        


    def _update_parameters(self, batch, ppo_trainer, timing_raw):


        # Initialize metrics dictionary if not exists
        if not hasattr(batch, 'meta_info'):
            batch.meta_info = {}
        if 'metrics' not in batch.meta_info:
            batch.meta_info['metrics'] = {}

        # Filter out data from untrained agents
        if self.agent_untrained and len(self.agent_untrained) > 0:
            if 'agent_name' in batch.non_tensor_batch:
                agent_names = batch.non_tensor_batch['agent_name']
                # Keep only samples from agents that are not in agent_untrained list
                keep_indices = [i for i, name in enumerate(agent_names) if name not in self.agent_untrained]

                if len(keep_indices) < len(agent_names):
                    colorful_print(f"Filtering training data: keeping {len(keep_indices)}/{len(agent_names)} samples (excluding agents: {self.agent_untrained})", "yellow")
                    batch = batch.select_idxs(keep_indices)

                    # If all samples are filtered out, return early
                    if len(keep_indices) == 0:
                        colorful_print("Warning: All samples filtered out, skipping parameter update", "red")
                        return batch

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
        if "response_mask" in batch.batch.keys():
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

      
        # update actor
        with simple_timer("update_actor", timing_raw):
            batch.meta_info["multi_turn"] = ppo_trainer.config.actor_rollout_ref.rollout.multi_turn.enable
            
            if self.lora_differ_mode:
                agent_names = batch.non_tensor_batch['agent_name']
                unique_agents = sorted(set(agent_names))

                # Filter out untrained agents from unique_agents list
                if self.agent_untrained:
                    unique_agents = [agent for agent in unique_agents if agent not in self.agent_untrained]
                    if len(unique_agents) < len(set(agent_names)):
                        colorful_print(f"LoRA mode: Excluding untrained agents {self.agent_untrained} from training", "yellow")

                agent_batch_dict = {}
                for agent_name in unique_agents:
                    agent_mask = np.array([name == agent_name for name in agent_names])
                    agent_indices = np.where(agent_mask)[0].tolist()
                    # Construct sub-batch for each agent and align to dp world size if needed to avoid blocking in distributed updates
                    sub_batch = batch.select_idxs(agent_indices)
                    try:
                        dp_world_size = ppo_trainer.actor_rollout_wg.world_size
                    except Exception:
                        dp_world_size = 1
                    if dp_world_size > 1:
                        sub_batch, _ = pad_dataproto_to_divisor(sub_batch, dp_world_size)
                    agent_batch_dict[agent_name] = sub_batch
                    colorful_print(f"Agent {agent_name}: {len(agent_indices)} samples (training enabled)", "cyan")
                
                # Collect metrics from all agents
                all_actor_metrics_list = []
                for agent_name, agent_batch in agent_batch_dict.items():
                    colorful_print(f"Updating LoRA for agent: {agent_name}", "green")
                    agent_output = ppo_trainer.actor_rollout_wg.update_actor(agent_batch)
                    all_actor_metrics_list.append(agent_output.meta_info["metrics"])
                
                # Merge metrics from multiple agents
                # Convert List[Dict[str, value]] to Dict[str, List[value]]
                if all_actor_metrics_list:
                    from collections import defaultdict
                    merged_metrics = defaultdict(list)
                    for metrics_dict in all_actor_metrics_list:
                        for key, value in metrics_dict.items():
                            # Ensure value is a scalar before appending
                            if isinstance(value, (list, tuple, np.ndarray)):
                                # If value is already a collection, take its mean
                                merged_metrics[key].append(float(np.mean(value)))
                            else:
                                merged_metrics[key].append(float(value))
                    # Now reduce the merged metrics
                    actor_output_metrics = reduce_metrics(dict(merged_metrics))
                else:
                    actor_output_metrics = {}
            else:
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

            # Return the potentially updated batch so caller can keep latest fields
            return batch

    

    def _initialize_logger_safely(self):
        from verl.utils.tracking import Tracking
        from datetime import datetime
        import os
        
        # Generate log path: logs/experiment_name/date/time
        current_time = datetime.now()
        date_str = current_time.strftime("%m-%d")
        time_str = current_time.strftime("%H-%M-%S")
        
        experiment_name = self.config.training.experiment_name
        log_dir = os.path.join("logs", experiment_name, date_str, time_str)
        os.makedirs(log_dir, exist_ok=True)
        
        logger = Tracking(
            project_name=self.config.training.project_name,
            experiment_name=experiment_name,
            default_backend=self.config.training.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        colorful_print(f"Logger initialized with log_dir: {log_dir}", "cyan")
        return logger

    def fit(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        logger = self._initialize_logger_safely()

        # Load checkpoint if resume is enabled
        # This must be done after init_workers() and before training loop
        for trainer in self.ppo_trainer_dict.values():
            loaded_step = trainer._load_checkpoint()
            if loaded_step > 0:
                colorful_print(f"Resumed training from global step {loaded_step}", "green")
                self.global_steps = loaded_step
                break  # All trainers should have the same global_steps
        else:
            self.global_steps = 0

        self.total_training_steps = self.config.training.total_training_steps
        progress_bar = tqdm(range(self.total_training_steps), desc="Training Progress", position=0, leave=True)
        self.max_steps_duration = 0
        
        while self.global_steps < self.total_training_steps:
            progress_bar.update(1)
            progress_bar.set_description(f"Step {self.global_steps}")
            pprint(f"step {self.global_steps} started")
            
            batch_per_trainer: Dict[str,DataProto]={}
            for model_name in self.ppo_trainer_dict.keys():
                batch_per_trainer[model_name] = DataProto.from_dict({})  # Placeholder
                
            metrics = {}
            timing_raw = {}
            if self.global_steps == 0:
                colorful_print(f"Starting initial validation for multi-agent system (using base model)", "cyan")
                # Ensure use_lora_for_generation is False for initial validation
                self.use_lora_for_generation = False
                self.agent_execution_engine.use_lora_for_generation = False
                colorful_print(f"use_lora_for_generation set to False for step 0 validation", "yellow")
                start_time = time.time()
                # validation before training
                #val_metrics = self._validate(global_steps=self.global_steps)
         
                #metrics.update(val_metrics)
                
                #current_avg_success_rate = val_metrics.get('validation/average/success_rate', 0.0)
               
                #agent_summary = {}
                #for key, value in val_metrics.items():
                #    if "/success_rate" in key and "/agent_" in key:
                #        agent_name = key.split("/agent_")[1].split("/")[0]
                #        agent_summary[agent_name] = value
                
             

                # Process each trainer's batch

            with simple_timer("step", timing_raw):

                with simple_timer("collect_trajectory", timing_raw):
                    # Step 0: Use base model for trajectory collection (LoRA not trained yet)
                    if self.global_steps == 0 and self.lora_differ_mode:
                        self.use_lora_for_generation = False
                        self.agent_execution_engine.use_lora_for_generation = False
                        colorful_print(f"Step {self.global_steps}: Using base model for trajectory collection (LoRA not trained yet)", "yellow")

                    self.agent_execution_engine.init_agents_and_envs(mode="train", step_idx=self.global_steps)

                    # CRITICAL: Always wake_up before use and sleep after use to maintain strict pairing
                    # This ensures wake_up and sleep are always paired 1:1
                    for model_name,rollout_engine in self.rollout_engine_dict.items():
                        rollout_engine.wake_up()

                    # Sync any pending LoRA updates to vLLM AFTER wake_up (to avoid memory conflicts)
                    # Note: LoRA is prepared in step N's update_actor, synced at step N+1's wake_up
                    if self.lora_differ_mode and self.global_steps >= 1:
                        
                        self.use_lora_for_generation = True
                        self.agent_execution_engine.use_lora_for_generation = True
                        
                    gen_batch_output_per_policy =asyncio.run( self.agent_execution_engine.generate_multiple_rollouts_concurrent(self.agent_execution_engine.env_idx_list,rollout_mode=self.config.get("rollout_mode","tree")))
                    
                    # Always sleep after trajectory collection to maintain strict pairing
                    for model_name,rollout_engine in self.rollout_engine_dict.items():
                        rollout_engine.sleep()
                    
                    for model_name, trainer in self.ppo_trainer_dict.items():
                        dp_world_size = trainer.actor_rollout_wg.world_size
                        batch_per_trainer_temp = self._pad_dataproto_to_world_size(
                            gen_batch_output_per_policy[model_name], dp_world_size
                        )
                        if batch_per_trainer[model_name].batch is None:
                        # If empty, assign directly
                            
                            batch_per_trainer[model_name] = batch_per_trainer_temp
                        else:
                            # Use concat instead of union because each response content is different
                            batch_per_trainer[model_name] = DataProto.concat([
                                batch_per_trainer[model_name], 
                                batch_per_trainer_temp
                            ])
                
                timing_raw = {}
                with simple_timer("update_parameters", timing_raw):
                    # Apply UID assignment and filtering for each model
                    sample_num = self.config.training.train_sample_num
                    for model_name, trainer in self.ppo_trainer_dict.items():
                        if model_name in batch_per_trainer and batch_per_trainer[model_name].batch is not None:
                            filter_ratio = getattr(trainer.config, 'filter_ratio', 0.0)
                            filter_method = getattr(trainer.config, 'filter_method', 'uid')
                            batch_per_trainer[model_name] = self._assign_consistent_uids(
                                batch_per_trainer[model_name], 
                                filter_ratio=filter_ratio, 
                                mode=filter_method, 
                                sample_num=sample_num,
                                rollout_mode=self.config.get("rollout_mode","tree")
                            )
                    
                    all_trainer_metrics = {}
                    
                    def update_single_trainer(model_name, batch, trainer):
                        
                        local_timing_raw = {}
                        # Keep the updated batch with advantages/returns for later metrics
                        updated_batch = self._update_parameters(batch, trainer, local_timing_raw)
                        
                        trainer_metrics = updated_batch.meta_info.get('metrics', {}) if hasattr(updated_batch, 'meta_info') else {}
                        agent_names = updated_batch.non_tensor_batch.get('agent_name') if hasattr(updated_batch, 'non_tensor_batch') else None
                        
                        return {"status": "success", "model_name": model_name, "timing": local_timing_raw, 
                                "metrics": trainer_metrics, "agent_names": agent_names, "updated_batch": updated_batch}
                    
                
                    # Update trainers
                    for model_name, trainer in self.ppo_trainer_dict.items():
                        if model_name in gen_batch_output_per_policy:
                            result = update_single_trainer(model_name, batch_per_trainer[model_name], trainer)
                            
                            if result["status"] == "error":
                                colorful_print(f"Error updating trainer for {model_name}: {result['error']}", "red")
                                colorful_print(f"Traceback:\n{result['traceback']}", "red")
                                raise RuntimeError(f"Failed to update trainer for {model_name}: {result['error']}")
                            # Merge timing metrics
                            for key, value in result["timing"].items():
                                timing_raw[key] = max(timing_raw.get(key, 0), value)
                            
                            # Merge trainer metrics by agent
                            trainer_metrics = result["metrics"]
                        

                            # Replace the trainer's batch with the updated version for downstream metrics
                            if "updated_batch" in result and result["updated_batch"] is not None:
                                batch_per_trainer[model_name] = result["updated_batch"]
                    
                    metrics.update(all_trainer_metrics)
                    
                    # After step 1 training completes, enable LoRA for future generations
                    # Note: LoRA weights are automatically synced to vLLM during update_actor()
                    if self.global_steps == 1 and self.lora_differ_mode and not self.use_lora_for_generation:
                        self.use_lora_for_generation = True
                        self.agent_execution_engine.use_lora_for_generation = True
                        colorful_print(f"Step {self.global_steps}: LoRA training completed, weights auto-synced to vLLM, enabling LoRA for generations", "green")

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

            

            if self.global_steps % self.config.training.val_freq == 0 and self.global_steps != 0:
                val_metrics = self._validate(global_steps=self.global_steps)
                metrics.update(val_metrics)
                agent_summary = {}
                for key, value in val_metrics.items():
                    if "/success_rate" in key and "/agent_" in key:
                        agent_name = key.split("/agent_")[1].split("/")[0]
                        agent_summary[agent_name] = value
            self.global_steps += 1
            for ppo_trainer in self.ppo_trainer_dict.values():
                ppo_trainer.global_steps = self.global_steps
            try:
                logger.log(data=metrics, step=self.global_steps)
            except Exception as e:
                pprint(f"Warning: Failed to log metrics to logger: {type(e).__name__}: {e}")
                pprint(f"Metrics that failed to log: {list(metrics.keys())}")

            # Clean up old image folders if multimodal is enabled
            enable_multimodal = getattr(self.config.training, 'enable_multimodal', False)
            if enable_multimodal:
                try:
                    # Get image save directory from config
                    image_save_dir = "tmp_image"  # default
                    if hasattr(self.config, 'env') and hasattr(self.config.env, 'image_save_dir'):
                        image_save_dir = self.config.env.image_save_dir
                    elif hasattr(self.config.training, 'image_save_dir'):
                        image_save_dir = self.config.training.image_save_dir

                    # Get max subfolders from config (default: 20)
                    max_image_steps = getattr(self.config.training, 'max_image_steps', 20)

                    # Clean up old image folders
                    cleanup_old_image_folders(
                        base_dir=image_save_dir,
                        max_subfolders=max_image_steps,
                        verbose=True
                    )
                except Exception as e:
                    pprint(f"Warning: Failed to clean up image folders: {type(e).__name__}: {e}")

            # Check if any trainer has reached its total training steps
            if self.global_steps >= self.total_training_steps:
                progress_bar.close()
                
                # perform final validation and print summary
               
                return
        
        progress_bar.close()

    def _save_best_checkpoint(self, env_success_rate):
        """
        Save checkpoint if the current env_success_rate is better than the best recorded.
        
        Args:
            env_success_rate: Current validation environment success rate
        """
        if_save = getattr(self.config.training, 'if_save', True)

        if not if_save:
            colorful_print(f"Checkpoint saving disabled (if_save=False). Current env success rate: {env_success_rate:.4f}", "yellow")
            return

        # Allow checkpoint saving at step 0 for testing purposes
        # if self.global_steps == 0:
        #     colorful_print(f"Skip saving checkpoint at step 0. Current env success rate: {env_success_rate:.4f}", "yellow")
        #     return

        # Only save if this is a new best result
        if env_success_rate <= self.best_success_rate:
            colorful_print(f"Current env success rate: {env_success_rate:.4f} (best: {self.best_success_rate:.4f})", "yellow")
            return

        # Update best success rate and save checkpoint
        self.best_success_rate = env_success_rate
        colorful_print(f"New best env success rate: {env_success_rate:.4f}, saving checkpoint...", "green")

        from datetime import datetime
        import os
        import shutil

        current_time = datetime.now()
        date_str = current_time.strftime("%Y%m%d")
        experiment_name = self.config.training.experiment_name

        base_checkpoint_dir = "checkpoints"
        save_base = self.config.specialization != "lora"
        spec = self.config.specialization
        save_jobs = []

        # Determine which trainers to save based on specialization mode
        if spec == "prompt":
            for _, trainer in self.ppo_trainer_dict.items():
                save_jobs.append(("shared_model", trainer))
        elif spec == "lora":
            for agent_name, policy_name in self.agent_policy_mapping.items():
                trainer = self.ppo_trainer_dict[policy_name]
                save_jobs.append((agent_name, trainer))
        elif spec == "full":
            num_base_models = len(self.config.base_models) if hasattr(self.config, "base_models") else 0
            if num_base_models == 1:
                for agent_name, policy_name in self.agent_policy_mapping.items():
                    trainer = self.ppo_trainer_dict[policy_name]
                    save_jobs.append((agent_name, trainer))
            else:
                for model_name, trainer in self.ppo_trainer_dict.items():
                    save_jobs.append((model_name, trainer))
        else:
            for model_name, trainer in self.ppo_trainer_dict.items():
                save_jobs.append((model_name, trainer))

        # Save each trainer's checkpoint
        for target_name, trainer in save_jobs:
            trainer._save_checkpoint(save_base=save_base)


    def _validate(self, global_steps=0):
        self.agent_execution_engine.init_agents_and_envs(mode="validate", step_idx=global_steps)
        batch_per_trainer: Dict[str,DataProto]={}
        for model_name in self.ppo_trainer_dict.keys():
            batch_per_trainer[model_name] = DataProto.from_dict({})
        

        
        # CRITICAL: Always wake_up before use and sleep after use to maintain strict pairing
        # This ensures wake_up and sleep are always paired 1:1
        for _, rollout_engine in self.rollout_engine_dict.items():
            rollout_engine.wake_up()
            
        gen_batch_output_per_policy =asyncio.run( self.agent_execution_engine.generate_multiple_rollouts_concurrent(self.agent_execution_engine.env_idx_list, rollout_mode="tree"))
        
        # Always sleep after validation to maintain strict pairing
        for model_name,rollout_engine in self.rollout_engine_dict.items():
            rollout_engine.sleep()
        
        for model_name in self.ppo_trainer_dict.keys():
            if batch_per_trainer[model_name].batch is None:
                batch_per_trainer[model_name] = gen_batch_output_per_policy[model_name]
            else:
                batch_per_trainer[model_name] = DataProto.concat([
                    batch_per_trainer[model_name], 
                    gen_batch_output_per_policy[model_name]
                ])
        
        # Calculate success metrics from env state
        total_rollout_num = len(self.agent_execution_engine.rollout_idx_list)
        success_rollout_rate_dict: Dict[str, float] = {}
        success_turn_ave_dict: Dict[str, float] = {}
        env_state_success_count = 0
        
        # Count success from env.state
        for env in self.agent_execution_engine.envs:
            if hasattr(env, 'success') and env.success:
                env_state_success_count += 1
        
        env_success_rate = env_state_success_count / total_rollout_num if total_rollout_num > 0 else 0.0
        
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
        
        validation_metrics = {}
        for agent_name in self.agent_execution_engine.turn_order:
            success_rate = success_rollout_rate_dict.get(agent_name, 0.0)
            avg_turns = success_turn_ave_dict.get(agent_name, 0.0)
            
            validation_metrics[f"validation/agent_{agent_name}/success_rate"] = success_rate
            validation_metrics[f"validation/agent_{agent_name}/avg_turns"] = avg_turns
        
        if success_rollout_rate_dict:
            success_rates = list(success_rollout_rate_dict.values())
            avg_turns_list = list(success_turn_ave_dict.values())
            
            validation_metrics["validation/average/success_rate"] = sum(success_rates) / len(success_rates)
            validation_metrics["validation/average/avg_turns"] = sum(avg_turns_list) / len(avg_turns_list)
        
        validation_metrics["validation/env_state_success_rate"] = env_success_rate
        
        # Save checkpoint if this is the best validation result
        if global_steps > 0:
            self._save_best_checkpoint(env_success_rate)
            
        return validation_metrics
    
    def _pad_dataproto_to_world_size(self, batch, world_sizes):
        batch, pad_size = pad_dataproto_to_divisor(batch, world_sizes)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        return batch
    
    def _assign_consistent_uids(self, data_proto, filter_ratio=0.0, mode="mean", sample_num=1, rollout_mode="tree"):
        """
        Assign consistent UIDs to data and optionally filter based on rewards.
        
        Args:
            data_proto: DataProto object containing trajectory data
            filter_ratio: Ratio of samples to filter (0.0 to 1.0)
            mode: Filtering mode - "mean", "std", "dapo", or "uid"
            sample_num: Number of samples per environment
        
        Returns:
            Filtered DataProto object
        """
        import uuid
        import numpy as np
        from collections import defaultdict
        
        uid_mapping = {}
        all_rewards = []
        uid_reward_groups = defaultdict(list)

        non_tensor_batch = data_proto.non_tensor_batch
        
        if not all(key in non_tensor_batch for key in ["env_idx", "turn_idx", "agent_idx"]):
            # If required keys are missing, just assign random UIDs and return
            data_proto.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(data_proto))], dtype=object
            )
            return data_proto
        
        rollout_indices = non_tensor_batch["env_idx"]
        turn_indices = non_tensor_batch["turn_idx"] 
        agent_indices = non_tensor_batch["agent_idx"]
        rewards = non_tensor_batch.get("reward", [])
        
        uids = []
        for i in range(len(data_proto)):
            if rollout_mode == "no_tree":
                key = (rollout_indices[i],)
            else:
                key = (rollout_indices[i]//sample_num, turn_indices[i], agent_indices[i])
            if key not in uid_mapping:
                uid_mapping[key] = str(uuid.uuid4())
            uid = uid_mapping[key]
            uids.append(uid)
            
            if len(rewards) > 0 and filter_ratio > 0:
                reward_val = float(rewards[i]) if rewards[i] is not None else 0.0
                uid_reward_groups[uid].append((i, reward_val))
            
            if len(rewards) > 0:
                reward_val = float(rewards[i]) if rewards[i] is not None else 0.0
                all_rewards.append(reward_val)
        
        data_proto.non_tensor_batch["uid"] = np.array(uids, dtype=object)
    
        
        def range_normalized_variance(rewards_in_group):
            """Calculate variance normalized by the range squared"""
            rewards_in_group = np.asarray(rewards_in_group, dtype=float)
            rng = np.max(rewards_in_group) - np.min(rewards_in_group)
            if rng == 0:
                return 0.0
            return np.var(rewards_in_group, ddof=0) / (rng ** 2)
        
        sample_to_remove = set()
        if rollout_mode == "no_tree":
            # For no_tree mode, keep only samples with maximum turn_indices for each env
            env_max_turn = {}
            for i in range(len(data_proto)):
                env_id = rollout_indices[i]
                turn_id = turn_indices[i]
                if env_id not in env_max_turn:
                    env_max_turn[env_id] = turn_id
                else:
                    env_max_turn[env_id] = max(env_max_turn[env_id], turn_id)
            
            # Mark samples with non-maximum turn_indices for removal
            for i in range(len(data_proto)):
                env_id = rollout_indices[i]
                turn_id = turn_indices[i]
                if turn_id < env_max_turn[env_id]:
                    sample_to_remove.add(i)
            
            colorful_print(f"no_tree mode: keeping only max turn_indices samples, removing {len(sample_to_remove)} samples", "yellow")
        elif mode == "dapo":
            uids_to_remove = []
            for uid, samples in uid_reward_groups.items():
                rewards_in_group = [s[1] for s in samples]
                variance = range_normalized_variance(rewards_in_group)
                if variance==0:
                    uids_to_remove.append(uid)
            for uid in uids_to_remove:
                if uid in uid_reward_groups:
                    for sample_idx, reward_val in uid_reward_groups[uid]:
                        sample_to_remove.add(sample_idx)

        elif filter_ratio > 0:
            # Calculate the variance of each uid group
            
            if mode == "std":
                uid_variances = {}
                for uid, samples in uid_reward_groups.items():
                    if len(samples) > 1:
                        rewards_in_group = [s[1] for s in samples]
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
                                for sample_idx, reward_val in uid_reward_groups[uid]:
                                    sample_to_remove.add(sample_idx)
            elif mode == "mean":
                uid_means = {}
                for uid, samples in uid_reward_groups.items():
                    if len(samples) > 1:
                        rewards_in_group = [s[1] for s in samples]
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
                                for sample_idx, reward_val in uid_reward_groups[uid]:
                                    sample_to_remove.add(sample_idx)
            elif mode=="uid":
                if filter_ratio > 0:
                    for uid, samples in uid_reward_groups.items():
                        if len(samples) > 1:
                            rewards_in_group = [s[1] for s in samples]
                            group_mean = np.mean(rewards_in_group)
                            samples_with_deviation = [(s[0], abs(s[1] - group_mean)) for s in samples]
                            samples_with_deviation.sort(key=lambda x: x[1], reverse=True)
                            num_to_remove = int(len(samples_with_deviation) * filter_ratio)
                            for i in range(num_to_remove):
                                sample_idx, _ = samples_with_deviation[i]
                                sample_to_remove.add(sample_idx)
        
        if sample_to_remove and len(sample_to_remove) > 0:
            keep_indices = [i for i in range(len(data_proto)) 
                           if i not in sample_to_remove]
            
            if len(keep_indices) < len(data_proto):
                # Use DataProto's built-in select_idxs method for more robust filtering
                data_proto = data_proto.select_idxs(keep_indices)
        
        if all_rewards:
            summary = {
                "total_samples": len(all_rewards),
                "mean_reward": float(np.mean(all_rewards)),
                "std_reward": float(np.std(all_rewards)),
                "filtered_samples": len(sample_to_remove) if filter_ratio > 0 else 0,
                "remain_samples": len(data_proto)
            }
            
            colorful_print(f"UID assignment summary: {summary}", "green")
        
        return data_proto
    
    def _cleanup_llm_servers(self, servers):
       
        for server in servers:
            try:
                ray.kill(server)
                colorful_print(f"Killed LLM server: {server}", "yellow")
            except Exception as e:
                colorful_print(f"Error killing LLM server {server}: {e}", "red")
    
    def cleanup(self):
        """Clean up all resources including trainers and resource pools"""
        try:
            colorful_print("Starting MultiAgentsPPOTrainer cleanup...", "yellow")

            # Clean up execution engine
            if hasattr(self, 'agent_execution_engine') and self.agent_execution_engine is not None:
                try:
                    if hasattr(self.agent_execution_engine, 'cleanup'):
                        self.agent_execution_engine.cleanup()
                    colorful_print("Cleaned up agent_execution_engine", "yellow")
                except Exception as e:
                    colorful_print(f"Error cleaning up agent_execution_engine: {e}", "red")

            # Clean up aiohttp sessions
            try:
                import asyncio
                from pettingllms.trainer.async_generate import cleanup_shared_session

                # Try to get the current event loop, or create a new one if needed
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_closed():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # Run the cleanup coroutine
                loop.run_until_complete(cleanup_shared_session())
                colorful_print("Cleaned up aiohttp shared session", "yellow")
            except Exception as e:
                colorful_print(f"Error cleaning up aiohttp session: {e}", "yellow")

            # Clean up LLM servers
            if hasattr(self, 'llm_servers') and self.llm_servers:
                colorful_print("Cleaning up LLM servers...", "yellow")
                self._cleanup_llm_servers(self.llm_servers)
                self.llm_servers.clear()

            # Clean up PPO trainers
            if hasattr(self, 'ppo_trainer_dict'):
                colorful_print(f"Cleaning up {len(self.ppo_trainer_dict)} PPO trainers...", "yellow")
                for model_name, trainer in self.ppo_trainer_dict.items():
                    try:
                        # Call the trainer's cleanup method
                        if hasattr(trainer, 'cleanup'):
                            trainer.cleanup()
                        colorful_print(f"Cleaned up trainer for model: {model_name}", "yellow")
                    except Exception as e:
                        colorful_print(f"Error cleaning up trainer for {model_name}: {e}", "red")
                self.ppo_trainer_dict.clear()

            # Clean up resource pool managers
            if hasattr(self, 'resource_pool_manager') and self.resource_pool_manager is not None:
                try:
                    if isinstance(self.resource_pool_manager, list):
                        colorful_print(f"Cleaning up {len(self.resource_pool_manager)} resource pool managers...", "yellow")
                        for i, manager in enumerate(self.resource_pool_manager):
                            try:
                                if hasattr(manager, 'cleanup'):
                                    manager.cleanup()
                                colorful_print(f"Cleaned up resource pool manager {i}", "yellow")
                            except Exception as e:
                                colorful_print(f"Error cleaning up resource pool manager {i}: {e}", "red")
                    else:
                        if hasattr(self.resource_pool_manager, 'cleanup'):
                            self.resource_pool_manager.cleanup()
                        colorful_print("Cleaned up resource_pool_manager", "yellow")
                except Exception as e:
                    colorful_print(f"Error cleaning up resource_pool_manager: {e}", "red")

            colorful_print("Multi-agent trainer cleanup completed", "green")
        except Exception as e:
            colorful_print(f"Error during cleanup: {e}", "red")
