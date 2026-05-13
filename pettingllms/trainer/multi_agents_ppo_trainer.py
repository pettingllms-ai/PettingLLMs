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
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
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
from pettingllms.verl.ray_trainer import apply_kl_penalty
from verl.trainer.ppo import core_algos

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
        tokenizer_path_dict=None,
        role_worker_mapping: dict[Role, WorkerType] = None,
        resource_pool_manager: ResourcePoolManager = None,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        agent_policy_mapping: dict = None,
        processor_dict=None,
    ):
        self.config = config
        self.processor_dict = processor_dict or {}
        self.tokenizer_dict = tokenizer_dict
        self.tokenizer_path_dict = tokenizer_path_dict or {}
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

        # train_data_mode: "all" | "designer_only" | "executor_only"
        train_data_mode = getattr(config.training, 'train_data_mode', 'all') if hasattr(config, 'training') else 'all'
        if train_data_mode == "designer_only":
            self.agent_untrained = list(set(self.agent_untrained) | {"WorkflowAgent"})
            colorful_print(f"train_data_mode=designer_only: excluding WorkflowAgent from training", "yellow")
        elif train_data_mode == "executor_only":
            self.agent_untrained = list(set(self.agent_untrained) | {"Designer"})
            colorful_print(f"train_data_mode=executor_only: excluding Designer from training", "yellow")
        else:
            colorful_print(f"train_data_mode=all: training on all agent data", "yellow")

        # Separate learning rates for designer vs executor
        self.designer_lr = getattr(config.training, 'designer_lr', None) if hasattr(config, 'training') else None
        self.executor_lr = getattr(config.training, 'executor_lr', None) if hasattr(config, 'training') else None
        # LR alternating: swap designer/executor LR every N steps (legacy soft-freeze mode)
        self.lr_alternate_steps = int(getattr(config.training, 'lr_alternate_steps', 0)) if hasattr(config, 'training') else 0
        # IBR (Iterated Best-Response) co-training: hard-freeze one policy every N steps.
        # Requires split_policy mode (2 independent models). Supersedes lr_alternate_steps.
        self.phase_alternate_steps = int(getattr(config.training, 'phase_alternate_steps', 0)) if hasattr(config, 'training') else 0
        self.first_phase_trains = str(getattr(config.training, 'first_phase_trains', 'executor')) if hasattr(config, 'training') else 'executor'
        # Resolved at fit-time: model_name (policy_name) of the currently-frozen policy, or None.
        self._current_frozen_policy = None
        if self.designer_lr is not None and self.executor_lr is not None:
            colorful_print(f"Separate LRs enabled: designer_lr={self.designer_lr}, executor_lr={self.executor_lr}", "yellow")
            if self.phase_alternate_steps > 0 and self.lr_alternate_steps > 0:
                colorful_print(f"WARNING: phase_alternate_steps and lr_alternate_steps both >0; phase_alternate_steps takes precedence.", "red")
            if self.phase_alternate_steps > 0:
                colorful_print(f"IBR enabled: hard-freeze alternation every {self.phase_alternate_steps} steps, phase 0 trains {self.first_phase_trains}", "yellow")
            elif self.lr_alternate_steps > 0:
                colorful_print(f"LR alternating (soft-freeze) every {self.lr_alternate_steps} steps", "yellow")
        else:
            colorful_print(f"Using shared LR for all agents", "yellow")

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
        
        # Check if this is a split_policy scenario (multiple agents with different policies)
        policy_names = set(agent_config.policy_name for agent_config in config.agent_policy_configs.agent_configs.values())
        num_models = len(config.models) if hasattr(config, 'models') else 0
        is_split_policy = len(policy_names) > 1 and num_models > 1
        
        if specialization in ["prompt", "lora"] and not is_split_policy:
            # Single PPO trainer for prompt/lora specialization (single model)
            self._create_single_ppo_trainer()
        else:
            # Multiple PPO trainers for full/other specialization or split_policy
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
        ppo_config.trainer.experiment_name = config.training.experiment_name
        if ppo_config.actor_rollout_ref.model.lora_rank > 0:
            print("Enabling LoRA in single PPO trainer")
            ppo_config.actor_rollout_ref.rollout.enable_lora = True
            ppo_config.actor_rollout_ref.rollout.max_loras = self.lora_num
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
        # Tag the trainer with its policy name so _save_checkpoint /
        # _load_checkpoint write/read per-policy subdirs (avoids overwriting
        # in split-policy mode).
        ppo_trainer.policy_name = model_name

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
            # Per-policy subdir tag for save/load (see _save_checkpoint).
            ppo_trainer.policy_name = model_name

            self.ppo_trainer_dict[model_name] = ppo_trainer





    def init_multi_agent_sys_execution_engine(self):
        self.rollout_engine_dict = {}
        self.tokenizer_dict = {}
        self.execution_engine_tokenizer_path_dict = {}
        self.server_address_dict = {}

        for model_name, trainer in self.ppo_trainer_dict.items():
            self.rollout_engine_dict[model_name] = trainer.async_rollout_manager
            self.tokenizer_dict[model_name] = trainer.tokenizer
            # Get tokenizer_path from initial tokenizer_path_dict
            if model_name in self.tokenizer_path_dict:
                self.execution_engine_tokenizer_path_dict[model_name] = self.tokenizer_path_dict[model_name]
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
                tokenizer_path_dict=self.execution_engine_tokenizer_path_dict,
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
        import sys
        print(f"[DEBUG HANG] ========== ENTERING _update_parameters ==========", flush=True)
        print(f"[DEBUG HANG] ppo_trainer type: {type(ppo_trainer).__name__}", flush=True)
        sys.stdout.flush()

        # Check if batch is empty or None
        if batch is None:
            colorful_print("Error: batch is None, skipping parameter update", "red")
            raise ValueError("Cannot update parameters: batch is None")
        
        if not hasattr(batch, 'batch') or batch.batch is None:
            colorful_print("Warning: batch.batch is None or empty, skipping parameter update", "red")
            # Return a minimal batch structure to avoid downstream errors
            if not hasattr(batch, 'meta_info'):
                batch.meta_info = {}
            if 'metrics' not in batch.meta_info:
                batch.meta_info['metrics'] = {}
            batch.meta_info['metrics']['skipped'] = True
            batch.meta_info['metrics']['reason'] = "Empty batch (all rollouts failed)"
            return batch
        
        # Check if batch has required keys
        if "prompts" not in batch.batch or "responses" not in batch.batch:
            colorful_print(f"Warning: batch missing required keys. Available keys: {list(batch.batch.keys()) if batch.batch else 'None'}", "red")
            if not hasattr(batch, 'meta_info'):
                batch.meta_info = {}
            if 'metrics' not in batch.meta_info:
                batch.meta_info['metrics'] = {}
            batch.meta_info['metrics']['skipped'] = True
            batch.meta_info['metrics']['reason'] = f"Missing required keys: prompts or responses"
            return batch

        # Initialize metrics dictionary if not exists
        if not hasattr(batch, 'meta_info'):
            batch.meta_info = {}
        if 'metrics' not in batch.meta_info:
            batch.meta_info['metrics'] = {}

        # Filter out data from untrained agents
        if self.agent_untrained and len(self.agent_untrained) > 0:
            if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch and 'agent_name' in batch.non_tensor_batch:
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

        # Check if prompts/responses are empty after filtering
        if len(batch.batch["prompts"]) == 0 or len(batch.batch["responses"]) == 0:
            colorful_print(f"Warning: Empty prompts ({len(batch.batch.get('prompts', []))}) or responses ({len(batch.batch.get('responses', []))}), skipping parameter update", "red")
            batch.meta_info['metrics']['skipped'] = True
            batch.meta_info['metrics']['reason'] = "Empty prompts or responses after filtering"
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
        import sys
        print(f"[DEBUG HANG] ========== ENTERING compute_log_prob ==========", flush=True)
        print(f"[DEBUG HANG] batch size: {len(batch) if batch is not None else 'None'}", flush=True)
        print(f"[DEBUG HANG] batch.batch keys: {list(batch.batch.keys()) if batch.batch is not None else 'None'}", flush=True)
        sys.stdout.flush()

        with simple_timer("old_log_prob", timing_raw):
            try:
                dp_world_size = ppo_trainer.actor_rollout_wg.world_size
                print(f"[DEBUG HANG] dp_world_size: {dp_world_size}", flush=True)
            except Exception as e:
                print(f"[DEBUG HANG] Failed to get world_size: {e}", flush=True)
                dp_world_size = 1
            pad_size = 0
            if dp_world_size > 1:
                print(f"[DEBUG HANG] Padding batch to divisor {dp_world_size}...", flush=True)
                batch, pad_size = pad_dataproto_to_divisor(batch, dp_world_size)
                print(f"[DEBUG HANG] Padding done, new batch size: {len(batch)}, pad_size: {pad_size}", flush=True)

            print(f"[DEBUG HANG] >>> Calling compute_log_prob NOW (this may hang)... <<<", flush=True)
            sys.stdout.flush()
            old_log_prob = ppo_trainer.actor_rollout_wg.compute_log_prob(batch)
            print(f"[DEBUG HANG] <<< compute_log_prob COMPLETED >>>", flush=True)
            sys.stdout.flush()

            batch = batch.union(old_log_prob)
            print(f"[DEBUG HANG] batch.union completed", flush=True)

            # Unpad after compute_log_prob to prevent padding duplicates from
            # contaminating GRPO advantage computation (padding copies first N
            # samples including their UIDs, biasing group mean/std)
            if pad_size > 0:
                batch = unpad_dataproto(batch, pad_size)
                print(f"[DEBUG HANG] Unpadded batch back to {len(batch)}", flush=True)


        # Compute reference log_prob if needed for KL loss or KL in reward
        need_ref_log_prob = ppo_trainer.use_reference_policy or ppo_trainer.config.algorithm.use_kl_in_reward
        if need_ref_log_prob:
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

        # Apply KL penalty to rewards if enabled
        if ppo_trainer.config.algorithm.use_kl_in_reward:
            with simple_timer("kl_penalty", timing_raw):
                # Get or create KL controller
                if not hasattr(ppo_trainer, 'kl_ctrl_in_reward'):
                    ppo_trainer.kl_ctrl_in_reward = core_algos.get_kl_controller(
                        ppo_trainer.config.algorithm.kl_ctrl
                    )
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=ppo_trainer.kl_ctrl_in_reward,
                    kl_penalty=ppo_trainer.config.algorithm.kl_penalty
                )
                batch.meta_info["metrics"].update(kl_metrics)
                colorful_print(f"Applied KL penalty: {kl_metrics}", "cyan")

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
            # Debug: Check advantage statistics                                                                                                                        
            adv = batch.batch["advantages"]                                                                                                                            
            print(f"[GRPO DEBUG] Advantage stats: mean={adv.mean():.6f}, std={adv.std():.6f}, "                                                                        
                    f"zeros={((adv == 0).sum() / adv.numel() * 100):.1f}%, "                                                                                             
                    f"min={adv.min():.6f}, max={adv.max():.6f}")

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
                if self.designer_lr is not None and self.executor_lr is not None:
                    # Split batch by agent name and apply different LRs
                    agent_names = batch.non_tensor_batch['agent_name']
                    unique_agents = sorted(set(agent_names))

                    # Filter out untrained agents
                    if self.agent_untrained:
                        unique_agents = [agent for agent in unique_agents if agent not in self.agent_untrained]

                    # Build sub-batches with override_lr
                    agent_batch_list = []
                    for agent_name in unique_agents:
                        agent_mask = np.array([name == agent_name for name in agent_names])
                        agent_indices = np.where(agent_mask)[0].tolist()
                        if not agent_indices:
                            continue
                        sub_batch = batch.select_idxs(agent_indices)
                        try:
                            dp_world_size = ppo_trainer.actor_rollout_wg.world_size
                        except Exception:
                            dp_world_size = 1
                        if dp_world_size > 1:
                            sub_batch, _ = pad_dataproto_to_divisor(sub_batch, dp_world_size)
                        # Determine effective LR.
                        # IBR mode (phase_alternate_steps > 0): use fixed per-agent LR. The
                        # frozen side never reaches this code path because update_actor is
                        # skipped at the per-model loop level in fit().
                        if self.phase_alternate_steps > 0:
                            override_lr = self.designer_lr if agent_name == "Designer" else self.executor_lr
                        elif self.lr_alternate_steps > 0:
                            # Legacy soft-freeze: swap designer/executor LRs every N steps.
                            phase = (self.global_steps // self.lr_alternate_steps) % 2
                            if phase == 0:
                                effective_designer_lr = self.designer_lr
                                effective_executor_lr = self.executor_lr
                            else:
                                effective_designer_lr = self.executor_lr
                                effective_executor_lr = self.designer_lr
                            override_lr = effective_designer_lr if agent_name == "Designer" else effective_executor_lr
                        else:
                            override_lr = self.designer_lr if agent_name == "Designer" else self.executor_lr
                        sub_batch.meta_info["override_lr"] = override_lr
                        agent_batch_list.append((agent_name, sub_batch))
                        if self.phase_alternate_steps > 0:
                            phase_str = f" [IBR phase={'train' if self._current_frozen_policy != self.agent_policy_mapping.get(agent_name) else 'frozen'}]"
                        elif self.lr_alternate_steps > 0:
                            phase_str = f" [soft-phase={'designer' if (self.global_steps // self.lr_alternate_steps) % 2 == 0 else 'executor'}-focused]"
                        else:
                            phase_str = ""
                        colorful_print(f"Using lr={override_lr} for {agent_name} ({len(agent_indices)} samples){phase_str}", "cyan")

                    # Only step LR scheduler on the last sub-batch
                    for i, (agent_name, sub_batch) in enumerate(agent_batch_list):
                        sub_batch.meta_info["step_lr_scheduler"] = (i == len(agent_batch_list) - 1)

                    # Update actor for each sub-batch and merge metrics
                    all_actor_metrics_list = []
                    for agent_name, sub_batch in agent_batch_list:
                        agent_output = ppo_trainer.actor_rollout_wg.update_actor(sub_batch)
                        all_actor_metrics_list.append(agent_output.meta_info["metrics"])

                    if all_actor_metrics_list:
                        from collections import defaultdict
                        merged_metrics = defaultdict(list)
                        for metrics_dict in all_actor_metrics_list:
                            for key, value in metrics_dict.items():
                                if isinstance(value, (list, tuple, np.ndarray)):
                                    merged_metrics[key].append(float(np.mean(value)))
                                else:
                                    merged_metrics[key].append(float(value))
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

        # Return the updated batch so caller can keep latest fields (advantages, returns, etc.)
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
        _is_first_iter = True
        _ran_resume_val = False

        while self.global_steps < self.total_training_steps:
            progress_bar.update(1)
            progress_bar.set_description(f"Step {self.global_steps}")
            pprint(f"step {self.global_steps} started")

            # IBR (Iterated Best-Response) phase selection:
            # Hard-freeze one policy for phase_alternate_steps consecutive steps, then swap.
            # Requires split_policy mode (len(ppo_trainer_dict) >= 2).
            self._current_frozen_policy = None
            if self.phase_alternate_steps > 0 and len(self.ppo_trainer_dict) >= 2:
                phase = (self.global_steps // self.phase_alternate_steps) % 2
                # Phase 0 trains `first_phase_trains` side (freezes the other).
                trains_executor_in_phase_0 = (self.first_phase_trains.lower() == "executor")
                train_executor_now = (phase == 0) == trains_executor_in_phase_0
                # Resolve frozen policy name via agent_policy_mapping (agent_name -> policy_name).
                if train_executor_now:
                    frozen_agent = "Designer"
                else:
                    frozen_agent = "Executor"
                self._current_frozen_policy = self.agent_policy_mapping.get(frozen_agent)
                colorful_print(
                    f"[IBR] Step {self.global_steps}: phase={phase}, "
                    f"training={'Executor' if train_executor_now else 'Designer'}, "
                    f"frozen_policy={self._current_frozen_policy}",
                    "cyan",
                )

            batch_per_trainer: Dict[str,DataProto]={}
            for model_name in self.ppo_trainer_dict.keys():
                batch_per_trainer[model_name] = DataProto.from_dict({})  # Placeholder
                
            metrics = {}
            timing_raw = {}

            # Resolve val_before_train once (controls both fresh-start and resume-time validation)
            val_before_train = self.config.get("val_before_train", True)
            for trainer in self.ppo_trainer_dict.values():
                vbt = getattr(trainer.config, 'trainer', None)
                if vbt is not None:
                    val_before_train = vbt.get("val_before_train", val_before_train)
                    break

            # val_before_train: run validation BEFORE the first training step (fresh or resumed)
            if _is_first_iter and val_before_train:
                if self.global_steps == 0:
                    colorful_print(f"Running validation BEFORE first training step (base model)", "cyan")
                    if self.lora_differ_mode:
                        self.use_lora_for_generation = False
                        self.agent_execution_engine.use_lora_for_generation = False
                else:
                    colorful_print(f"Running validation at resumed step {self.global_steps}", "cyan")
                val_metrics = self._validate(global_steps=self.global_steps)
                metrics.update(val_metrics)
                try:
                    logger.log(data=metrics, step=self.global_steps)
                except Exception as e:
                    pprint(f"Warning: Failed to log val_before_train metrics: {e}")
                _ran_resume_val = True
            else:
                _ran_resume_val = False
            _is_first_iter = False

            with simple_timer("step", timing_raw):

                with simple_timer("collect_trajectory", timing_raw):
                    # Step 0: Use base model for trajectory collection (LoRA not trained yet)
                    if self.global_steps == 0 and self.lora_differ_mode:
                        self.use_lora_for_generation = False
                        self.agent_execution_engine.use_lora_for_generation = False
                        colorful_print(f"Step {self.global_steps}: Using base model for trajectory collection (LoRA not trained yet)", "yellow")

                    self.agent_execution_engine.init_agents_and_envs(mode="train", step_idx=self.global_steps)

                    # GPU memory diagnostics before rollout
                    try:
                        for gpu_i in range(torch.cuda.device_count()):
                            alloc = torch.cuda.memory_allocated(gpu_i) / 1024**3
                            reserved = torch.cuda.memory_reserved(gpu_i) / 1024**3
                            print(f"[GPU MEM] Step {self.global_steps} GPU{gpu_i}: allocated={alloc:.2f}GB, reserved={reserved:.2f}GB")
                    except Exception:
                        pass

                    # Rollout with automatic retry on vLLM engine crash
                    max_rollout_retries = int(os.environ.get("MAX_ROLLOUT_RETRIES", "3"))
                    gen_batch_output_per_policy = None
                    for _rollout_attempt in range(max_rollout_retries):
                        try:
                            if _rollout_attempt > 0:
                                print(f"[ROLLOUT RETRY] Attempt {_rollout_attempt + 1}/{max_rollout_retries} after engine crash")
                                # Re-init envs for retry (data may need refresh)
                                self.agent_execution_engine.init_agents_and_envs(mode="train", step_idx=self.global_steps)

                            # CRITICAL: Always wake_up before use and sleep after use to maintain strict pairing
                            for model_name, rollout_engine in self.rollout_engine_dict.items():
                                rollout_engine.wake_up()

                            # Sync any pending LoRA updates to vLLM AFTER wake_up
                            if self.lora_differ_mode and self.global_steps >= 1:
                                self.use_lora_for_generation = True
                                self.agent_execution_engine.use_lora_for_generation = True

                            gen_batch_output_per_policy = asyncio.run(
                                self.agent_execution_engine.generate_multiple_rollouts_concurrent(
                                    self.agent_execution_engine.env_idx_list,
                                    rollout_mode=self.config.get("rollout_mode", "tree")
                                )
                            )

                            # Sleep after successful trajectory collection
                            # Note: reset_prefix_cache is handled inside sleep() by the async server
                            # Do NOT call reset_prefix_cache separately — it corrupts vLLM state
                            # when called from a different thread/event loop
                            for model_name, rollout_engine in self.rollout_engine_dict.items():
                                try:
                                    rollout_engine.sleep()
                                except Exception as e:
                                    pprint(f"[WARNING] rollout_engine.sleep() failed for {model_name}: {e}.")

                            # Rollout succeeded, break retry loop
                            break

                        except Exception as rollout_err:
                            print(f"[ROLLOUT CRASH] Step {self.global_steps}, attempt {_rollout_attempt + 1}/{max_rollout_retries}: {type(rollout_err).__name__}: {rollout_err}")
                            import traceback
                            traceback.print_exc()

                            # Force sleep to mark engine as dead (triggers reinit on next wake_up)
                            for model_name, rollout_engine in self.rollout_engine_dict.items():
                                try:
                                    rollout_engine.sleep()
                                except Exception:
                                    pass

                            # Cleanup GPU memory before retry
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            if _rollout_attempt >= max_rollout_retries - 1:
                                print(f"[ROLLOUT FATAL] All {max_rollout_retries} attempts failed at step {self.global_steps}. Raising.")
                                raise

                            print(f"[ROLLOUT RETRY] Waiting 10s before retry to let GPU state settle...")
                            import time
                            time.sleep(10)

                    if gen_batch_output_per_policy is None:
                        print(f"[ROLLOUT SKIP] Step {self.global_steps} produced no output, skipping to next step.")
                        self.global_steps += 1
                        continue

                    for model_name, trainer in self.ppo_trainer_dict.items():
                        dp_world_size = trainer.actor_rollout_wg.world_size
                        batch_per_trainer_temp = self._pad_dataproto_to_world_size(
                            gen_batch_output_per_policy[model_name], dp_world_size
                        )
                        if batch_per_trainer[model_name].batch is None:
                            batch_per_trainer[model_name] = batch_per_trainer_temp
                        else:
                            batch_per_trainer[model_name] = DataProto.concat([
                                    batch_per_trainer[model_name], 
                                    batch_per_trainer_temp
                                ])
                
                # Free generation results after batch is assembled to prevent OOM
                # This is critical for tree design mode (4×8) which produces many DataProtos
                del gen_batch_output_per_policy
                import gc
                gc.collect()
                # Note: torch.cuda.empty_cache() not called here because the
                # coordinator process may not have CUDA access (GPU work runs
                # in Ray worker processes)

                timing_raw = {}
                with simple_timer("update_parameters", timing_raw):
                    # Apply UID assignment and filtering for each model
                    sample_num = self.config.training.train_sample_num
                    design_sample_num = getattr(self.config.training, 'design_sample_num', None)
                    execute_sample_num = getattr(self.config.training, 'execute_sample_num', None)
                    # Auto-calculate sample_num for tree design mode
                    if design_sample_num and execute_sample_num:
                        sample_num = design_sample_num * execute_sample_num
                    for model_name, trainer in self.ppo_trainer_dict.items():
                        if model_name in batch_per_trainer and batch_per_trainer[model_name].batch is not None:
                            filter_ratio = getattr(trainer.config, 'filter_ratio', 0.0)
                            filter_method = getattr(trainer.config, 'filter_method', 'uid')
                            executor_group_mode = getattr(self.config.training, 'executor_group_mode', None)
                            batch_per_trainer[model_name] = self._assign_consistent_uids(
                                batch_per_trainer[model_name],
                                filter_ratio=filter_ratio,
                                mode=filter_method,
                                sample_num=sample_num,
                                rollout_mode=self.config.get("rollout_mode","tree"),
                                design_sample_num=design_sample_num,
                                execute_sample_num=execute_sample_num,
                                executor_group_mode=executor_group_mode,
                            )

                    import sys
                    print(f"[DEBUG HANG] ========== UID ASSIGNMENT LOOP COMPLETED ==========", flush=True)
                    print(f"[DEBUG HANG] batch_per_trainer sizes: {{k: len(v) if v is not None and v.batch is not None else 'None' for k, v in batch_per_trainer.items()}}", flush=True)
                    sys.stdout.flush()

                    all_trainer_metrics = {}
                    
                    def update_single_trainer(model_name, batch, trainer):
                        
                        # Check if batch is valid before processing
                        if batch is None:
                            colorful_print(f"Error: batch is None for model {model_name}, skipping update", "red")
                            return {"status": "error", "model_name": model_name, "timing": {}, 
                                    "metrics": {"error": "batch is None"}, "agent_names": None, "updated_batch": None}
                        
                        if not hasattr(batch, 'batch') or batch.batch is None:
                            colorful_print(f"Warning: batch.batch is None for model {model_name}, skipping update", "red")
                            return {"status": "skipped", "model_name": model_name, "timing": {}, 
                                    "metrics": {"skipped": True, "reason": "Empty batch"}, "agent_names": None, "updated_batch": batch}
                        
                        local_timing_raw = {}
                        try:
                            # Keep the updated batch with advantages/returns for later metrics
                            updated_batch = self._update_parameters(batch, trainer, local_timing_raw)
                            
                            trainer_metrics = updated_batch.meta_info.get('metrics', {}) if hasattr(updated_batch, 'meta_info') else {}
                            agent_names = updated_batch.non_tensor_batch.get('agent_name') if hasattr(updated_batch, 'non_tensor_batch') and updated_batch.non_tensor_batch else None
                            
                            return {"status": "success", "model_name": model_name, "timing": local_timing_raw, 
                                    "metrics": trainer_metrics, "agent_names": agent_names, "updated_batch": updated_batch}
                        except Exception as e:
                            colorful_print(f"Error updating parameters for model {model_name}: {e}", "red")
                            import traceback
                            colorful_print(f"Traceback: {traceback.format_exc()}", "red")
                            return {"status": "error", "model_name": model_name, "timing": local_timing_raw, 
                                    "metrics": {"error": str(e)}, "agent_names": None, "updated_batch": batch}
                    
                
                    # Update trainers
                    import sys
                    print(f"[DEBUG HANG] ========== STARTING TRAINER UPDATE LOOP ==========", flush=True)
                    print(f"[DEBUG HANG] ppo_trainer_dict keys: {list(self.ppo_trainer_dict.keys())}", flush=True)
                    print(f"[DEBUG HANG] gen_batch_output_per_policy: (already freed)", flush=True)
                    print(f"[DEBUG HANG] batch_per_trainer keys: {list(batch_per_trainer.keys())}", flush=True)
                    sys.stdout.flush()

                    for model_name, trainer in self.ppo_trainer_dict.items():
                        print(f"[DEBUG HANG] Processing model: {model_name}", flush=True)
                        # IBR: hard-skip update_actor for the currently-frozen policy.
                        # Weights stay pinned on both FSDP worker and vLLM side (no divergence).
                        if self._current_frozen_policy is not None and model_name == self._current_frozen_policy:
                            colorful_print(
                                f"[IBR] Step {self.global_steps}: {model_name} is frozen, skip update_actor",
                                "cyan",
                            )
                            continue
                        if model_name in batch_per_trainer:
                            # Additional check before calling update_single_trainer
                            if model_name not in batch_per_trainer:
                                colorful_print(f"Warning: model {model_name} not in batch_per_trainer, skipping", "yellow")
                                continue

                            batch = batch_per_trainer[model_name]
                            if batch is None or (hasattr(batch, 'batch') and batch.batch is None):
                                colorful_print(f"Warning: Empty batch for model {model_name}, skipping parameter update", "yellow")
                                continue

                            print(f"[DEBUG HANG] Calling update_single_trainer for {model_name}, batch size: {len(batch)}", flush=True)
                            sys.stdout.flush()
                            result = update_single_trainer(model_name, batch, trainer)
                            print(f"[DEBUG HANG] update_single_trainer completed for {model_name}, status: {result.get('status', 'unknown')}", flush=True)
                            
                            # Merge timing metrics
                            for key, value in result["timing"].items():
                                timing_raw[key] = max(timing_raw.get(key, 0), value)
                            
                            # Merge trainer metrics by agent
                            trainer_metrics = result["metrics"]
                            for k, v in trainer_metrics.items():
                                if isinstance(v, (int, float)):
                                    all_trainer_metrics[f"{model_name}/{k}"] = v

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
                print(f"[PRINT DEBUG] Processing batch for model: {model_name}")
                print(f"[PRINT DEBUG] batch is None: {batch is None}")
                if batch is not None:
                    print(f"[PRINT DEBUG] batch.batch is None: {batch.batch is None}")
                    if batch.batch is not None:
                        print(f"[PRINT DEBUG] batch.batch keys: {list(batch.batch.keys())}")
                        if "token_level_scores" in batch.batch:
                            print(f"[PRINT DEBUG] token_level_scores shape: {batch.batch['token_level_scores'].shape if hasattr(batch.batch['token_level_scores'], 'shape') else 'N/A'}")
                        else:
                            print(f"[PRINT DEBUG] WARNING: token_level_scores not in batch.batch!")
                if batch is None or batch.batch is None:
                    print(f"[PRINT DEBUG] ERROR: Cannot compute metrics - batch is None or batch.batch is None!")
                    continue
                try:
                    for metric_name, metric_value in compute_data_metrics(batch=batch, use_critic=any(trainer.use_critic for trainer in self.ppo_trainer_dict.values())).items():
                        metric_name_policy= model_name + "_" + metric_name
                        metrics[metric_name_policy] = metric_value
                except Exception as e:
                    print(f"[METRICS WARNING] compute_data_metrics failed for {model_name}: {e}")

                try:
                    for metric_name, metric_value in compute_timing_metrics(batch=batch, timing_raw=timing_raw).items():
                        metric_name_policy= model_name + "_" + metric_name
                        metrics[metric_name_policy] = metric_value
                except Exception as e:
                    print(f"[METRICS WARNING] compute_timing_metrics failed for {model_name}: {e}")

                # Per-agent reward breakdown
                ntb_keys = list(batch.non_tensor_batch.keys()) if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None else []
                print(f"[METRICS DEBUG] {model_name} non_tensor_batch keys: {ntb_keys}")
                if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None and 'agent_name' in batch.non_tensor_batch and 'reward' in batch.non_tensor_batch:
                    agent_names = batch.non_tensor_batch['agent_name']
                    rewards = batch.non_tensor_batch['reward']
                    from collections import defaultdict
                    agent_rewards = defaultdict(list)
                    for name, r in zip(agent_names, rewards):
                        agent_rewards[name].append(float(r) if r is not None else 0.0)
                    for agent_name, rews in agent_rewards.items():
                        rews_arr = np.array(rews)
                        metrics[f"{model_name}/reward_by_agent/{agent_name}/mean"] = float(np.mean(rews_arr))
                        metrics[f"{model_name}/reward_by_agent/{agent_name}/std"] = float(np.std(rews_arr))
                        metrics[f"{model_name}/reward_by_agent/{agent_name}/count"] = len(rews)
                        metrics[f"{model_name}/reward_by_agent/{agent_name}/nonzero_ratio"] = float(np.count_nonzero(rews_arr) / len(rews_arr))

                # Per-agent response length breakdown (Designer vs WorkflowAgent/Executor)
                if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None and 'agent_name' in batch.non_tensor_batch:
                    try:
                        max_resp_len = batch.batch["responses"].shape[-1]
                        resp_mask = batch.batch["attention_mask"][:, -max_resp_len:].bool()
                        resp_lengths = resp_mask.sum(-1).float().cpu().numpy()
                        prompt_mask_all = batch.batch["attention_mask"][:, :-max_resp_len].bool()
                        prompt_lengths = prompt_mask_all.sum(-1).float().cpu().numpy()
                        agent_names_rl = batch.non_tensor_batch['agent_name']
                        from collections import defaultdict as _dd_rl
                        agent_resp_lens = _dd_rl(list)
                        agent_prompt_lens = _dd_rl(list)
                        for i, name in enumerate(agent_names_rl):
                            agent_resp_lens[name].append(resp_lengths[i])
                            agent_prompt_lens[name].append(prompt_lengths[i])
                        for agent_name, lens in agent_resp_lens.items():
                            arr = np.array(lens)
                            metrics[f"{model_name}/response_length_by_agent/{agent_name}/mean"] = float(np.mean(arr))
                            metrics[f"{model_name}/response_length_by_agent/{agent_name}/max"] = float(np.max(arr))
                            metrics[f"{model_name}/response_length_by_agent/{agent_name}/min"] = float(np.min(arr))
                            metrics[f"{model_name}/response_length_by_agent/{agent_name}/clip_ratio"] = float(np.mean(arr >= max_resp_len))
                        for agent_name, lens in agent_prompt_lens.items():
                            arr = np.array(lens)
                            metrics[f"{model_name}/prompt_length_by_agent/{agent_name}/mean"] = float(np.mean(arr))
                            metrics[f"{model_name}/prompt_length_by_agent/{agent_name}/max"] = float(np.max(arr))
                    except Exception as e:
                        print(f"[METRICS WARNING] per-agent response_length failed for {model_name}: {e}")

                # Reward component breakdown (correctness / delivery / solution)
                if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None:
                    for reward_component in ['correctness_reward', 'delivery_reward', 'solution_reward']:
                        if reward_component in batch.non_tensor_batch:
                            vals = np.array([float(v) for v in batch.non_tensor_batch[reward_component]])
                            metrics[f"{model_name}/reward_breakdown/{reward_component}/mean"] = float(np.mean(vals))
                            metrics[f"{model_name}/reward_breakdown/{reward_component}/nonzero_ratio"] = float(np.count_nonzero(vals) / len(vals)) if len(vals) > 0 else 0.0

                            # Per-agent breakdown
                            if 'agent_name' in batch.non_tensor_batch:
                                agent_names = batch.non_tensor_batch['agent_name']
                                from collections import defaultdict as _dd_rc
                                agent_component = _dd_rc(list)
                                for name, v in zip(agent_names, vals):
                                    agent_component[name].append(float(v))
                                for agent_name, agent_vals in agent_component.items():
                                    arr = np.array(agent_vals)
                                    metrics[f"{model_name}/reward_breakdown/{agent_name}/{reward_component}/mean"] = float(np.mean(arr))
                                    metrics[f"{model_name}/reward_breakdown/{agent_name}/{reward_component}/nonzero_ratio"] = float(np.count_nonzero(arr) / len(arr)) if len(arr) > 0 else 0.0

                # Per-task-type reward breakdown (code vs math)
                if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None and 'task_type' in batch.non_tensor_batch and 'reward' in batch.non_tensor_batch:
                    task_types = batch.non_tensor_batch['task_type']
                    rewards_all = batch.non_tensor_batch['reward']
                    from collections import defaultdict as _dd2
                    type_rewards = _dd2(list)
                    for tt, r in zip(task_types, rewards_all):
                        type_rewards[str(tt)].append(float(r) if r is not None else 0.0)
                    for ttype, rews in type_rewards.items():
                        rews_arr = np.array(rews)
                        metrics[f"{model_name}/reward_by_type/{ttype}/mean"] = float(np.mean(rews_arr))
                        metrics[f"{model_name}/reward_by_type/{ttype}/nonzero_ratio"] = float(np.count_nonzero(rews_arr) / len(rews_arr))
                        metrics[f"{model_name}/reward_by_type/{ttype}/count"] = len(rews)

                # Per-task-type correctness_reward breakdown (code vs math)
                if (hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None
                        and 'task_type' in batch.non_tensor_batch
                        and 'correctness_reward' in batch.non_tensor_batch):
                    task_types_cr = batch.non_tensor_batch['task_type']
                    cr_vals = batch.non_tensor_batch['correctness_reward']
                    from collections import defaultdict as _dd_cr
                    type_cr = _dd_cr(list)
                    for tt, v in zip(task_types_cr, cr_vals):
                        type_cr[str(tt)].append(float(v) if v is not None else 0.0)
                    for ttype, vals in type_cr.items():
                        arr = np.array(vals)
                        metrics[f"{model_name}/correctness_reward_by_type/{ttype}/mean"] = float(np.mean(arr))
                        metrics[f"{model_name}/correctness_reward_by_type/{ttype}/nonzero_ratio"] = float(np.count_nonzero(arr) / len(arr)) if len(arr) > 0 else 0.0

                # Per-task-type response length breakdown (code vs math)
                if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None and 'task_type' in batch.non_tensor_batch:
                    try:
                        max_resp_len = batch.batch["responses"].shape[-1]
                        resp_mask = batch.batch["attention_mask"][:, -max_resp_len:].bool()
                        resp_lengths = resp_mask.sum(-1).float().cpu().numpy()
                        task_types_rl = batch.non_tensor_batch['task_type']
                        from collections import defaultdict as _dd_trl
                        type_resp_lens = _dd_trl(list)
                        for i, tt in enumerate(task_types_rl):
                            type_resp_lens[str(tt)].append(resp_lengths[i])
                        for ttype, lens in type_resp_lens.items():
                            arr = np.array(lens)
                            metrics[f"{model_name}/response_length_by_type/{ttype}/mean"] = float(np.mean(arr))
                            metrics[f"{model_name}/response_length_by_type/{ttype}/max"] = float(np.max(arr))
                            metrics[f"{model_name}/response_length_by_type/{ttype}/clip_ratio"] = float(np.mean(arr >= max_resp_len))
                            metrics[f"{model_name}/response_length_by_type/{ttype}/count"] = len(lens)
                    except Exception as e:
                        print(f"[METRICS WARNING] per-type response_length failed for {model_name}: {e}")

                # Tree design-specific metrics
                design_sample_num = getattr(self.config.training, 'design_sample_num', None)
                execute_sample_num = getattr(self.config.training, 'execute_sample_num', None)
                has_design_idx = 'design_idx' in batch.non_tensor_batch
                if has_design_idx and design_sample_num and execute_sample_num and execute_sample_num > 1:
                    design_indices = batch.non_tensor_batch['design_idx']
                    env_indices = batch.non_tensor_batch['env_idx']
                    agent_idx_arr = batch.non_tensor_batch['agent_idx']
                    uid_arr = batch.non_tensor_batch.get('uid', np.array([]))

                    metrics[f"{model_name}/tree_design/design_sample_num"] = design_sample_num
                    metrics[f"{model_name}/tree_design/execute_sample_num"] = execute_sample_num

                    # Split rewards by Designer vs Executor
                    designer_mask = (agent_idx_arr == 0)
                    executor_mask = (agent_idx_arr == 1)
                    designer_rewards = rewards[designer_mask] if designer_mask.any() else np.array([])
                    executor_rewards = rewards[executor_mask] if executor_mask.any() else np.array([])

                    if len(designer_rewards) > 0:
                        metrics[f"{model_name}/tree_design/designer_reward_mean"] = float(np.mean(designer_rewards))
                        metrics[f"{model_name}/tree_design/designer_reward_std"] = float(np.std(designer_rewards))
                        metrics[f"{model_name}/tree_design/designer_sample_count"] = len(designer_rewards)
                        # Debug: show reward distribution for designer
                        unique_rewards, counts = np.unique(np.round(designer_rewards, 4), return_counts=True)
                        print(f"[METRICS DEBUG] {model_name} designer_rewards distribution: "
                              f"{dict(zip(unique_rewards.tolist(), counts.tolist()))}")
                    if len(executor_rewards) > 0:
                        metrics[f"{model_name}/tree_design/executor_reward_mean"] = float(np.mean(executor_rewards))
                        metrics[f"{model_name}/tree_design/executor_reward_std"] = float(np.std(executor_rewards))
                        metrics[f"{model_name}/tree_design/executor_sample_count"] = len(executor_rewards)

                    # Per-design reward analysis (executor side: intra vs inter design variance)
                    from collections import defaultdict as _dd
                    design_reward_groups = _dd(list)
                    for i in range(len(batch)):
                        if agent_idx_arr[i] == 1:  # Executor
                            d_key = (int(env_indices[i]), int(design_indices[i]))
                            design_reward_groups[d_key].append(float(rewards[i]) if rewards[i] is not None else 0.0)

                    if design_reward_groups:
                        # Intra-design std: average std within each design's executions
                        intra_stds = [float(np.std(rews)) for rews in design_reward_groups.values() if len(rews) > 1]
                        if intra_stds:
                            metrics[f"{model_name}/tree_design/intra_design_reward_std"] = float(np.mean(intra_stds))

                        # Inter-design std: std of per-design mean rewards
                        design_means = [float(np.mean(rews)) for rews in design_reward_groups.values()]
                        if len(design_means) > 1:
                            metrics[f"{model_name}/tree_design/inter_design_reward_std"] = float(np.std(design_means))
                            metrics[f"{model_name}/tree_design/inter_design_reward_mean"] = float(np.mean(design_means))

                    # GRPO group counts from UIDs
                    if len(uid_arr) > 0:
                        designer_uids = set(uid_arr[designer_mask]) if designer_mask.any() else set()
                        executor_uids = set(uid_arr[executor_mask]) if executor_mask.any() else set()
                        metrics[f"{model_name}/tree_design/num_grpo_groups_designer"] = len(designer_uids)
                        metrics[f"{model_name}/tree_design/num_grpo_groups_executor"] = len(executor_uids)

            # Per problem_type reward metrics (for mixed training)
            if hasattr(self, 'agent_execution_engine') and hasattr(self.agent_execution_engine, 'envs'):
                envs = self.agent_execution_engine.envs
                for ptype in ["math", "code"]:
                    # Use e.env_idx (the problem-level index set in mixed_env.py) rather than
                    # enumerate position, because DataProto["env_idx"] stores the problem-level
                    # index (0..n_problems-1), not the rollout-list position (0..n_problems*samples-1).
                    typed_env_idxs = set(
                        getattr(e, 'env_idx', None) for e in envs
                        if getattr(e, 'problem_type', None) == ptype
                    )
                    typed_env_idxs.discard(None)
                    if typed_env_idxs and 'env_idx' in batch.non_tensor_batch:
                        env_idx_arr = batch.non_tensor_batch['env_idx']
                        typed_mask = np.isin(env_idx_arr, list(typed_env_idxs))
                        if typed_mask.any():
                            typed_rewards = rewards[typed_mask]
                            metrics[f"{model_name}/reward_by_type/{ptype}_mean"] = float(np.mean(typed_rewards))
                            metrics[f"{model_name}/reward_by_type/{ptype}_nonzero_ratio"] = float(
                                np.count_nonzero(typed_rewards) / len(typed_rewards)
                            )

            # Standard data and timing metrics
            #metrics.update(compute_data_metrics(batch=first_batch, use_critic=any(trainer.use_critic for trainer in self.ppo_trainer_dict.values())))
            #metrics.update(compute_timing_metrics(batch=first_batch, timing_raw=timing_raw))

            # Add training step metrics
            metrics.update({
                "training/global_step": self.global_steps,
                
            })

            

            # Run validation at every val_freq steps (skip step 0 if val_before_train is False)
            val_before_train = self.config.get("val_before_train", True)
            # Also check per-trainer config
            for trainer in self.ppo_trainer_dict.values():
                vbt = getattr(trainer.config, 'trainer', None)
                if vbt is not None:
                    val_before_train = vbt.get("val_before_train", val_before_train)
                    break
            should_validate = (self.global_steps % self.config.training.val_freq == 0)
            # Step 0: skip post-training validation since val_before_train already ran it
            if self.global_steps == 0:
                should_validate = False
            # Skip if we already ran val at the start of this iteration (resume-val)
            if _ran_resume_val:
                should_validate = False
            if should_validate:
                # Free training batch BEFORE validation to reclaim GPU memory
                # for KV cache re-allocation during wake_up()
                # batch_per_trainer holds 984 × 8192 tensors (~huge GPU memory)
                del batch_per_trainer
                batch_per_trainer = {}
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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

            # Periodic checkpoint saving based on save_freq
            save_freq = getattr(self.config.training, 'save_freq', -1)
            if save_freq > 0 and self.global_steps % save_freq == 0:
                colorful_print(f"Periodic checkpoint save at step {self.global_steps}", "cyan")
                save_base = self.config.specialization != "lora"
                for trainer in self.ppo_trainer_dict.values():
                    trainer._save_checkpoint(save_base=save_base)
                import gc
                gc.collect()

            # Debug: show all metric keys being logged
            reward_keys = [k for k in metrics.keys() if 'reward_by_agent' in k or 'tree_design' in k or 'by_type' in k]
            print(f"[METRICS DEBUG] Logging {len(metrics)} metrics total, reward/tree/by_type keys: {reward_keys}")

            try:
                logger.log(data=metrics, step=self.global_steps)
            except Exception as e:
                pprint(f"Warning: Failed to log metrics to logger: {type(e).__name__}: {e}")
                pprint(f"Metrics that failed to log: {list(metrics.keys())}")

            # End-of-step memory cleanup: free training batch and force GC
            # Critical for tree design mode to prevent gradual OOM
            del batch_per_trainer
            batch_per_trainer = {}
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
        # Support multiple benchmarks: env.benchmark can be a single string or a list
        benchmark_cfg = getattr(self.config.env, 'benchmark', 'AIME24')
        from omegaconf import ListConfig
        if isinstance(benchmark_cfg, (list, tuple, ListConfig)):
            benchmarks = list(benchmark_cfg)
        else:
            benchmarks = [benchmark_cfg]

        all_metrics = {}
        best_env_success_rate = 0.0

        for benchmark in benchmarks:
            prefix = f"validation/{benchmark}" if len(benchmarks) > 1 else "validation"
            # Temporarily override the benchmark in config
            original_benchmark = self.config.env.benchmark
            self.config.env.benchmark = benchmark

            metrics, env_success_rate = self._validate_single_benchmark(global_steps, prefix)
            all_metrics.update(metrics)
            best_env_success_rate = max(best_env_success_rate, env_success_rate)

            # Restore original benchmark
            self.config.env.benchmark = original_benchmark

            # Cleanup between benchmarks to prevent OOM
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save checkpoint based on best env_success_rate across all benchmarks
        if global_steps > 0:
            self._save_best_checkpoint(best_env_success_rate)

        return all_metrics

    def _validate_single_benchmark(self, global_steps, prefix="validation"):
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
            try:
                rollout_engine.sleep()
            except Exception as e:
                logger.warning(f"[WARNING] rollout_engine.sleep() failed for {model_name} during validation: {e}")

        for model_name in self.ppo_trainer_dict.keys():
            if batch_per_trainer[model_name].batch is None:
                batch_per_trainer[model_name] = gen_batch_output_per_policy[model_name]
            else:
                batch_per_trainer[model_name] = DataProto.concat([
                    batch_per_trainer[model_name],
                    gen_batch_output_per_policy[model_name]
                ])

        # Free generation output after copying to batch_per_trainer
        del gen_batch_output_per_policy
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

            validation_metrics[f"{prefix}/agent_{agent_name}/success_rate"] = success_rate
            validation_metrics[f"{prefix}/agent_{agent_name}/avg_turns"] = avg_turns

        if success_rollout_rate_dict:
            success_rates = list(success_rollout_rate_dict.values())
            avg_turns_list = list(success_turn_ave_dict.values())

            validation_metrics[f"{prefix}/average/success_rate"] = sum(success_rates) / len(success_rates)
            validation_metrics[f"{prefix}/average/avg_turns"] = sum(avg_turns_list) / len(avg_turns_list)

        validation_metrics[f"{prefix}/env_state_success_rate"] = env_success_rate

        # Per problem_type validation metrics (backward compatible)
        for ptype in ["math", "code"]:
            typed_envs = [e for e in self.agent_execution_engine.envs
                          if getattr(e, 'problem_type', None) == ptype]
            if typed_envs:
                typed_success = sum(1 for e in typed_envs if getattr(e, 'success', False))
                validation_metrics[f"{prefix}/{ptype}_success_rate"] = typed_success / len(typed_envs)

        # Free validation batch data before returning
        del batch_per_trainer
        import gc
        gc.collect()

        return validation_metrics, env_success_rate

    def _pad_dataproto_to_world_size(self, batch, world_sizes):
        batch, pad_size = pad_dataproto_to_divisor(batch, world_sizes)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        return batch
    
    def _assign_consistent_uids(self, data_proto, filter_ratio=0.0, mode="mean", sample_num=1, rollout_mode="tree",
                                design_sample_num=None, execute_sample_num=None, executor_group_mode=None):
        """
        Assign consistent UIDs to data and optionally filter based on rewards.

        Args:
            data_proto: DataProto object containing trajectory data
            filter_ratio: Ratio of samples to filter (0.0 to 1.0)
            mode: Filtering mode - "mean", "std", "dapo", or "uid"
            sample_num: Number of samples per environment
            rollout_mode: Rollout mode ("tree" or "no_tree")
            design_sample_num: Number of designs per problem (tree design mode)
            execute_sample_num: Number of executions per design (tree design mode)

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
        
        # Detect tree design mode from design_idx in non_tensor_batch
        has_design_idx = "design_idx" in non_tensor_batch
        use_tree_grouping = (has_design_idx and execute_sample_num is not None and execute_sample_num > 1)

        print(f"[DEBUG UID] Input: len={len(data_proto)}, rollout_mode={rollout_mode}, sample_num={sample_num}, "
              f"use_tree_grouping={use_tree_grouping}, design_sample_num={design_sample_num}, execute_sample_num={execute_sample_num}")
        print(f"[DEBUG UID] Rewards: len={len(rewards)}, mean={np.mean(rewards) if len(rewards) > 0 else 'N/A'}, nonzero={np.count_nonzero(rewards) if len(rewards) > 0 else 0}")

        design_indices = non_tensor_batch.get("design_idx", None)

        uids = []
        for i in range(len(data_proto)):
            if rollout_mode == "no_tree":
                key = (rollout_indices[i],)
            elif use_tree_grouping:
                # In tree design mode, env_idx IS the problem index (0..n_problems-1).
                # Do NOT divide by sample_num — that formula is for rollout indices.
                base_env = int(rollout_indices[i])
                if agent_indices[i] == 0:
                    # Designer: same problem's all designs share one GRPO group
                    key = (base_env, 0, 0)
                else:
                    # executor_group_mode controls grouping:
                    #   "question" = all executors per problem share one GRPO group (group size = N * M)
                    #   "design"   = executors from the same design share one GRPO group (group size = M)
                    #   None/auto  = auto: question if execute_sample_num <= 2, else design
                    if executor_group_mode == "question":
                        key = (base_env, 0, 1)
                    elif executor_group_mode == "design":
                        key = (base_env, int(design_indices[i]), 1)
                    elif execute_sample_num is not None and execute_sample_num <= 2:
                        key = (base_env, 0, 1)
                    else:
                        key = (base_env, int(design_indices[i]), 1)
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
            
            print(f"[DEBUG UID] no_tree mode: removing {len(sample_to_remove)} samples out of {len(data_proto)}")
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
            
            print(f"[DEBUG UID] Output: total_samples={len(all_rewards)}, mean_reward={np.mean(all_rewards):.4f}, remain_samples={len(data_proto)}, removed={len(sample_to_remove)}")
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
