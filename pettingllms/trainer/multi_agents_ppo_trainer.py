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
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
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
from verl.utils.torch_functional import pad_sequence_to_length
from typing import Dict
from pettingllms.router.router import Router
from verl.utils.profiler.performance import _timer
from verl.trainer.ppo.reward import load_reward_manager

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
        self.ppo_trainer_dict = {}
        self.tokenizer_dict = tokenizer_dict
        
        
        if hasattr(config, 'models'):
            for i, (model_key, model_config) in enumerate(config.models.items()):
                model_name = model_config.name
                print(f"model_config: {model_config}")
                if hasattr(model_config, 'ppo_trainer_config'):
                    ppo_config = model_config.ppo_trainer_config
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
                    # Prefer VERL's canonical PPO trainer config as the base to ensure required `_target_` fields andbox_fusion = {"url": None, "max_concurrent": 64, "memory_limit_mb": 1024}

                    ppo_trainer = RayPPOTrainer(
                        config=ppo_config,
                        tokenizer=model_tokenizer,
                        role_worker_mapping=role_worker_mapping,
                        resource_pool_manager=resource_pool_manager[i],
                        ray_worker_group_cls=ray_worker_group_cls,
                        
                    )
                    self.ppo_trainer_dict[model_name] = ppo_trainer
                    self.tokenizer_dict[model_name] = model_tokenizer
                    colorful_print(f"PPO trainer created for model: {model_name}", "green")
        colorful_print(f"the number of ppo_trainer_dict: {len(self.ppo_trainer_dict)}", "green")
        
        colorful_print(f"Number of PPO trainers: {len(self.ppo_trainer_dict)}", "cyan")
        colorful_print(f"Number of agent mappings: {len(self.agent_policy_mapping)}", "cyan")
        
 

    def init_multi_agent_sys_execution_engine(self):
        # Get the rollout engines and tokenizers from the trainers
        rollout_engine_dict = {}
        tokenizer_dict = {}
        router_dict = {}
        for model_name, trainer in self.ppo_trainer_dict.items():
            rollout_engine_dict[model_name] = trainer.actor_rollout_wg
            tokenizer_dict[model_name] = trainer.tokenizer
            server_addresses = getattr(trainer.actor_rollout_wg, "server_addresses", [])
            # Construct an independent Router for each model
            router_dict[model_name] = Router(config=trainer.config, tokenizer=trainer.tokenizer, addresses=server_addresses)
        
        self.agent_execution_engine = MultiAgentsExecutionEngine(
            config=self.config,
            tokenizer_dict=tokenizer_dict,
            processor_dict=self.processor_dict,
            router_dict=router_dict,
            agent_policy_mapping=self.agent_policy_mapping,
        )

    def init_workers(self):
        """Initialize workers for all PPO trainers."""
        colorful_print("Initializing workers for all PPO trainers...", "cyan")
        for model_name, trainer in self.ppo_trainer_dict.items():
            colorful_print(f"Initializing workers for trainer: {model_name}", "green")
            trainer.init_workers()
        colorful_print("All workers initialized successfully", "green")

    def _update_parameters(self, batch, gen_batch_output, ppo_trainer, timing_raw):
        #TODO: uid
        batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
        #TODO: repeat to align with repeated responses in rollout
        batch = batch.repeat(repeat_times=ppo_trainer.config.actor_rollout_ref.rollout.n, interleave=True)
        batch = batch.union(gen_batch_output)

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
        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask_batch[:, prompt_length:].sum(dim=-1)

        for i in range(len(batch.batch["responses"])):
            valid_response_length = valid_response_length_sequences[i]-1
            if valid_response_length >= 0 and valid_response_length < reward_tensor.shape[1]:
                reward_tensor[i, valid_response_length] = batch.non_tensor_batch["reward"][i]

        batch.batch["token_level_scores"] = reward_tensor
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]



        # recompute old_log_probs
        with _timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = ppo_trainer.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)


        if ppo_trainer.use_reference_policy:
            # compute reference log_prob
            with _timer("ref", timing_raw, color="olive"):
                if not ppo_trainer.ref_in_actor:
                    ref_log_prob = ppo_trainer.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = ppo_trainer.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

            # compute values
        if ppo_trainer.use_critic:
            with _timer("values", timing_raw, color="cyan"):
                values = ppo_trainer.critic_wg.compute_values(batch)
                batch = batch.union(values)

        with _timer("adv", timing_raw, color="brown"):

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
            with _timer("update_critic", timing_raw, color="pink"):
                critic_output = ppo_trainer.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            batch.meta_info["metrics"].update(critic_output_metrics)

        # implement critic warmup
        if ppo_trainer.config.trainer.critic_warmup <= ppo_trainer.global_steps:
            # update actor
            with _timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = ppo_trainer.config.actor_rollout_ref.rollout.multi_turn.enable
                actor_output = ppo_trainer.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            batch.meta_info["metrics"].update(actor_output_metrics)

        # Log rollout generations if enabled
        rollout_data_dir = ppo_trainer.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            with _timer("dump_rollout_generations", timing_raw, color="green"):
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

    
    
    def __generate_agent_trajectories_async(self, timing_raw=None, meta_info=None, mode="Token"):
        """
        Generates agent trajectories asynchronously using the agent execution engine.

        This method runs the asynchronous `trajectory_generator` in a
        separate thread and yields the results synchronously through a queue.
        This allows the main training loop (which might be synchronous) to consume
        asynchronously generated trajectories.

        Args:
            timing_raw (dict, optional): Dictionary to store timing information. Defaults to {}.
            meta_info (dict, optional): Additional metadata for the generation process. Defaults to None.

        Yields:
            Any: Items generated by the `trajectory_generator`, typically
                 representing parts or results of agent trajectories in token format.
        """
        if timing_raw is None:
            timing_raw = {}
        queue = Queue()

        def runner():
            async def consume():
                # Iterate all rollout_idx and generate one by one
                num_envs = len(self.agent_execution_engine.env_list) if hasattr(self.agent_execution_engine, "env_list") else len(self.agent_execution_engine.envs)
                for rollout_idx in range(num_envs):
                    item = await self.agent_execution_engine.generate_single_rollout(rollout_idx=rollout_idx, timing_raw=timing_raw, meta_info=meta_info)
                    queue.put(item)
                queue.put(None)  # sentinel to signal done

            asyncio.run(consume())

        Thread(target=runner, daemon=True).start()
        while True:
            item = queue.get()
            if item is None:
                break
            yield item

    def _filter_trajectory(self, trajectories):
        # TODO:Filter the trajectory
        return trajectories

    def generate_and_filter_agent_trajectories(self, timing_raw=None, meta_info=None):
        """
        Generates agent trajectories by interacting with the environment. Does not close or reset the environment afterwards.

        Returns:
            DataProto: Representation of the last step of agent's trajectories.
            Dict[str:List[DataProto]]: Index of the trajectory to the rest of the steps from the trajectory.
        """
        if timing_raw is None:
            timing_raw = {}
        
        with _timer("collect_trajectory", timing_raw):
            # Accumulate data for each policy/model
            trajectories: dict[str, DataProto] = {}
            gen_seq_generator = self.__generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=meta_info, mode="Step")
            for _, one_rollout_dict in enumerate(gen_seq_generator):
                # one_rollout_dict: dict[policy_name, DataProto]
                for key, dpr in one_rollout_dict.items():
                    if key not in trajectories:
                        trajectories[key] = DataProto()
                    trajectories[key] = trajectories[key].union(dpr)
            

        with _timer("filter_trajectory", timing_raw):
            # TODO:Filter the trajectory
            final_gen_batch_output = self._filter_trajectory(trajectories)
        return final_gen_batch_output
    




    def fit(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        

        # load checkpoint before doing anything
        for model_name, trainer in self.ppo_trainer_dict.items():
            colorful_print(f"Loading checkpoint for trainer {model_name}", "cyan")
            trainer._load_checkpoint()
            start_time = time.time()
            if trainer.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
                val_metrics = trainer._validate()
                pprint(f"Initial validation metrics: {val_metrics}")
                logger.log(data=val_metrics, step=self.global_steps)
            print(f"Time taken to validate agent_{model_name}: {time.time() - start_time}")



        # perform validation before training
       
        
        # Use the minimum total training steps among sub-trainers as the global total
        if len(self.ppo_trainer_dict) > 0:
            self.total_training_steps = min(tr.total_training_steps for tr in self.ppo_trainer_dict.values())
        else:
            self.total_training_steps = 0
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
       
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            pprint(f"epoch {epoch}, step {self.global_steps} started")
            #for batch_dict in self.train_dataloader:
                #batch: DataProto = DataProto.from_single_dict(batch_dict)
            meta_info = {
                    "agent_rollout": True,  # no need to generate multiple ones since environment is repeated already
                    "epoch": epoch,
                    "global_step": self.global_steps,
                }
            batch_size = self.config.data.gen_batch_size
            # Get batch_num from traindataset
            
            #init dataproto with batch_size
            batch_per_trainer: Dict[str,DataProto]={}
            
            # Load data for each trainer
            for model_name in self.ppo_trainer_dict.keys():
                # For now, create a placeholder batch
                batch_per_trainer[model_name] = DataProto(batch_size=batch_size)  # Placeholder
                
            metrics = {}
            timing_raw = {}

                # Process each trainer's batch

            with _timer("step", timing_raw):
                

                
                # Directly call generate_agent_trajectories_async for simplified logic
                with _timer("collect_trajectory", timing_raw):
                    gen_batch_output_per_policy = self.generate_and_filter_agent_trajectories(timing_raw=timing_raw, meta_info=meta_info)
                
                # Sort trajectories by their idx, to ensure they are in order.
    
                with _timer("update_parameters", timing_raw):
                    for model_name, trainer in self.ppo_trainer_dict.items():
                        # Update parameters for the corresponding policy/model
                        if model_name not in gen_batch_output_per_policy:
                            # Skip if this model has not generated data
                            continue
                        self._update_parameters(
                            batch_per_trainer[model_name],
                            gen_batch_output_per_policy[model_name],
                            trainer,
                            timing_raw,
                        )
            
                # TODO:validate
                with _timer("testing", timing_raw):
                    # Use the first trainer's validation metrics (or aggregate)
                    first_trainer = next(iter(self.ppo_trainer_dict.values()))
                    val_metrics: dict = first_trainer._validate()
                metrics.update(val_metrics)

                

                #save checkpoint done
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer("save_checkpoint", timing_raw):
                        for model_name, trainer in self.ppo_trainer_dict.items():
                            colorful_print(f"Saving checkpoint for trainer {model_name}", "cyan")
                            trainer._save_checkpoint()

            # TODO: collect metrics
            # Use the first trainer's batch for metrics calculation
            first_model_name = list(self.ppo_trainer_dict.keys())[0]
            first_batch = batch_per_trainer[first_model_name]
            metrics.update(compute_data_metrics(batch=first_batch, use_critic=any(trainer.use_critic for trainer in self.ppo_trainer_dict)))
            metrics.update(compute_timing_metrics(batch=first_batch, timing_raw=timing_raw))

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            self.global_steps += 1

            # Check if any trainer has reached its total training steps
            if any(self.global_steps >= trainer.total_training_steps for trainer in self.ppo_trainer_dict.values()):
                # perform validation after training
                first_trainer = next(iter(self.ppo_trainer_dict.values()))
                if first_trainer.val_reward_fn is not None:
                    val_metrics = first_trainer._validate()
                    pprint(f"Final validation metrics: {val_metrics}")
                    logger.log(data=val_metrics, step=self.global_steps)
                return

    def _validate(self):
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        for test_data in self.val_dataloader:
            test_batch_per_trainer = {}
            for model_name in self.ppo_trainer_dict.keys():
                test_batch_per_trainer[model_name] = DataProto.from_single_dict(test_data)
                test_batch_per_trainer[model_name].non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch_per_trainer[model_name].batch))], dtype=object)
                n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
                test_batch_per_trainer[model_name] = test_batch_per_trainer[model_name].repeat(repeat_times=n_val_samples, interleave=True)
                test_batch_per_trainer[model_name].pop(["input_ids", "attention_mask", "position_ids"])  # these are not needed for environment based interaction
                test_batch_per_trainer[model_name].meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": False,
                    "validate": True,
                    "agent_rollout": True,
                }
            

            
            # Directly call generate_agent_trajectories_async for simplified logic
            timing_raw = {}
            with _timer("collect_trajectory", timing_raw):
                steps = []
                # Use the first model's batch for trajectory generation
                first_model_name = list(self.ppo_trainer_dict.keys())[0]
                gen_seq_generator = self.generate_agent_trajectories_async(timing_raw=timing_raw, meta_info=test_batch_per_trainer[first_model_name].meta_info, mode="Step")
                for _, trajectory in enumerate(gen_seq_generator):
                    steps.append(trajectory)
            
            # Sort trajectories by their idx, to ensure they are in order.
            steps.sort(key=lambda x: x["idx"])

            with _timer("transform_trajectory", timing_raw):
                # Transform the raw trajectories into DataProto format.
                test_output_gen_batch = self._transform_agent_steps(steps, uids=test_batch_per_trainer[first_model_name].non_tensor_batch["uid"])
            
            # for validation, we only need the last step
            is_last_step = test_output_gen_batch.non_tensor_batch["is_last_step"]
            last_step_indices = np.where(is_last_step == True)[0]
            test_output_gen_batch = test_output_gen_batch.select_idxs(last_step_indices)  # This batch only has last steps
            
            # Union the generated output with all trainer batches
            for model_name in self.ppo_trainer_dict.keys():
                test_batch_per_trainer[model_name] = test_batch_per_trainer[model_name].union(test_output_gen_batch)

            # Use the first trainer's batch for reward calculation (should be the same for all trainers in validation)
            reward_tensor = test_batch_per_trainer[first_model_name].batch["token_level_scores"]

            rewards_lst.append(reward_tensor.sum(-1).cpu())
            data_source_lst.append(test_batch_per_trainer[first_model_name].non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            uid_lst.append(test_batch_per_trainer[first_model_name].non_tensor_batch["uid"])

        reward_tensor = torch.cat(rewards_lst, dim=0)  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}

        # to group for pass@k
        uid_tensor = np.concatenate(uid_lst, axis=0)
        data_source_uid_pass_rates = {}  # data source to {uid: pass or not}

        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]

            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

            # pass@k
            if data_source not in data_source_uid_pass_rates:
                data_source_uid_pass_rates[data_source] = {}

            uid = uid_tensor[i]
            if uid not in data_source_uid_pass_rates[data_source]:
                data_source_uid_pass_rates[data_source][uid] = 0  # default to not pass
            # take highest score
            data_source_uid_pass_rates[data_source][uid] = max(data_source_uid_pass_rates[data_source][uid], reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            # clip rewards to be between 0 and 1
            rewards_array = np.array(rewards)
            rewards_array = np.clip(rewards_array, 0, 1)
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards_array)

        for data_source, pass_rates in data_source_uid_pass_rates.items():
            pass_k_lst = []
            for uid, pass_score in pass_rates.items():
                pass_k_lst.append(pass_score >= 1)  # assuming 1 means passed
            metric_dict[f"val/test_score/pass@k/{data_source}"] = np.mean(pass_k_lst)

        return metric_dict

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

  
    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        # Collect world sizes from all trainers
        for trainer in self.ppo_trainer_dict.values():
            if hasattr(trainer, 'use_critic') and trainer.use_critic and hasattr(trainer, 'critic_wg') and trainer.critic_wg.world_size != 0:
                world_sizes.append(trainer.critic_wg.world_size)
            if hasattr(trainer, 'use_reference_policy') and trainer.use_reference_policy and hasattr(trainer, 'ref_policy_wg') and trainer.ref_policy_wg.world_size != 0:
                world_sizes.append(trainer.ref_policy_wg.world_size)
            if hasattr(trainer, 'use_rm') and trainer.use_rm and hasattr(trainer, 'rm_wg') and trainer.rm_wg.world_size != 0:
                world_sizes.append(trainer.rm_wg.world_size)
            if hasattr(trainer, 'actor_rollout_wg') and trainer.actor_rollout_wg.world_size != 0:
                world_sizes.append(trainer.actor_rollout_wg.world_size)
        
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True

        return batch
