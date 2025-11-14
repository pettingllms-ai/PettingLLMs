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
from pettingllms.trainer.multiagentssys_register import AGENT_CLASS_MAPPING, ENV_CLASS_MAPPING, ENV_BATCH_CLASS_MAPPING, ENV_WORKER_CLASS_MAPPING
from functools import partial
import multiprocessing
from pettingllms.utils.performance import create_timer
import copy
from pettingllms.trainer.async_generate import convert_prompt_to_dpr, llm_async_generate
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.multi_agent_env.code.code_worker import get_ray_docker_worker_cls


logger = logging.getLogger(__name__)




class MultiAgentsExecutionEngine:
    def _load_config_parameters(self):
        self.max_prompt_length = getattr(self.config.training, 'max_prompt_length', 1024)
        self.max_response_length = getattr(self.config.training, 'max_response_length', 1024)
        self.turn_order = self.config.multi_agent_interaction.turn_order
        self.num_interacting_agents = self.config.multi_agent_interaction.num_interacting_agents
        self.parallel = getattr(self.config.multi_agent_interaction, 'parallel', False)
        self.generate_timeout = getattr(self.config.training, 'generate_timeout', 300.0)
        if self.num_interacting_agents != len(self.turn_order):
            raise ValueError("num_interacting_agents must be equal to the length of turn_order")
          
        
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
        self.experiment_name = self.config.training.experiment_name
        self.env_name = env_name
        self.env_class = ENV_CLASS_MAPPING[env_name]
        self.agent_class_list = [AGENT_CLASS_MAPPING[agent_name] for agent_name in self.turn_order]
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
        if self.env_name in ENV_WORKER_CLASS_MAPPING:
                _worker_factory_or_cls = ENV_WORKER_CLASS_MAPPING[self.env_name]
                try:
                    # Get num_workers first to pass to worker factory for CPU calculation
                    num_workers = self.config.training.get("num_workers", 180)
                    # Try to call factory with num_workers parameter if it's callable
                    if callable(_worker_factory_or_cls):
                        try:
                            RayDockerWorker = _worker_factory_or_cls(num_workers=num_workers)
                        except TypeError:
                            # Fallback if factory doesn't accept num_workers
                            RayDockerWorker = _worker_factory_or_cls()
                    else:
                        RayDockerWorker = _worker_factory_or_cls
                except Exception as e:
                    print(f"Failed to create RayDockerWorker from mapping for env '{self.env_name}': {e}")
                    RayDockerWorker = None
        else:
            num_workers = self.config.training.get("num_workers", 180)
            RayDockerWorker = get_ray_docker_worker_cls(num_workers=num_workers)
        print("begin to create Ray docker workers")
        if RayDockerWorker is not None and hasattr(RayDockerWorker, "remote"):
            num_workers = self.config.training.get("num_workers", 180)
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
        for agent_name in self.turn_order:
            agent_config = self.agent_config_dict.get(agent_name, None)
            # Read enable_thinking from agent config, default to False
            enable_thinking = getattr(agent_config, 'enable_thinking', False) if agent_config else False
            self.agent_enable_thinking[agent_name] = enable_thinking
            print(f"Agent '{agent_name}' enable_thinking: {enable_thinking}")
        
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
            
        self.agent_groups_list = []
        for rollout_idx in range(len(self.envs)):
            self.agent_groups_list_per_rollout = []
            for agent_idx, agent_name in enumerate(self.turn_order):
                agent_class = self.agent_class_list[agent_idx]

                agent_init_params = {'env_idx': rollout_idx, 'agent_sample_idx': rollout_idx, 'rollout_idx': rollout_idx}
                agent_init_params['benchmark'] = getattr(self.config.env, 'benchmark', 'AIME24') if hasattr(self.config, 'env') else 'AIME24'

                self.agent_groups_list_per_rollout.append(agent_class(**agent_init_params))
            self.agent_groups_list.append(self.agent_groups_list_per_rollout)
        
    
                   
            
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
                self.mode, env_idx, rollout_idx, "turn_start",
                f"Starting turn {turn_idx + 1}",
                {"turn_idx": turn_idx + 1}
            )
            
            agent_outputs = []
            
            if self.parallel:
                print(f"[Engine][PARALLEL_MODE] rollout_idx={rollout_idx} turn={turn_idx} - Parallel LLM requests for all agents")
                
                async def generate_agent_response(agent_idx, agent_name):
                    current_agent = agent_group[agent_idx]
                    current_agent.update_from_env(turn_idx, env)
                    prompt = current_agent.current_prompt
                    policy_name = self.agent_policy_mapping.get(agent_name)
                    agent_enable_thinking = self.agent_enable_thinking.get(agent_name, False)
                    
                    format_prompt = convert_prompt_to_dpr(
                        self.tokenizer_dict[policy_name],
                        self.processor_dict.get(policy_name),
                        prompt,
                        self.max_prompt_length,
                        multi_modal=False,
                        enable_thinking=agent_enable_thinking
                    )
                    
                    if format_prompt is None:
                        return None
                    
                    ppo_trainer_config = self.ppo_trainer_config_dict.get(policy_name, None)
                    model_path = ppo_trainer_config.actor_rollout_ref.model.path
                    # Match vLLM's model registration logic
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
                        if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                            lora_id = self.agent_lora_mapping[agent_name]
                        
                        agent_config = self.agent_config_dict.get(agent_name, None)
                        agent_sample_num = getattr(agent_config, 'sample_num', 1) if agent_config else 1
                        
                        print(f"[Engine][PARALLEL_GEN] rollout_idx={rollout_idx} agent={agent_name} calling LLM")
                        output_dpr, response = await llm_async_generate(
                            rollout_idx=rollout_idx,
                            turn_idx=turn_idx,
                            agent_idx=agent_idx,
                            prompt_dpr=format_prompt,
                            ppo_trainer_config=ppo_trainer_config,
                            address=_address,
                            model_name=model_name,
                            tokenizer=self.tokenizer_dict[policy_name],
                            enable_thinking=agent_enable_thinking,
                            image_data=None,
                            application_id=str(uuid.uuid4()),
                            env_idx=rollout_idx // self.sample_num,
                            policy_name=policy_name,
                            timeout=self.generate_timeout,
                            mode=self.mode,
                            lora_id=lora_id,
                            agent_config=agent_config,
                        )
                    except Exception as e:
                        self.multi_logger.log_env_agent_info(
                            self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                            f"Failed to generate response: {e}",
                            {"error": str(e), "traceback": traceback.format_exc()}
                        )
                        output_dpr = None
                        response = ""
                    
                    if response is None:
                        response = ""
                    
                    return {
                        'agent_idx': agent_idx,
                        'agent_name': agent_name,
                        'current_agent': current_agent,
                        'output_dpr': output_dpr,
                        'response': response,
                        'prompt': prompt,
                        'policy_name': policy_name
                    }
                
                generation_tasks = [
                    generate_agent_response(agent_idx, agent_name)
                    for agent_idx, agent_name in enumerate(self.turn_order)
                ]
                agent_outputs = await asyncio.gather(*generation_tasks)
                agent_outputs = [out for out in agent_outputs if out is not None]
                
                for agent_output in agent_outputs:
                    agent_idx = agent_output['agent_idx']
                    agent_name = agent_output['agent_name']
                    current_agent = agent_output['current_agent']
                    response = agent_output['response']
                    
                    current_agent.update_from_model(response)

                    try:
                        # Deterministic worker assignment based on rollout_idx
                        # This ensures consistent worker-task mapping for file path isolation
                        env_worker_id = rollout_idx % self.num_workers
                        env_worker = self.env_workers[env_worker_id]

                        # Store worker_id and GPU group in env_data for task folder path generation
                        if hasattr(self.envs[rollout_idx], 'state'):
                            self.envs[rollout_idx].state.assigned_worker_id = env_worker_id
                            self.envs[rollout_idx].state.gpu_group_id = self.gpu_group_id

                        await asyncio.wait_for(
                            current_agent.step(self.envs[rollout_idx], env_worker=env_worker),
                            timeout=self.step_timeout
                        )
                    except asyncio.TimeoutError:
                        self.multi_logger.log_env_agent_info(
                            self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                            f"Environment step timed out after {self.step_timeout}s",
                            {"error": "timeout", "timeout_seconds": self.step_timeout}
                        )
                    
                    # Capture env state immediately after this agent's step
                    env_state_snapshot = env.state.to_dict_compact(agent_name=agent_name) if hasattr(env.state, 'to_dict_compact') else None
                    
                    agent_output['env_state_snapshot'] = env_state_snapshot
            else:
                print(f"[Engine][SEQUENTIAL_MODE] rollout_idx={rollout_idx} turn={turn_idx} - Sequential agent execution")
                
                for agent_idx, agent_name in enumerate(self.turn_order):
                    current_agent = agent_group[agent_idx]
                    current_agent.update_from_env(turn_idx, env)
                    prompt = current_agent.current_prompt
                    policy_name = self.agent_policy_mapping.get(agent_name)
                    agent_enable_thinking = self.agent_enable_thinking.get(agent_name, False)
                    
                    format_prompt = convert_prompt_to_dpr(
                        self.tokenizer_dict[policy_name],
                        self.processor_dict.get(policy_name),
                        prompt,
                        self.max_prompt_length,
                        multi_modal=False,
                        enable_thinking=agent_enable_thinking
                    )
                    
                    if format_prompt is None:
                        return None
                    
                    ppo_trainer_config = self.ppo_trainer_config_dict.get(policy_name, None)
                    model_path = ppo_trainer_config.actor_rollout_ref.model.path
                    # Match vLLM's model registration logic
                    if "checkpoint" in str(model_path):
                        model_name = str(model_path)
                    else:
                        model_name = "/".join(str(model_path).split("/")[-2:])
                    print(f"[Engine][MODEL_NAME] rollout_idx={rollout_idx} agent={agent_name} model_path={model_path} model_name={model_name}")
                    
                    output_dpr = None
                    response = None
                    
                    try:
                        _addresses = self.server_address_dict.get(policy_name)
                        if isinstance(_addresses, (list, tuple)):
                            _address = random.choice(_addresses) if len(_addresses) > 0 else _addresses[0]
                        else:
                            _address = _addresses
                        
                        lora_id = None
                        if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                            lora_id = self.agent_lora_mapping[agent_name]
                        
                        try:
                            lora_info = f" lora_id={lora_id}" if lora_id else ""
                            print(f"[Engine][generate_single_rollout] env_idx={env_idx} rollout_idx={rollout_idx} turn={turn_idx} agent={agent_name} policy={policy_name} chosen_address={_address}{lora_info}")
                        except Exception:
                            pass
                        
                        agent_config = self.agent_config_dict.get(agent_name, None)
                        agent_sample_num = getattr(agent_config, 'sample_num', 1) if agent_config else 1
                        
                        print(f"[Engine][DEBUG] About to call llm_async_generate for rollout_idx={rollout_idx}, agent={agent_name}")
                        output_dpr, response = await llm_async_generate(
                            rollout_idx=rollout_idx,
                            turn_idx=turn_idx,
                            agent_idx=agent_idx,
                            prompt_dpr=format_prompt,
                            ppo_trainer_config=ppo_trainer_config,
                            address=_address,
                            model_name=model_name,
                            tokenizer=self.tokenizer_dict[policy_name],
                            enable_thinking=agent_enable_thinking,
                            image_data=None,
                            application_id=str(uuid.uuid4()),
                            env_idx=rollout_idx // self.sample_num,
                            policy_name=policy_name,
                            timeout=self.generate_timeout,
                            mode=self.mode,
                            lora_id=lora_id,
                            agent_config=agent_config,
                        )
                    except Exception as e:
                        self.multi_logger.log_env_agent_info(
                            self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                            f"Failed to generate response: {e}",
                            {"error": str(e), "traceback": traceback.format_exc()}
                        )
                        output_dpr = None
                        response = ""
                    
                    if response is None:
                        response = ""
                    
                    current_agent.update_from_model(response)

                    try:
                        # Deterministic worker assignment based on rollout_idx
                        # This ensures consistent worker-task mapping for file path isolation
                        env_worker_id = rollout_idx % self.num_workers
                        env_worker = self.env_workers[env_worker_id]

                        # Store worker_id and GPU group in env_data for task folder path generation
                        if hasattr(self.envs[rollout_idx], 'state'):
                            self.envs[rollout_idx].state.assigned_worker_id = env_worker_id
                            self.envs[rollout_idx].state.gpu_group_id = self.gpu_group_id

                        await asyncio.wait_for(
                            current_agent.step(self.envs[rollout_idx], env_worker=env_worker),
                            timeout=self.step_timeout
                        )
                    except asyncio.TimeoutError:
                        self.multi_logger.log_env_agent_info(
                            self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                            f"Environment step timed out after {self.step_timeout}s",
                            {"error": "timeout", "timeout_seconds": self.step_timeout}
                        )
                    
                    # Capture env state immediately after this agent's step
                    env_state_snapshot = env.state.to_dict_compact(agent_name=agent_name) if hasattr(env.state, 'to_dict_compact') else None

                    agent_outputs.append({
                        'agent_idx': agent_idx,
                        'agent_name': agent_name,
                        'current_agent': current_agent,
                        'output_dpr': output_dpr,
                        'response': response,
                        'prompt': prompt,
                        'policy_name': policy_name,
                        'env_state_snapshot': env_state_snapshot
                    })
            
            # After all agents have completed their step in this turn, calculate rewards
            for agent_output in agent_outputs:
                agent_idx = agent_output['agent_idx']
                agent_name = agent_output['agent_name']
                current_agent = agent_output['current_agent']
                output_dpr = agent_output['output_dpr']
                response = agent_output['response']
                prompt = agent_output['prompt']
                policy_name = agent_output['policy_name']
                
                # Calculate reward using agent's calculate_reward method
                if hasattr(current_agent, 'calculate_reward'):
                    current_agent.calculate_reward(env)
                else:
                    # Fallback: append current agent_reward to history
                    if hasattr(current_agent, 'agent_reward'):
                        current_agent.reward_history.append(current_agent.agent_reward)
                
                # Now assign reward to output_dpr and add to trajectory
                if output_dpr is not None:
                    output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
                    output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name], dtype=object)
                    
                    if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                        batch_size = output_dpr.batch.batch_size[0] if hasattr(output_dpr.batch, 'batch_size') else len(output_dpr.batch)
                        lora_ids = [self.agent_lora_mapping[agent_name]] * batch_size
                        output_dpr.non_tensor_batch["lora_ids"] = np.array(lora_ids, dtype=object)
                
                    if trajectory_per_task_dict[policy_name].batch is None:
                        # If empty, assign directly
                        trajectory_per_task_dict[policy_name] = output_dpr
                    else:
                        try:
                            trajectory_per_task_dict[policy_name] = DataProto.concat([
                                trajectory_per_task_dict[policy_name],
                                output_dpr
                            ])
                        except Exception as e:
                           
                            print(f"The length of concatenated trajectory_per_task_dict[policy_name]: {len(trajectory_per_task_dict[policy_name])}")
                
                # Use captured state snapshot for this specific agent
                env_state_compact = agent_output.get('env_state_snapshot') or (env.state.to_dict_compact(agent_name=agent_name) if hasattr(env.state, 'to_dict_compact') else env.state)
                self.multi_logger.log_env_agent_info(
                        self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                        "Trajectory information updated",
                        {
                            "agent_prompt": {"text": prompt, "image": None},
                            "agent_response": response,
                            "env_state": env_state_compact,
                        }
                    )
        
            finish=False
            if env.done:
                finish=True
            if finish:
                agent_rewards={agent_name: agent.reward_history for agent_name, agent in zip(self.turn_order, agent_group)}
                
                self.multi_logger.log_rollout_summary(
                    self.mode, env_idx, rollout_idx, agent_rewards,
                    "success",
                    extra_data={
                        "turn_idx": turn_idx,
                        "message": f"Rollout {rollout_idx} completed successfully"
                    }
                )
                break
        agent_rewards={agent_name: agent.reward_history for agent_name, agent in zip(self.turn_order, agent_group)}
        self.multi_logger.log_rollout_summary(
                self.mode, env_idx, rollout_idx, agent_rewards,
                "rollout_complete",
                extra_data={
                    "turn_idx": turn_idx if 'turn_idx' in locals() else self.config.env.max_turns,
                    "message": f"Rollout {rollout_idx} completed"
                }
            )

        if self.mode=="validate":
            for i,agent_name in enumerate(self.turn_order):
                current_agent=self.agent_groups_list[rollout_idx][i]
                if current_agent.success:
                    self.success_rollout_idx_list_dict[agent_name].append(rollout_idx)
                    self.success_ave_turn_dict[agent_name] += turn_idx + 1
        
       
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
            

        
    async def generate_env_idx_rollout(self, env_idx, if_greedy=True,):
        """
        Generate a single rollout, adapted for multi-agent interaction in the code testing environment.
        
        Args:
            env: Code testing environment instance
            timing_raw: Timing record dictionary
            meta_info: Meta information
            if_greedy: Whether to use greedy sampling (True: select best reward, False: select first)
           
        Returns:
            DataProto: DataProto object containing trajectory data
        """
        trajectory_per_task_dict = {}
        rollout_idx_list=self.env_rollout_mapping[env_idx]
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()

        envs_list = [self.envs[rollout_idx] for rollout_idx in rollout_idx_list]
        agent_groups = [self.agent_groups_list[rollout_idx] for rollout_idx in rollout_idx_list]
        
        for turn_idx in range(self.max_turns):
            async def async_generate_response(idx, agent_idx, agent_name):
                rollout_idx = rollout_idx_list[idx]
                env = envs_list[idx]
                current_agent = agent_groups[idx][agent_idx]
                current_agent.update_from_env(turn_idx, env)
                prompt = current_agent.current_prompt
                policy_name = self.agent_policy_mapping.get(agent_name) 
                # Get enable_thinking for this specific agent
                agent_enable_thinking = self.agent_enable_thinking.get(agent_name, False)
                format_prompt = convert_prompt_to_dpr(
                    self.tokenizer_dict[policy_name], 
                    self.processor_dict.get(policy_name), 
                    prompt, 
                    self.max_prompt_length,
                    multi_modal=False,
                    enable_thinking=agent_enable_thinking
                )
                ppo_trainer_config = self.ppo_trainer_config_dict.get(policy_name, None)
                model_path = ppo_trainer_config.actor_rollout_ref.model.path
                # Match vLLM's model registration logic:
                # If path contains "checkpoint", use full path; otherwise use last 2 segments
                if "checkpoint" in str(model_path):
                    server_name = str(model_path)
                else:
                    server_name = "/".join(str(model_path).split("/")[-2:])
                print(f"[Engine][MODEL_NAME] env_idx={env_idx} rollout_idx={rollout_idx} turn={turn_idx} agent={agent_name} model_path={model_path} server_name={server_name}")
                
                output_dpr = None
                response = None
                try:
                    _addresses = self.server_address_dict.get(policy_name)
                    if isinstance(_addresses, (list, tuple)):
                        _address = random.choice(_addresses) if len(_addresses) > 0 else None
                    else:
                        _address = _addresses
                    if _address is None:
                        raise ValueError(f"No server address configured for policy '{policy_name}'")
                    
                    lora_id = None
                    if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                        lora_id = self.agent_lora_mapping[agent_name]
                   
                    # Get agent config for temperature settings
                    agent_config = self.agent_config_dict.get(agent_name, None)
                    
                    # Check if this agent supports sampling (has sample_num attribute)
                    agent_sample_num = getattr(agent_config, 'sample_num', 1) if agent_config else 1
                   
                    print(f"[Engine][DEBUG] About to call llm_async_generate in async_generate_response for rollout_idx={rollout_idx}, agent={agent_name}, turn_idx={turn_idx}")
                    print(f"[Engine][AWAIT_START] Calling llm_async_generate at {time.time()}")
                    output_dpr, response = await llm_async_generate(
                        rollout_idx=rollout_idx, 
                        turn_idx=turn_idx, 
                        agent_idx=agent_idx,
                        prompt_dpr=format_prompt, 
                        ppo_trainer_config=ppo_trainer_config,
                        address=_address,
                        model_name=server_name,
                        tokenizer=self.tokenizer_dict[policy_name],
                        enable_thinking=agent_enable_thinking,
                        image_data=None,
                        application_id=str(uuid.uuid4()),
                        env_idx=env_idx,
                        policy_name=policy_name,
                        timeout=self.generate_timeout,
                        mode=self.mode,
                        lora_id=lora_id,
                        agent_config=agent_config,
                    )
                    print(f"[Engine][AWAIT_COMPLETE] llm_async_generate returned at {time.time()}")
                    print(f"[Engine][DEBUG] llm_async_generate returned for rollout_idx={rollout_idx}, agent={agent_name}, response={'empty' if not response else 'has content'}")
                except Exception as e:
                    print(f"[Engine][ERROR] Exception in llm_async_generate for rollout_idx={rollout_idx}, agent={agent_name}: {e}")
                    output_dpr = None
                    response = None
                    # log error to help diagnose empty responses
                    try:
                        self.multi_logger.log_env_agent_info(
                            self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                            f"Failed to generate response: {e}",
                            {"error": str(e), "traceback": traceback.format_exc()}
                        )
                    except Exception:
                        pass
                
                return {
                    'idx': idx,
                    'agent_idx': agent_idx,
                    'agent_name': agent_name,
                    'rollout_idx': rollout_idx,
                    'output_dpr': output_dpr,
                    'response': response,
                    'prompt': prompt,
                    'policy_name': policy_name,
                    'env_state_snapshot': None,
                }
            
              
          
            if self.parallel:
                print(f"[Engine][PARALLEL_MODE] env_idx={env_idx} turn={turn_idx} - Parallel LLM for all agents, sequential steps")
                
                async def generate_with_timeout(idx, agent_idx, agent_name):
                    try:
                        result = await asyncio.wait_for(
                            async_generate_response(idx, agent_idx, agent_name),
                            timeout=self.generate_timeout
                        )
                        return result
                    except asyncio.TimeoutError:
                        rollout_idx = rollout_idx_list[idx]
                        policy_name = self.agent_policy_mapping.get(agent_name)
                        try:
                            current_agent = agent_groups[idx][agent_idx]
                            current_agent.update_from_env(turn_idx, envs_list[idx])
                            prompt = current_agent.current_prompt
                        except Exception:
                            prompt = None
                        
                        print(f"[TIMEOUT] Generate response timeout for rollout {rollout_idx} agent {agent_name}")
                        return {
                            'idx': idx,
                            'agent_idx': agent_idx,
                            'agent_name': agent_name,
                            'rollout_idx': rollout_idx,
                            'output_dpr': None,
                            'response': None,
                            'prompt': prompt,
                            'policy_name': policy_name,
                            'env_state_snapshot': None,
                        }
                    except Exception as e:
                        rollout_idx = rollout_idx_list[idx]
                        policy_name = self.agent_policy_mapping.get(agent_name)
                        try:
                            current_agent = agent_groups[idx][agent_idx]
                            current_agent.update_from_env(turn_idx, envs_list[idx])
                            prompt = current_agent.current_prompt
                        except Exception:
                            prompt = None
                        
                        print(f"[ERROR] Generate response failed for rollout {rollout_idx} agent {agent_name}: {e}")
                        return {
                            'idx': idx,
                            'agent_idx': agent_idx,
                            'agent_name': agent_name,
                            'rollout_idx': rollout_idx,
                            'output_dpr': None,
                            'response': None,
                            'prompt': prompt,
                            'policy_name': policy_name,
                            'env_state_snapshot': None,
                        }
                
                all_agents_results = []
                for agent_idx, agent_name in enumerate(self.turn_order):
                    tasks = [generate_with_timeout(idx, agent_idx, agent_name) for idx in range(len(rollout_idx_list))]
                    response_results = await asyncio.gather(*tasks, return_exceptions=False)
                    all_agents_results.append((agent_idx, agent_name, response_results))
                
                for agent_idx, agent_name, response_results in all_agents_results:
                    for idx in range(len(rollout_idx_list)):
                        result = response_results[idx]
                        rollout_idx = result['rollout_idx']
                        response = result['response']
                        
                        current_agent = agent_groups[idx][agent_idx]
                        if response is None:
                            response = ""
                        current_agent.update_from_model(response)

                        # Deterministic worker assignment based on rollout_idx
                        # This ensures consistent worker-task mapping for file path isolation
                        env_worker_id = rollout_idx % self.num_workers
                        env_worker = self.env_workers[env_worker_id]

                        # Store worker_id and GPU group in env_data for task folder path generation
                        if hasattr(self.envs[rollout_idx_list[idx]], 'state'):
                            self.envs[rollout_idx_list[idx]].state.assigned_worker_id = env_worker_id
                            self.envs[rollout_idx_list[idx]].state.gpu_group_id = self.gpu_group_id

                        try:
                            await asyncio.wait_for(
                                current_agent.step(self.envs[rollout_idx_list[idx]], env_worker=env_worker),
                                timeout=self.step_timeout
                            )
                        except asyncio.TimeoutError:
                            pass
                        
                        # Capture env state immediately after this agent's step
                        env_state_snapshot = envs_list[idx].state.to_dict_compact(agent_name=agent_name) if hasattr(envs_list[idx].state, 'to_dict_compact') else None
                        response_results[idx]['env_state_snapshot'] = env_state_snapshot
                
                # After all agents have completed their step in this turn, calculate rewards
                for agent_idx, agent_name, response_results in all_agents_results:
                    for idx in range(len(rollout_idx_list)):
                        current_agent = agent_groups[idx][agent_idx]
                        env = envs_list[idx]
                        
                        if hasattr(current_agent, 'calculate_reward'):
                            current_agent.calculate_reward(env)
                        else:
                            if hasattr(current_agent, 'agent_reward'):
                                if not hasattr(current_agent, 'reward_history'):
                                    current_agent.reward_history = []
                                current_agent.reward_history.append(current_agent.agent_reward)
                    
                    for idx in range(len(rollout_idx_list)):
                        result = response_results[idx]
                        rollout_idx = result['rollout_idx']
                        output_dpr = result['output_dpr']
                        response = result['response']
                        prompt = result['prompt']
                        policy_name = result['policy_name']
                        
                        current_agent = agent_groups[idx][agent_idx]
                        env = envs_list[idx]
                
                        if agent_name == self.turn_order[-1]:
                            # Use captured state snapshot for this specific agent
                            env_state_compact = result.get('env_state_snapshot') or (env.state.to_dict_compact(agent_name=agent_name) if hasattr(env.state, 'to_dict_compact') else env.state)
                            self.multi_logger.log_env_agent_info(
                                self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                                "Trajectory information updated",
                                {
                                    "agent_prompt": {"text": prompt if prompt is not None else "", "image": None},
                                    "agent_response": response,
                                    "env_state": env_state_compact,
                                }
                            )

                        if output_dpr is not None:
                            output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
                            output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name], dtype=object)
                            
                            if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                                batch_size = output_dpr.batch.batch_size[0] if hasattr(output_dpr.batch, 'batch_size') else len(output_dpr.batch)
                                lora_ids = [self.agent_lora_mapping[agent_name]] * batch_size
                                output_dpr.non_tensor_batch["lora_ids"] = np.array(lora_ids, dtype=object)
                            
                            if trajectory_per_task_dict[policy_name].batch is None:
                                trajectory_per_task_dict[policy_name] = output_dpr
                            else:
                                trajectory_per_task_dict[policy_name] = DataProto.concat([
                                    trajectory_per_task_dict[policy_name], 
                                    output_dpr
                                ])
                    
                    rollout_score_idx = []
                    for idx in range(len(rollout_idx_list)):
                        current_agent = agent_groups[idx][agent_idx]
                        agent_reward = current_agent.agent_reward if current_agent.agent_reward is not None else 0
                        rollout_score_idx.append(agent_reward)
                    
                    try:
                        if if_greedy:
                            best_i = int(np.argmax(np.asarray(rollout_score_idx)))
                        else:
                            best_i = 0
                    except Exception:
                        best_i = 0
                    
                    selected_env = envs_list[best_i]
                    selected_agent_group = agent_groups[best_i]
                    
                    task_done = selected_env.done
                    
                    envs_list = [copy.deepcopy(selected_env) for _ in envs_list]
                    agent_groups = [copy.deepcopy(selected_agent_group) for _ in agent_groups]
                    
                    if task_done:
                        break
            else:
                print(f"[Engine][SEQUENTIAL_MODE] env_idx={env_idx} turn={turn_idx} - Sequential agent execution")
                
                for agent_idx, agent_name in enumerate(self.turn_order):
                    async def generate_with_timeout(idx, agent_idx, agent_name):
                        try:
                            result = await asyncio.wait_for(
                                async_generate_response(idx, agent_idx, agent_name),
                                timeout=self.generate_timeout
                            )
                            return result
                        except asyncio.TimeoutError:
                            rollout_idx = rollout_idx_list[idx]
                            policy_name = self.agent_policy_mapping.get(agent_name)
                            try:
                                current_agent = agent_groups[idx][agent_idx]
                                current_agent.update_from_env(turn_idx, envs_list[idx])
                                prompt = current_agent.current_prompt
                            except Exception:
                                prompt = None
                            
                            print(f"[TIMEOUT] Generate response timeout for rollout {rollout_idx} agent {agent_name}")
                            return {
                                'idx': idx,
                                'agent_idx': agent_idx,
                                'agent_name': agent_name,
                                'rollout_idx': rollout_idx,
                                'output_dpr': None,
                                'response': None,
                                'prompt': prompt,
                                'policy_name': policy_name,
                                'env_state_snapshot': None,
                            }
                        except Exception as e:
                            rollout_idx = rollout_idx_list[idx]
                            policy_name = self.agent_policy_mapping.get(agent_name)
                            try:
                                current_agent = agent_groups[idx][agent_idx]
                                current_agent.update_from_env(turn_idx, envs_list[idx])
                                prompt = current_agent.current_prompt
                            except Exception:
                                prompt = None
                            
                            print(f"[ERROR] Generate response failed for rollout {rollout_idx} agent {agent_name}: {e}")
                            return {
                                'idx': idx,
                                'agent_idx': agent_idx,
                                'agent_name': agent_name,
                                'rollout_idx': rollout_idx,
                                'output_dpr': None,
                                'response': None,
                                'prompt': prompt,
                                'policy_name': policy_name,
                                'env_state_snapshot': None,
                            }
                    
                    tasks = [generate_with_timeout(idx, agent_idx, agent_name) for idx in range(len(rollout_idx_list))]
                    response_results = await asyncio.gather(*tasks, return_exceptions=False)
                    
                    for idx in range(len(rollout_idx_list)):
                        result = response_results[idx]
                        rollout_idx = result['rollout_idx']
                        response = result['response']
                        
                        current_agent = agent_groups[idx][agent_idx]
                        if response is None:
                            response = ""
                        current_agent.update_from_model(response)

                        # Deterministic worker assignment based on rollout_idx
                        # This ensures consistent worker-task mapping for file path isolation
                        env_worker_id = rollout_idx % self.num_workers
                        env_worker = self.env_workers[env_worker_id]

                        # Store worker_id and GPU group in env_data for task folder path generation
                        if hasattr(self.envs[rollout_idx_list[idx]], 'state'):
                            self.envs[rollout_idx_list[idx]].state.assigned_worker_id = env_worker_id
                            self.envs[rollout_idx_list[idx]].state.gpu_group_id = self.gpu_group_id

                        try:
                            await asyncio.wait_for(
                                current_agent.step(self.envs[rollout_idx_list[idx]], env_worker=env_worker),
                                timeout=self.step_timeout
                            )
                        except asyncio.TimeoutError:
                            pass
                        
                        # Capture env state immediately after this agent's step
                        env_state_snapshot = envs_list[idx].state.to_dict_compact(agent_name=agent_name) if hasattr(envs_list[idx].state, 'to_dict_compact') else None
                        response_results[idx]['env_state_snapshot'] = env_state_snapshot
                
                # After all agents have completed their step in this turn, calculate rewards
                for agent_idx, agent_name in enumerate(self.turn_order):   
                    for idx in range(len(rollout_idx_list)):
                        current_agent = agent_groups[idx][agent_idx]
                        env = envs_list[idx]
                        
                        if hasattr(current_agent, 'calculate_reward'):
                            current_agent.calculate_reward(env)
                        else:
                            if hasattr(current_agent, 'agent_reward'):
                                if not hasattr(current_agent, 'reward_history'):
                                    current_agent.reward_history = []
                                current_agent.reward_history.append(current_agent.agent_reward)
                    
                    for idx in range(len(rollout_idx_list)):
                        result = response_results[idx]
                        rollout_idx = result['rollout_idx']
                        output_dpr = result['output_dpr']
                        response = result['response']
                        prompt = result['prompt']
                        policy_name = result['policy_name']
                        
                        current_agent = agent_groups[idx][agent_idx]
                        env = envs_list[idx]
                        # Use captured state snapshot for this specific agent
                        env_state_compact = result.get('env_state_snapshot') or (env.state.to_dict_compact(agent_name=agent_name) if hasattr(env.state, 'to_dict_compact') else env.state)
                        self.multi_logger.log_env_agent_info(
                            self.mode, env_idx, rollout_idx, turn_idx + 1, agent_name,
                            "Trajectory information updated",
                            {
                                "agent_prompt": {"text": prompt if prompt is not None else "", "image": None},
                                "agent_response": response,
                                "env_state": env_state_compact,
                            }
                        )

                        if output_dpr is not None:
                            output_dpr.non_tensor_batch["reward"] = np.array([current_agent.agent_reward])
                            output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name], dtype=object)
                            
                            if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                                batch_size = output_dpr.batch.batch_size[0] if hasattr(output_dpr.batch, 'batch_size') else len(output_dpr.batch)
                                lora_ids = [self.agent_lora_mapping[agent_name]] * batch_size
                                output_dpr.non_tensor_batch["lora_ids"] = np.array(lora_ids, dtype=object)
                            
                            if trajectory_per_task_dict[policy_name].batch is None:
                                trajectory_per_task_dict[policy_name] = output_dpr
                            else:
                                trajectory_per_task_dict[policy_name] = DataProto.concat([
                                    trajectory_per_task_dict[policy_name], 
                                    output_dpr
                                ])
                    
                    rollout_score_idx = []
                    for idx in range(len(rollout_idx_list)):
                        current_agent = agent_groups[idx][agent_idx]
                        agent_reward = current_agent.agent_reward if current_agent.agent_reward is not None else 0
                        rollout_score_idx.append(agent_reward)
                    
                    try:
                        if if_greedy:
                            best_i = int(np.argmax(np.asarray(rollout_score_idx)))
                        else:
                            best_i = 0
                    except Exception:
                        best_i = 0
                    
                    selected_env = envs_list[best_i]
                    selected_agent_group = agent_groups[best_i]
                    
                    task_done = selected_env.done
                    
                    envs_list = [copy.deepcopy(selected_env) for _ in envs_list]
                    agent_groups = [copy.deepcopy(selected_agent_group) for _ in agent_groups]
                    
                    if task_done:
                        break
            
            if task_done:
                break
        
        return trajectory_per_task_dict


    async def generate_multiple_rollouts_concurrent(self, env_idx_list, rollout_mode="tree"):
        rollout_indices=[]
        for env_idx in env_idx_list:
            rollout_indices.extend(self.env_rollout_mapping[env_idx])
        concurrent_timer = create_timer("ConcurrentRollouts")
        concurrent_timer.start(f"Starting concurrent rollouts for {len(rollout_indices)} rollouts")
        
        concurrent_timer.checkpoint("Creating async tasks")
        
        if rollout_mode == "tree" and self.mode != "validate":
            tasks = [
                asyncio.create_task(
                    self.generate_env_idx_rollout(env_idx, if_greedy=getattr(self.config, 'if_greedy', True)), 
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
        with open("rollout_data.json", "w", encoding="utf-8") as f:
            json.dump(self.rollout_latency_dict, f, ensure_ascii=False, indent=4)
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
                "total_rollouts": len(rollout_indices),
                "failed": failed_count,
                "success_rate": f"{completed_count}/{len(tasks)}",
                "aggregated_policies": list(aggregated_results.keys()),
            }
        )
        
        # Log Ray status after concurrent execution
        self.multi_logger.log_ray_status(mode=self.mode, context="after_concurrent_batch")
        
        concurrent_timer.end("Concurrent rollouts completed successfully")
        return aggregated_results



