import asyncio
import logging
import time
import json
import traceback
import uuid
from tqdm.asyncio import tqdm
import random
import ray
try:
    from verl.protocol import DataProto
except Exception:  # fallback when verl is a src tree: verl/verl/protocol.py
    from verl import DataProto
import torch
import numpy as np
from pettingllms.trainer.multiagentsys_graph_register import (
    ENV_CLASS_MAPPING,
    ENV_BATCH_CLASS_MAPPING,
)
from functools import partial
import multiprocessing
from pettingllms.utils.performance import create_timer
import copy
from pettingllms.trainer.async_generate import convert_prompt_to_dpr, llm_async_generate
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.utils.openai import (
    build_agent_address_mapping,
    create_dummy_model_client,
    init_patch_context,
    patch_all,
    wrap_autogen_graph,
    get_trajectory_store
)
from pettingllms.trainer.core_algo import calculate_reward

logger = logging.getLogger(__name__)




class MultiAgentsExecutionEngineGraph:
    def _load_config_parameters(self):
        self.max_prompt_length = getattr(self.config.training, 'max_prompt_length', 1024)
        self.max_response_length = getattr(self.config.training, 'max_response_length', 1024)
        self.generate_timeout = getattr(self.config.training, 'generate_timeout', 300.0)
        # Multi-modal support configuration
        self.enable_multimodal = getattr(self.config.training, 'enable_multimodal', False)
        # Agent framework configuration
        self.agent_framework = getattr(self.config, 'agent_framework', 'autogen')
          
        
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
        # Read parameters from config with fallback to defaults
        self.timer.checkpoint("Loading config parameters")
        self._load_config_parameters()
        self.n_cpu = multiprocessing.cpu_count()
        # Environment configuration - direct access
        env_name = getattr(self.config.env, 'name', None)
        if env_name is None:
            raise ValueError("env.name is not set in the config.env")
            
        print(f"env_name: {env_name}")
        self.experiment_name = self.config.training.experiment_name
        self.env_name = env_name
        self.env_class = ENV_CLASS_MAPPING[env_name]
        self.agent_configs_raw = self.config.agent_policy_configs.agent_configs
        self.agent_config_dict = {}
        # Generate agent names list from agent_configs
        self.agent_names = []
        for agent_key, agent_config in self.agent_configs_raw.items():
            agent_name = agent_config.name
            self.agent_config_dict[agent_name] = agent_config
            self.agent_names.append(agent_name)
        # Calculate num_interacting_agents from agent_configs length
        self.num_interacting_agents = len(self.agent_names)
        self.step_timeout = getattr(self.config.training, 'step_timeout', 150.0)
        print(f"agent_config_dict keys: {list(self.agent_config_dict.keys())}")
        print(f"agent_names: {self.agent_names}")
        print(f"num_interacting_agents: {self.num_interacting_agents}")
        self.server_address_dict = server_address_dict 
        self.chat_parser_dict={}
        self.rollout_latency_dict = {}
        self.timer.checkpoint("MultiAgentsExecutionEngine initialization completed")

        


    def get_graph_function(self):
        """
        Get the appropriate graph function from config.training.workflow_function.
        
        Uses AGENT_WORKER_FLOW_FUNCTIONS registry to look up the function by name.
        
        Returns:
            graph_func: The graph function to execute
            
        Raises:
            ValueError: If workflow_function not in config or not in registry
        """
        from pettingllms.trainer.multiagentsys_graph_register import AGENT_WORKER_FLOW_FUNCTIONS_MAPPING

        # Get workflow_function name from config
        workflow_function_name = getattr(self.config, 'workflow_function', None)
        if workflow_function_name is None:
            raise ValueError("workflow_function not specified in config")

        # Look up in registry
        if workflow_function_name not in AGENT_WORKER_FLOW_FUNCTIONS_MAPPING:
            raise ValueError(
                f"workflow_function '{workflow_function_name}' not found in AGENT_WORKER_FLOW_FUNCTIONS_MAPPING. "
            )
        graph_func = AGENT_WORKER_FLOW_FUNCTIONS_MAPPING[workflow_function_name]
        return graph_func
    

    
    def calculate_cpu_per_rollout(self, num_concurrent_rollouts):
        total_cpu = self.n_cpu
        max_cpu_usage = total_cpu * 0.6  # Use at most 60% of CPUs

        # Calculate CPU per rollout
        cpu_per_rollout = max_cpu_usage / num_concurrent_rollouts
        cpu_per_rollout = max(0.1, cpu_per_rollout)

        return cpu_per_rollout

    def _log_rollout_tracking(self, rollout_tracking_dict):
        """
        Log rollout tracking information for each rollout.

        Args:
            rollout_tracking_dict: Dictionary containing tracking data for all rollouts
                Structure: {rollout_idx: {'hops': [...], 'env_idx': int, 'rollout_idx': int}}
        """
        print("\n" + "="*80)
        print("ROLLOUT TRACKING SUMMARY")
        print("="*80)

        for rollout_idx, tracking_data in sorted(rollout_tracking_dict.items()):
            env_idx = tracking_data['env_idx']
            hops = tracking_data['hops']

            print(f"\nRollout {rollout_idx} (Env {env_idx}):")
            print(f"  Total hops: {len(hops)}")

            for hop_data in hops:
                hop_idx = hop_data['hop_idx']
                agent_name = hop_data['agent_name']
                policy_name = hop_data['policy_name']
                dataproto_uuid = hop_data['dataproto_uuid']
                response_preview = hop_data['response'][:100] if len(hop_data['response']) > 100 else hop_data['response']

                print(f"    Hop {hop_idx}:")
                print(f"      Agent: {agent_name}")
                print(f"      Policy: {policy_name}")
                print(f"      DataProto UUID: {dataproto_uuid}")
                print(f"      Response preview: {response_preview}...")

        # Log to multi_logger for structured logging
        self.multi_logger.log_async_event(
            self.mode, -1, -1, "rollout_tracking_summary",
            "Rollout tracking data collected",
            {
                "total_rollouts": len(rollout_tracking_dict),
                "rollout_tracking": {
                    str(rollout_idx): {
                        'env_idx': data['env_idx'],
                        'total_hops': len(data['hops']),
                        'hops': [
                            {
                                'hop_idx': hop['hop_idx'],
                                'agent_name': hop['agent_name'],
                                'policy_name': hop['policy_name'],
                                'dataproto_uuid': hop['dataproto_uuid']
                            }
                            for hop in data['hops']
                        ]
                    }
                    for rollout_idx, data in rollout_tracking_dict.items()
                }
            }
        )

        print("="*80 + "\n")

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
        for agent_name in self.agent_names:
            agent_config = self.agent_config_dict.get(agent_name, None)
            # Read enable_thinking from agent config, default to False
            enable_thinking = getattr(agent_config, 'enable_thinking', False) if agent_config else False
            self.agent_enable_thinking[agent_name] = enable_thinking
            # Read enable_multimodal from agent config, fallback to global setting
            enable_multimodal = getattr(agent_config, 'enable_multimodal', self.enable_multimodal) if agent_config else self.enable_multimodal
            self.agent_enable_multimodal[agent_name] = enable_multimodal
            print(f"Agent '{agent_name}' enable_thinking: {enable_thinking}, enable_multimodal: {enable_multimodal}")
        
        if mode=="validate":
            self.sample_num=self.config.training.validate_sample_num
            self.gen_batch_size=1
            for agent_name in self.agent_names:
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
 
    async def generate_single_rollout(self, rollout_idx, model_client_dict, rollout_tracking_dict, cpu_per_rollout=None):
        """
        Generate a single rollout with tracking.

        Args:
            rollout_idx: Index of the rollout
            model_client_dict: Dictionary of model clients shared across rollouts
            rollout_tracking_dict: Dictionary to track rollout data (agent, hop, uuid)
            cpu_per_rollout: CPU allocation per rollout

        Returns:
            trajectory_per_task_dict: Dictionary of trajectories per policy
        """

        trajectory_per_task_dict = {p: DataProto() for p in self.tokenizer_dict.keys()}
        env_idx = rollout_idx // self.sample_num
        env = self.envs[rollout_idx]

        # Initialize tracking for this rollout
        if rollout_idx not in rollout_tracking_dict:
            rollout_tracking_dict[rollout_idx] = {
                'hops': [],  # List of {hop_idx, agent_name, dataproto_uuid, policy_name}
                'env_idx': env_idx,
                'rollout_idx': rollout_idx
            }

        # Get the autogen graph workflow function from registry using config.workflow_function
        graph_func = self.get_graph_function()

        # Wrap the graph function to track execution
        wrapped_graph = wrap_autogen_graph(graph_func)

        # Execute the patched autogen graph
        # The patch will intercept OpenAIChatCompletionClient.create() calls
        # and route them to llm_async_generate, collecting trajectories

        # Wrap and run the graph with CPU resource hint
        result_env = await wrapped_graph(env=env, model_client_dict=model_client_dict)
        # After graph execution, collect trajectories from the patch context
        trajectory_store = get_trajectory_store()

        # Calculate final reward from environment
        # For code env, reward is based on test pass rate
        final_reward = 0.0
        final_reward = getattr(result_env, 'final_reward', 0.0)

        # First, collect all output_dpr and mark env_final_reward
        collected_trajectories = []
        for (r_idx, h_idx, policy_name), (output_dpr, response) in trajectory_store.items():
            if r_idx != rollout_idx:
                continue  # Skip trajectories from other rollouts

            # Mark env_final_reward in non_tensor_batch for later reward calculation
            output_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward])

            # Handle LoRA if enabled
            agent_name = output_dpr.non_tensor_batch.get("agent_name", [None])[0]
            if self.lora_differ_mode and agent_name in self.agent_lora_mapping:
                batch_size = output_dpr.batch.batch_size[0] if hasattr(output_dpr.batch, 'batch_size') else len(output_dpr.batch)
                output_dpr.non_tensor_batch["lora_ids"] = np.array(
                    [self.agent_lora_mapping[agent_name]] * batch_size,
                    dtype=object
                )

            # Track this hop's information
            dataproto_uuid = str(uuid.uuid4())
            output_dpr.non_tensor_batch["dataproto_uuid"] = np.array([dataproto_uuid], dtype=object)

            rollout_tracking_dict[rollout_idx]['hops'].append({
                'hop_idx': h_idx,
                'agent_name': agent_name,
                'policy_name': policy_name,
                'dataproto_uuid': dataproto_uuid,
                'response': response
            })

            collected_trajectories.append((policy_name, output_dpr))
        
        # Now calculate rewards using the reward calculation function from core_algo
        reward_algorithm = getattr(self.config.training, 'reward_algorithm', 'default')
        reward_kwargs = {
            'max_hops': getattr(self.config.training, 'max_hops', 10),
            'discount_factor': getattr(self.config.training, 'reward_discount_factor', 0.99),
            'reward_threshold': getattr(self.config.training, 'reward_threshold', 0.5),
        }
        
        for policy_name, output_dpr in collected_trajectories:
            # Calculate reward for this trajectory using core_algo
            output_dpr = calculate_reward(
                output_dpr, 
                algorithm=reward_algorithm,
                **reward_kwargs
            )

           
            # Concatenate to trajectory dict
            if trajectory_per_task_dict[policy_name].batch is None:
                trajectory_per_task_dict[policy_name] = output_dpr
            else:
                trajectory_per_task_dict[policy_name] = DataProto.concat([
                    trajectory_per_task_dict[policy_name], 
                    output_dpr
                ])

        return trajectory_per_task_dict
            

    async def generate_multiple_rollouts_concurrent(self, env_idx_list, rollout_mode="tree"):
        rollout_indices=[]
        for env_idx in env_idx_list:
            rollout_indices.extend(self.env_rollout_mapping[env_idx])
        concurrent_timer = create_timer("ConcurrentRollouts")
        concurrent_timer.start(f"Starting concurrent rollouts for {len(rollout_indices)} rollouts")

        concurrent_timer.checkpoint("Building agent address mapping")

        # Build agent address mapping before creating clients
        agent_address_mapping = build_agent_address_mapping(
            agent_names=self.agent_names,
            agent_policy_mapping=self.agent_policy_mapping,
            server_address_dict=self.server_address_dict
        )

        for agent_name, address in agent_address_mapping.items():
            print(f"[Engine] Agent '{agent_name}' mapped to address: {address}")

        concurrent_timer.checkpoint("Creating model clients")

        # Build model_client_dict with dummy clients (will be intercepted by patch)
        model_client_dict = {}
        for agent_name in self.agent_names:
            agent_config = self.agent_config_dict.get(agent_name)
            policy_name = getattr(agent_config, 'policy_name', None)
            model_client = create_dummy_model_client(self.agent_framework)
            # Update the model in _create_args which is where it's actually stored
            model_client._create_args['model'] = policy_name
            # Store agent_name directly on the client so we can retrieve it later
            model_client._agent_name = agent_name

            model_client_dict[agent_name] = model_client

        concurrent_timer.checkpoint("Applying patches")

        # Apply patch_all with all necessary mappings
        patch_all(
            server_address_dict=self.server_address_dict,
            tokenizer_dict=self.tokenizer_dict,
            ppo_trainer_config_dict=self.ppo_trainer_config_dict,
            agent_policy_mapping=self.agent_policy_mapping,
            agent_framework=self.agent_framework,
            agent_address_mapping=agent_address_mapping,
            agent_lora_mapping=self.agent_lora_mapping,
            agent_config_dict=self.agent_config_dict,
            processor_dict=self.processor_dict
        )

        concurrent_timer.checkpoint("Initializing rollout tracking dictionary")

        # Create rollout tracking dictionary to track each rollout's hop/agent/uuid data
        rollout_tracking_dict = {}

        concurrent_timer.checkpoint("Creating async tasks")

        # Calculate CPU allocation per rollout
        num_concurrent_rollouts = self.gen_batch_size * self.sample_num
        cpu_per_rollout = self.calculate_cpu_per_rollout(num_concurrent_rollouts)

        tasks = [
            asyncio.create_task(
                self.generate_single_rollout(
                    rollout_idx,
                    model_client_dict=model_client_dict,
                    rollout_tracking_dict=rollout_tracking_dict,
                    cpu_per_rollout=cpu_per_rollout
                ),
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
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)}, {failed_count} failed)")
                    
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

        # Log rollout tracking information
        concurrent_timer.checkpoint("Logging rollout tracking data")
        self._log_rollout_tracking(rollout_tracking_dict)

        if self.mode=="validate":
            for agent_name in self.agent_names:
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

        # Store rollout tracking dict for later access
        self.rollout_tracking_dict = rollout_tracking_dict

        return aggregated_results

