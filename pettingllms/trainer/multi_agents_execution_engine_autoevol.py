import asyncio
import logging
import time
import json
import traceback
import uuid
import os
import pickle
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
from pettingllms.multi_agent_env.autoevol.gen_agent import MASGenerator, MASExecutor
from functools import partial
import multiprocessing
from pettingllms.utils.performance import create_timer
import copy
from pettingllms.trainer.async_generate import convert_prompt_to_dpr, llm_async_generate
from pettingllms.utils.logger_config import get_multi_logger
from pettingllms.multi_agent_env.math.math_worker import get_ray_docker_worker_cls


logger = logging.getLogger(__name__)

_DEBUG_ENGINE = False


def _align_non_tensor_batch_keys(dpr_a: DataProto, dpr_b: DataProto) -> tuple:
    """Align non_tensor_batch keys between two DataProtos before concat.

    DataProto.concat requires all items to have the same keys.  Workflow
    DataProtos may carry extra keys (e.g. response_token_count) that the
    designer DataProto does not have, causing an AssertionError.  This
    helper adds missing keys (filled with None) so concat succeeds.
    """
    keys_a = set(dpr_a.non_tensor_batch.keys()) if dpr_a.non_tensor_batch else set()
    keys_b = set(dpr_b.non_tensor_batch.keys()) if dpr_b.non_tensor_batch else set()

    for key in keys_b - keys_a:
        dpr_a.non_tensor_batch[key] = np.array([None] * len(dpr_a), dtype=object)
    for key in keys_a - keys_b:
        dpr_b.non_tensor_batch[key] = np.array([None] * len(dpr_b), dtype=object)

    return dpr_a, dpr_b


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
        # Tree design sampling parameters
        self.design_sample_num = getattr(self.config.training, 'design_sample_num', None)
        self.execute_sample_num = getattr(self.config.training, 'execute_sample_num', None)
        # Multi-modal support configuration
        self.enable_multimodal = getattr(self.config.training, 'enable_multimodal', False)
        # Concurrency tuning parameters
        self.mas_concurrency = getattr(self.config.training, 'mas_concurrency', 10)
        self.llm_max_concurrent = getattr(self.config.training, 'llm_max_concurrent', 20)
          
        
    def __init__(
        self,
        config,
        ppo_trainer_config_dict=None,
        tokenizer_dict=None,
        tokenizer_path_dict=None,
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
        self.tokenizer_path_dict = tokenizer_path_dict or {}
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
        # Create agent class list based on turn_order
        # Store classes, not instances - instances will be created later with proper parameters
        self.agent_class_list = []
        for agent_name in self.turn_order:
            if agent_name == "Designer" or "designer" in agent_name.lower():
                self.agent_class_list.append(MASGenerator)
            elif agent_name == "Executor" or "executor" in agent_name.lower():
                self.agent_class_list.append(MASExecutor)
            else:
                # Default to MASGenerator for backward compatibility
                self.agent_class_list.append(MASGenerator)
        self.agent_configs_raw = self.config.agent_policy_configs.agent_configs
        self.agent_config_dict = {}
        for agent_key, agent_config in self.agent_configs_raw.items():
            agent_name = agent_config.name
            self.agent_config_dict[agent_name] = agent_config
        self.step_timeout = getattr(self.config.training, 'step_timeout', 1200.0)
        print(f"agent_config_dict keys: {list(self.agent_config_dict.keys())}")
        self.server_address_dict = server_address_dict or {}
        self.chat_parser_dict={}
        self.rollout_latency_dict = {}

        # Validate tokenizer_path_dict and agent_policy_mapping consistency
        print("=" * 80)
        print("Validating tokenizer_path_dict and agent_policy_mapping...")
        print(f"agent_policy_mapping: {self.agent_policy_mapping}")
        print(f"tokenizer_path_dict keys: {list(self.tokenizer_path_dict.keys())}")
        print(f"tokenizer_dict keys: {list(self.tokenizer_dict.keys()) if self.tokenizer_dict else []}")

        # Check if all policy names have corresponding tokenizer paths
        missing_tokenizer_paths = []
        for agent_name, policy_name in self.agent_policy_mapping.items():
            if policy_name not in self.tokenizer_path_dict:
                missing_tokenizer_paths.append((agent_name, policy_name))
                logger.warning(f"Policy '{policy_name}' for agent '{agent_name}' not found in tokenizer_path_dict")

        if missing_tokenizer_paths:
            logger.warning("Missing tokenizer_path_dict entries. This may cause errors during execution.")
            logger.warning("Consider checking your configuration to ensure model names match policy names.")
        else:
            print("All policy names have corresponding tokenizer paths.")
        print("=" * 80)

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
        

    async def _cleanup_after_step(self, rollout_idx: int):
        """
        Clean up resources after step() to prevent memory leaks from AG2/OpenAI clients.
        This helps prevent 'illegal memory' errors when running many concurrent rollouts.
        Note: Main cleanup happens in gen_agent._cleanup_ag2_resources() and subprocess cleanup code.
        This is an additional periodic garbage collection.
        """
        import gc

        try:
            # Force garbage collection every 4 rollouts (more frequent for tree design mode)
            if rollout_idx % 4 == 0:
                gc.collect()

        except Exception as e:
            logger.debug(f"Non-critical cleanup error for rollout {rollout_idx}: {e}")

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

        # Tree design sampling: auto-calculate sample_num from design × execute
        if self.design_sample_num and self.execute_sample_num and mode != "validate":
            self.sample_num = self.design_sample_num * self.execute_sample_num
            print(f"[TREE DESIGN] Using tree design sampling: design_sample_num={self.design_sample_num}, "
                  f"execute_sample_num={self.execute_sample_num}, sample_num={self.sample_num}")
        else:
            # Fallback: degrade to current behavior
            if not self.design_sample_num:
                self.design_sample_num = self.sample_num
            if not self.execute_sample_num:
                self.execute_sample_num = 1

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
     
        self.gen_batch_size=len(self.envs)//self.sample_num

        self.env_idx_list=range(len(self.envs)//self.sample_num)
        self.rollout_idx_list=range(len(self.envs))
        self.env_rollout_mapping={}
        for env_idx in range(len(self.env_idx_list)):
            self.env_rollout_mapping[env_idx] = [_ for _ in range(env_idx*self.sample_num, (env_idx+1)*self.sample_num)]
        self.timer.checkpoint("Starting batched env initialization")
            
        # For autoevol with split policy, create both Designer and Executor agents
        self.agent_groups_list = []
        for rollout_idx in range(len(self.envs)):
            agent_init_params = {
                'env_idx': rollout_idx,
                'agent_sample_idx': rollout_idx,
                'rollout_idx': rollout_idx,
                'task_type': getattr(self.envs[rollout_idx], 'problem_type',
                             getattr(self.config.env, 'task_type', 'math'))
            }
            agent_init_params['benchmark'] = getattr(self.config.env, 'benchmark', 'AIME24') if hasattr(self.config, 'env') else 'AIME24'

            # Create agent instances based on turn_order
            agent_group = []
            for agent_class in self.agent_class_list:
                agent = agent_class(**agent_init_params)
                agent_group.append(agent)
            self.agent_groups_list.append(agent_group)
        
    
                   
            
    async def generate_single_rollout(self, rollout_idx):
        """
        Generate a single rollout for autoevol with split policy:
        - Designer (model 0) generates MAS code
        - Executor (model 1) executes the MAS code
        Both use outcome reward to update their policies.

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

        env = self.envs[rollout_idx]
        agent_group = self.agent_groups_list[rollout_idx]  # List of agents: [Designer] or [Designer, Executor]
        
        # Get agent names and their corresponding policies
        designer_name = self.turn_order[0] if len(self.turn_order) > 0 else "Designer"
        executor_name = self.turn_order[1] if len(self.turn_order) > 1 else None
        
        designer_policy = self.agent_policy_mapping.get(designer_name)
        executor_policy = self.agent_policy_mapping.get(executor_name) if executor_name else designer_policy  # Use designer policy if executor not present
        
        designer_agent = agent_group[0] if len(agent_group) > 0 else None
        executor_agent = agent_group[1] if len(agent_group) > 1 else None
        
        # Check if we have at least designer agent
        if designer_agent is None:
            logger.error(f"Missing designer agent for rollout {rollout_idx}")
            return trajectory_per_task_dict
        
        # Check if we have executor agent (for split policy mode)
        has_executor = executor_agent is not None and executor_name is not None

        mode_str = "split policy" if has_executor else "shared policy"
        self.multi_logger.log_async_event(
            self.mode, env_idx, rollout_idx, "generation_start",
            f"Starting {mode_str} rollout for rollout {rollout_idx}",
            {"rollout_idx": rollout_idx, "designer": designer_name, "executor": executor_name if has_executor else None, "has_executor": has_executor}
        )

        # ========== PHASE 1: Designer generates MAS code ==========
        designer_enable_thinking = self.agent_enable_thinking.get(designer_name, False)
        designer_enable_multimodal = self.agent_enable_multimodal.get(designer_name, False)
        
        # Step 1: Designer updates from environment to get prompt
        designer_agent.update_from_env(env)
        designer_prompt = designer_agent.current_prompt

        # Step 2: Format designer prompt for model
        print(f"[PRINT DEBUG] Converting designer prompt to DataProto...")
        print(f"[PRINT DEBUG] prompt type: {type(designer_prompt)}")
        print(f"[PRINT DEBUG] prompt keys: {list(designer_prompt.keys()) if isinstance(designer_prompt, dict) else 'N/A'}")
        if isinstance(designer_prompt, dict):
            print(f"[PRINT DEBUG] prompt['text'] length: {len(designer_prompt.get('text', ''))}")
            print(f"[PRINT DEBUG] prompt['system']: {designer_prompt.get('system', 'N/A')[:100]}")
        
        try:
            designer_format_prompt = convert_prompt_to_dpr(
                self.tokenizer_dict[designer_policy],
                self.processor_dict.get(designer_policy),
                designer_prompt,
                self.max_prompt_length,
                multi_modal=designer_enable_multimodal,
                enable_thinking=designer_enable_thinking
            )
            print(f"[PRINT DEBUG] designer_format_prompt created: {designer_format_prompt is not None}")
            if designer_format_prompt is not None:
                batch_exists = hasattr(designer_format_prompt, 'batch') and designer_format_prompt.batch is not None
                if batch_exists:
                    try:
                        batch_keys = list(designer_format_prompt.batch.keys())
                        print(f"[PRINT DEBUG] designer_format_prompt.batch keys: {batch_keys}")
                    except Exception as e:
                        print(f"[PRINT DEBUG] Error getting batch keys: {e}")
                else:
                    print(f"[PRINT DEBUG] designer_format_prompt.batch is None or doesn't exist")
        except Exception as e:
            print(f"[PRINT DEBUG] ========== ERROR in convert_prompt_to_dpr for designer ==========")
            print(f"[PRINT DEBUG] Exception: {e}")
            import traceback
            print(f"[PRINT DEBUG] Traceback: {traceback.format_exc()}")
            raise

        if designer_format_prompt is None:
            print(f"[PRINT DEBUG] ========== ERROR: designer_format_prompt is None ==========")
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 0, designer_name,
                "Failed to format designer prompt",
                {"error": "designer_format_prompt is None"}
            )
            return trajectory_per_task_dict

        # Step 3: Generate MAS code using Designer's LLM
        designer_ppo_config = self.ppo_trainer_config_dict.get(designer_policy, None)
        designer_model_path = designer_ppo_config.actor_rollout_ref.model.path if designer_ppo_config else None
        if designer_model_path:
            designer_model_name = "/".join(str(designer_model_path).split("/")[-2:])
        else:
            designer_model_name = designer_policy

        print(f"[PRINT DEBUG] ========== Designer MAS Generation Start (rollout_idx={rollout_idx}, env_idx={env_idx}) ==========")
        print(f"[PRINT DEBUG] designer_policy: {designer_policy}")
        print(f"[PRINT DEBUG] designer_model_name: {designer_model_name}")

        designer_output_dpr = None
        designer_response = None

        try:
            designer_addresses = self.server_address_dict.get(designer_policy)
            if isinstance(designer_addresses, (list, tuple)):
                designer_address = random.choice(designer_addresses) if len(designer_addresses) > 0 else designer_addresses[0]
            else:
                designer_address = designer_addresses

            designer_lora_id = None
            if self.lora_differ_mode and self.use_lora_for_generation and designer_name in self.agent_lora_mapping:
                designer_lora_id = self.agent_lora_mapping[designer_name]

            designer_config = self.agent_config_dict.get(designer_name, None)
            designer_sample_num = getattr(designer_config, 'sample_num', 1) if designer_config else 1

            designer_output_dpr, designer_response = await llm_async_generate(
                rollout_idx=rollout_idx,
                turn_idx=0,
                agent_idx=0,
                prompt_dpr=designer_format_prompt,
                ppo_trainer_config=designer_ppo_config,
                address=designer_address,
                model_name=designer_model_name,
                tokenizer=self.tokenizer_dict[designer_policy],
                enable_thinking=designer_enable_thinking,
                application_id=str(uuid.uuid4()),
                env_idx=env_idx,
                policy_name=designer_policy,
                timeout=self.generate_timeout,
                mode=self.mode,
                lora_id=designer_lora_id,
                agent_config=designer_config,
                sample_num=designer_sample_num,
            )
                
        except Exception as e:
            import traceback
            print(f"[PRINT DEBUG] ========== EXCEPTION in Designer MAS Generation ==========")
            print(f"[PRINT DEBUG] Exception: {e}")
            print(traceback.format_exc())
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 0, designer_name,
                f"Failed to generate designer response: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            designer_output_dpr = None
            designer_response = ""

        if designer_response is None:
            designer_response = ""

        # Step 4: Update designer agent with model response (extract code)
        designer_agent.update_from_model(designer_response)
        designer_code = designer_agent.generated_code if hasattr(designer_agent, 'generated_code') else ""
        
        print(f"[PRINT DEBUG] Designer generated code length: {len(designer_code)}")

        # ========== PHASE 2: Executor executes MAS code (or Designer executes if shared policy) ==========
        if has_executor:
            executor_enable_thinking = self.agent_enable_thinking.get(executor_name, False)
            executor_enable_multimodal = self.agent_enable_multimodal.get(executor_name, False)
            
            # Step 5: Executor updates from environment with designer's code
            executor_agent.update_from_env(env, designer_code=designer_code)
            executor_prompt =  executor_agent.current_prompt
        else:
            # Shared policy mode: Designer executes its own code
            executor_name = designer_name  # Use designer name for consistency
            executor_policy = designer_policy  # Use designer policy
            executor_agent = designer_agent  # Use designer agent
            executor_enable_thinking = designer_enable_thinking
            executor_enable_multimodal = designer_enable_multimodal
            
            # Designer already has env info, just set the code
            executor_agent.designer_code = designer_code
            executor_prompt = executor_agent.current_prompt
            #{"text": designer_code, "image": None, "system": "You are an expert in executing Multi-Agent System workflows."}
        
        # Step 6: Format executor prompt for model (but executor doesn't need to generate, just execute)
        # For executor, we still need to create a DataProto for training, but the prompt is the code itself
        try:
            executor_format_prompt = convert_prompt_to_dpr(
                self.tokenizer_dict[executor_policy],
                self.processor_dict.get(executor_policy),
                executor_prompt,
                self.max_prompt_length,
                multi_modal=executor_enable_multimodal,
                enable_thinking=False  # Executor should not use thinking
            )
        except Exception as e:
            print(f"[PRINT DEBUG] ========== ERROR in convert_prompt_to_dpr for executor ==========")
            print(f"[PRINT DEBUG] Exception: {e}")
            import traceback
            print(traceback.format_exc())
            executor_format_prompt = None

        # Step 7: Generate executor response (though it's mainly for creating DataProto)
        executor_ppo_config = self.ppo_trainer_config_dict.get(executor_policy, None)
        executor_model_path = executor_ppo_config.actor_rollout_ref.model.path if executor_ppo_config else None
        if executor_model_path:
            executor_model_name = "/".join(str(executor_model_path).split("/")[-2:])
        else:
            executor_model_name = executor_policy

        executor_output_dpr = None
        executor_response = ""

        # In shared policy mode, skip executor LLM generation (Designer already generated everything)
        if not has_executor:
            # Use designer's output for executor in shared policy mode
            executor_output_dpr = designer_output_dpr
            executor_response = designer_response if hasattr(designer_agent, 'generated_code') else ""
        elif executor_format_prompt is not None:
            try:
                executor_addresses = self.server_address_dict.get(executor_policy)
                if isinstance(executor_addresses, (list, tuple)):
                    executor_address = random.choice(executor_addresses) if len(executor_addresses) > 0 else executor_addresses[0]
                else:
                    executor_address = executor_addresses

                executor_lora_id = None
                if self.lora_differ_mode and self.use_lora_for_generation and executor_name in self.agent_lora_mapping:
                    executor_lora_id = self.agent_lora_mapping[executor_name]

                executor_config = self.agent_config_dict.get(executor_name, None)
                executor_sample_num = getattr(executor_config, 'sample_num', 1) if executor_config else 1

                executor_output_dpr, executor_response = await llm_async_generate(
                    rollout_idx=rollout_idx,
                    turn_idx=1,
                    agent_idx=1,
                    prompt_dpr=executor_format_prompt,
                    ppo_trainer_config=executor_ppo_config,
                    address=executor_address,
                    model_name=executor_model_name,
                    tokenizer=self.tokenizer_dict[executor_policy],
                    enable_thinking=False,  # Executor should not use thinking
                    application_id=str(uuid.uuid4()),
                    env_idx=env_idx,
                    policy_name=executor_policy,
                    timeout=self.generate_timeout,
                    mode=self.mode,
                    lora_id=executor_lora_id,
                    agent_config=executor_config,
                    sample_num=executor_sample_num,
                )
            except Exception as e:
                import traceback
                print(f"[PRINT DEBUG] ========== EXCEPTION in Executor Generation ==========")
                print(f"[PRINT DEBUG] Exception: {e}")
                print(traceback.format_exc())
                executor_output_dpr = None
                executor_response = ""

        if executor_response is None:
            executor_response = ""

        # Step 8: Update executor agent (though it mainly uses designer's code)
        # In shared policy mode, executor_agent is the same as designer_agent, so this is a no-op
        if has_executor:
            executor_agent.update_from_model(executor_response)
        else:
            # In shared policy mode, executor_agent is designer_agent, already updated
            pass

        # Step 9: Execute MAS code via executor's step() method
        final_reward = 0.0
        workflow_dataproto_list = []
        mas_execution_success = False
        
        try:
            env_worker_id = rollout_idx % self.num_workers
            env_worker = self.env_workers[env_worker_id]

            if hasattr(env, 'state'):
                env.state.assigned_worker_id = env_worker_id
                env.state.gpu_group_id = self.gpu_group_id

            # Prepare output directory for MAS execution
            import os
            benchmark_name = getattr(self.config.env, 'benchmark', '') if hasattr(self.config, 'env') else ''
            if self.mode == "validate" and benchmark_name:
                output_base_dir = os.path.join('./tmp_auto_mas', self.experiment_name, self.mode, str(benchmark_name))
            else:
                output_base_dir = os.path.join('./tmp_auto_mas', self.experiment_name, self.mode)
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
                    self.mode, env_idx, rollout_idx, 1, executor_name,
                    f"Failed to create output directory: {e}",
                    {"output_dir": output_dir, "error": str(e)}
                )
                raise

            # Get tokenizer_path for executor
            executor_tokenizer_path = self.tokenizer_path_dict.get(executor_policy)

            logger.info(f"Calling executor step() with tokenizer_path: {executor_tokenizer_path}")

            # Step 10: Execute MAS code using executor's step() method
            # IMPORTANT: Use designer's server_address and model_name for AIClient
            # This ensures the MAS sub-agents use the designer's model
            print(f"[EXECUTOR STEP] Starting MAS execution for rollout {rollout_idx}")
            print(f"[EXECUTOR STEP] Using designer's model: server_address={designer_address}, model_name={designer_model_name}")
            print(f"[EXECUTOR STEP] Executor tokenizer_path: {executor_tokenizer_path}")
            
            step_result = await asyncio.wait_for(
                executor_agent.step(
                    env_data=env,
                    output_dir=output_dir,
                    server_address=designer_address,  # Use designer's server address for AIClient
                    model_name=designer_model_name,   # Use designer's model name for AIClient
                    tokenizer_path=executor_tokenizer_path,  # But use executor's tokenizer_path if different
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length,
                    ppo_trainer_config=executor_ppo_config,
                    enable_thinking=False,  # Executor should not use thinking
                    step_timeout=self.step_timeout,
                    env_worker=env_worker
                ),
                timeout=self.step_timeout + 60.0
            )

            # Extract results from step
            workflow_dataproto_list = step_result.get("workflow_dpr", [])
            mas_execution_success = step_result.get("execution_success", False)
            final_reward = step_result.get("reward", 0.0)
            designer_reward = step_result.get("designer_reward", final_reward)

            print(f"[EXECUTOR STEP RESULT] MAS execution success: {mas_execution_success}")
            print(f"[EXECUTOR STEP RESULT] Agent reward: {final_reward}, Designer reward: {designer_reward}")
            print(f"[EXECUTOR STEP RESULT] Number of workflow DataProtos: {len(workflow_dataproto_list)}")

            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, executor_name,
                "MAS execution completed",
                {
                    "mas_execution_success": mas_execution_success,
                    "final_reward": final_reward,
                    "num_dataprotos": len(workflow_dataproto_list) if workflow_dataproto_list else 0,
                    "output_dir": output_dir
                }
            )

        except asyncio.TimeoutError:
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, executor_name,
                "MAS execution timed out",
                {"error": "timeout"}
            )
            mas_execution_success = False
            workflow_dataproto_list = []
            final_reward = 0.0
        except Exception as e:
            import traceback
            self.multi_logger.log_env_agent_info(
                self.mode, env_idx, rollout_idx, 1, executor_name,
                f"MAS execution failed: {e}",
                {"error": str(e), "traceback": traceback.format_exc()}
            )
            mas_execution_success = False
            workflow_dataproto_list = []
            final_reward = 0.0
        finally:
            # Additional cleanup for any lingering resources after step
            try:
                await self._cleanup_after_step(rollout_idx)
            except Exception as cleanup_err:
                logger.debug(f"Cleanup warning for rollout {rollout_idx}: {cleanup_err}")
            

        # Step 11: Merge all DataProtos and assign outcome rewards
        # Both Designer and Executor use the same outcome reward (final_reward)
        
        # Designer's DataProto - only correctness + format reward (no penalties)
        if designer_output_dpr is not None:
            designer_batch_size = len(designer_output_dpr)
            designer_output_dpr.non_tensor_batch["reward"] = np.array([designer_reward] * designer_batch_size)
            designer_output_dpr.non_tensor_batch["agent_name"] = np.array([designer_name] * designer_batch_size, dtype=object)
            designer_output_dpr.non_tensor_batch["env_final_reward"] = np.array([designer_reward] * designer_batch_size)
            designer_output_dpr.non_tensor_batch["turn_idx"] = np.array([0] * designer_batch_size)
            designer_output_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * designer_batch_size)
            designer_output_dpr.non_tensor_batch["rollout_idx"] = np.array([rollout_idx] * designer_batch_size)
            designer_output_dpr.non_tensor_batch["agent_idx"] = np.array([0] * designer_batch_size)

            if self.lora_differ_mode and designer_name in self.agent_lora_mapping:
                designer_lora_ids = [self.agent_lora_mapping[designer_name]] * designer_batch_size
                designer_output_dpr.non_tensor_batch["lora_ids"] = np.array(designer_lora_ids, dtype=object)

            if trajectory_per_task_dict[designer_policy].batch is None:
                trajectory_per_task_dict[designer_policy] = designer_output_dpr
            else:
                _align_non_tensor_batch_keys(trajectory_per_task_dict[designer_policy], designer_output_dpr)
                trajectory_per_task_dict[designer_policy] = DataProto.concat([
                    trajectory_per_task_dict[designer_policy],
                    designer_output_dpr
                ])

        # Executor's DataProto - reward based on final outcome
        # In shared policy mode, executor_output_dpr may be None or same as designer_output_dpr
        # Only add executor DataProto if it's different from designer (split policy mode)
        if executor_output_dpr is not None and has_executor:
            executor_batch_size = len(executor_output_dpr)
            executor_output_dpr.non_tensor_batch["reward"] = np.array([final_reward] * executor_batch_size)
            executor_output_dpr.non_tensor_batch["agent_name"] = np.array([executor_name] * executor_batch_size, dtype=object)
            executor_output_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward] * executor_batch_size)
            executor_output_dpr.non_tensor_batch["turn_idx"] = np.array([1] * executor_batch_size)
            executor_output_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * executor_batch_size)
            executor_output_dpr.non_tensor_batch["rollout_idx"] = np.array([rollout_idx] * executor_batch_size)
            executor_output_dpr.non_tensor_batch["agent_idx"] = np.array([1] * executor_batch_size)

            if self.lora_differ_mode and executor_name in self.agent_lora_mapping:
                executor_lora_ids = [self.agent_lora_mapping[executor_name]] * executor_batch_size
                executor_output_dpr.non_tensor_batch["lora_ids"] = np.array(executor_lora_ids, dtype=object)

            if trajectory_per_task_dict[executor_policy].batch is None:
                trajectory_per_task_dict[executor_policy] = executor_output_dpr
            else:
                _align_non_tensor_batch_keys(trajectory_per_task_dict[executor_policy], executor_output_dpr)
                trajectory_per_task_dict[executor_policy] = DataProto.concat([
                    trajectory_per_task_dict[executor_policy],
                    executor_output_dpr
                ])

        # Workflow DataProtos from MAS workflow execution (assigned to executor)
        # Use "WorkflowAgent" as agent_name to distinguish from Designer's own DataProto
        workflow_agent_name = "WorkflowAgent" if not has_executor else executor_name
        if workflow_dataproto_list:
            for workflow_dpr in workflow_dataproto_list:
                batch_size = len(workflow_dpr)
                workflow_dpr.non_tensor_batch["reward"] = np.array([final_reward] * batch_size)
                workflow_dpr.non_tensor_batch["agent_name"] = np.array([workflow_agent_name] * batch_size, dtype=object)
                workflow_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward] * batch_size)
                workflow_dpr.non_tensor_batch["turn_idx"] = np.array([1] * batch_size)
                workflow_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * batch_size)
                workflow_dpr.non_tensor_batch["rollout_idx"] = np.array([rollout_idx] * batch_size)
                workflow_dpr.non_tensor_batch["agent_idx"] = np.array([1] * batch_size)

                if self.lora_differ_mode and executor_name in self.agent_lora_mapping:
                    executor_lora_ids = [self.agent_lora_mapping[executor_name]] * batch_size
                    workflow_dpr.non_tensor_batch["lora_ids"] = np.array(executor_lora_ids, dtype=object)

                if trajectory_per_task_dict[executor_policy].batch is None:
                    trajectory_per_task_dict[executor_policy] = workflow_dpr
                else:
                    _align_non_tensor_batch_keys(trajectory_per_task_dict[executor_policy], workflow_dpr)
                    trajectory_per_task_dict[executor_policy] = DataProto.concat([
                        trajectory_per_task_dict[executor_policy],
                        workflow_dpr
                    ])

            logger.info(f"Added {len(workflow_dataproto_list)} workflow DataProto entries for rollout {rollout_idx}")

        # Step 12: Log results
        env_state_compact = env.state.to_dict_compact(agent_name=designer_name) if hasattr(env.state, 'to_dict_compact') else env.state

        self.multi_logger.log_env_agent_info(
            self.mode, env_idx, rollout_idx, 0, designer_name,
            "Designer MAS generation completed",
            {
                "agent_prompt": {"text": designer_prompt.get("text", "") if isinstance(designer_prompt, dict) else str(designer_prompt), "image": None},
                "agent_response": designer_response,
                "generated_code_length": len(designer_code),
                "reward": final_reward
            }
        )

        self.multi_logger.log_env_agent_info(
            self.mode, env_idx, rollout_idx, 1, executor_name,
            "Executor MAS execution completed",
            {
                "agent_prompt": {"text": executor_prompt.get("text", "")[:500] if isinstance(executor_prompt, dict) else str(executor_prompt)[:500], "image": None},
                "agent_response": executor_response,
                "env_state": env_state_compact,
                "reward": final_reward
            }
        )

        # Step 13: Log rollout summary
        agent_rewards = {designer_name: final_reward, executor_name: final_reward}
        self.multi_logger.log_rollout_summary(
            self.mode, env_idx, rollout_idx, agent_rewards,
            "rollout_complete",
            extra_data={
                "turn_idx": 1,
                "message": f"Rollout {rollout_idx} completed with split policy",
                "reward": final_reward,
                "designer": designer_name,
                "executor": executor_name
            }
        )

        if self.mode == "validate":
            # Consider successful if reward > 0.5
            if final_reward > 0.5:
                if designer_name in self.success_rollout_idx_list_dict:
                    self.success_rollout_idx_list_dict[designer_name].append(rollout_idx)
                if executor_name in self.success_rollout_idx_list_dict:
                    self.success_rollout_idx_list_dict[executor_name].append(rollout_idx)
                if designer_name in self.success_ave_turn_dict:
                    self.success_ave_turn_dict[designer_name] += 1
                if executor_name in self.success_ave_turn_dict:
                    self.success_ave_turn_dict[executor_name] += 1
                env.success = True
            else:
                env.success = False
        
       
        #trajectory_per_task_dict = self._assign_consistent_uids(trajectory_per_task_dict)
        
        # record latency for this rollout
        try:
            
            latency_s = time.perf_counter() - start_time
            self.rollout_latency_dict[rollout_idx] = {"latency_s": latency_s, "reward": final_reward}
            self.multi_logger.log_async_event(
                self.mode, env_idx, rollout_idx, "rollout_latency",
                f"Rollout {rollout_idx} latency: {latency_s:.3f}s",
                {"latency_s": float(latency_s), "reward": float(final_reward)}
            )
        except Exception:
            pass

        # Debug: log what this rollout is returning
        for policy_name, policy_data in trajectory_per_task_dict.items():
            if policy_data.batch is not None:
                rollout_len = len(policy_data)
                rewards = policy_data.non_tensor_batch.get("reward", [])
                print(f"[DEBUG RETURN] Rollout {rollout_idx} returning {rollout_len} samples for {policy_name}, rewards={rewards[:5] if len(rewards) > 0 else []}")
            else:
                print(f"[DEBUG RETURN] Rollout {rollout_idx} returning EMPTY DataProto for {policy_name} (batch=None)")

        return trajectory_per_task_dict
            


    async def _generate_single_design(self, env_idx, design_idx):
        """
        Phase 1 of tree design: Generate a single MAS design using the Designer LLM.
        Extracted from generate_single_rollout Phase 1.

        Args:
            env_idx: Index of the environment/problem
            design_idx: Index of the design (0 to design_sample_num-1)

        Returns:
            dict with designer_code, designer_output_dpr, designer_response, designer_reward_base
        """
        N, M = self.design_sample_num, self.execute_sample_num
        rollout_idx_list = self.env_rollout_mapping[env_idx]
        # Use the first rollout_idx for this design as the canonical index
        canonical_rollout_idx = rollout_idx_list[design_idx * M]

        env = self.envs[canonical_rollout_idx]
        agent_group = self.agent_groups_list[canonical_rollout_idx]

        designer_name = self.turn_order[0] if len(self.turn_order) > 0 else "Designer"
        designer_policy = self.agent_policy_mapping.get(designer_name)
        designer_agent = agent_group[0] if len(agent_group) > 0 else None

        if designer_agent is None:
            raise RuntimeError(f"Missing designer agent for env_idx={env_idx}, design_idx={design_idx}")

        designer_enable_thinking = self.agent_enable_thinking.get(designer_name, False)
        designer_enable_multimodal = self.agent_enable_multimodal.get(designer_name, False)

        # Step 1: Designer updates from environment to get prompt
        designer_agent.update_from_env(env)
        designer_prompt = designer_agent.current_prompt

        # Step 2: Format designer prompt for model
        designer_format_prompt = convert_prompt_to_dpr(
            self.tokenizer_dict[designer_policy],
            self.processor_dict.get(designer_policy),
            designer_prompt,
            self.max_prompt_length,
            multi_modal=designer_enable_multimodal,
            enable_thinking=designer_enable_thinking
        )

        if designer_format_prompt is None:
            raise RuntimeError(f"Failed to format designer prompt for env_idx={env_idx}, design_idx={design_idx}")

        # Step 3: Generate MAS code using Designer's LLM
        designer_ppo_config = self.ppo_trainer_config_dict.get(designer_policy, None)
        designer_model_path = designer_ppo_config.actor_rollout_ref.model.path if designer_ppo_config else None
        if designer_model_path:
            designer_model_name = "/".join(str(designer_model_path).split("/")[-2:])
        else:
            designer_model_name = designer_policy

        designer_addresses = self.server_address_dict.get(designer_policy)
        if isinstance(designer_addresses, (list, tuple)):
            designer_address = random.choice(designer_addresses) if len(designer_addresses) > 0 else designer_addresses[0]
        else:
            designer_address = designer_addresses

        designer_lora_id = None
        if self.lora_differ_mode and self.use_lora_for_generation and designer_name in self.agent_lora_mapping:
            designer_lora_id = self.agent_lora_mapping[designer_name]

        designer_config = self.agent_config_dict.get(designer_name, None)
        designer_sample_num_cfg = getattr(designer_config, 'sample_num', 1) if designer_config else 1

        designer_output_dpr, designer_response = await llm_async_generate(
            rollout_idx=canonical_rollout_idx,
            turn_idx=0,
            agent_idx=0,
            prompt_dpr=designer_format_prompt,
            ppo_trainer_config=designer_ppo_config,
            address=designer_address,
            model_name=designer_model_name,
            tokenizer=self.tokenizer_dict[designer_policy],
            enable_thinking=designer_enable_thinking,
            application_id=str(uuid.uuid4()),
            env_idx=env_idx,
            policy_name=designer_policy,
            timeout=self.generate_timeout,
            mode=self.mode,
            lora_id=designer_lora_id,
            agent_config=designer_config,
            sample_num=designer_sample_num_cfg,
        )

        if designer_response is None:
            designer_response = ""

        # Step 4: Update designer agent with model response (extract code)
        designer_agent.update_from_model(designer_response)
        designer_code = designer_agent.generated_code if hasattr(designer_agent, 'generated_code') else ""

        print(f"[TREE DESIGN] env_idx={env_idx}, design_idx={design_idx}: Designer generated code length={len(designer_code)}")

        return {
            "designer_code": designer_code,
            "designer_output_dpr": designer_output_dpr,
            "designer_response": designer_response,
            "designer_address": designer_address,
            "designer_model_name": designer_model_name,
            "designer_prompt": designer_prompt,
        }

    async def _execute_single_design(self, env_idx, rollout_idx, design_idx, exec_idx, design_result):
        """
        Phase 2 of tree design: Execute a single MAS design once.
        Extracted from generate_single_rollout Phase 2.

        Args:
            env_idx: Index of the environment/problem
            rollout_idx: Global rollout index for this execution
            design_idx: Which design this execution belongs to
            exec_idx: Execution index within this design (0 to execute_sample_num-1)
            design_result: Dict from _generate_single_design containing designer_code, etc.

        Returns:
            dict with reward, executor_output_dpr, workflow_dataproto_list, designer_output_dpr_ref
        """
        designer_code = design_result["designer_code"]
        designer_address = design_result["designer_address"]
        designer_model_name = design_result["designer_model_name"]
        # Note: We no longer deep copy here to save memory. The designer_output_dpr
        # is only used read-only during execution; the single deep copy happens in
        # generate_tree_rollout Phase 3 when assembling the training batch.
        designer_output_dpr_ref = design_result["designer_output_dpr"]

        env = self.envs[rollout_idx]
        agent_group = self.agent_groups_list[rollout_idx]

        designer_name = self.turn_order[0] if len(self.turn_order) > 0 else "Designer"
        executor_name = self.turn_order[1] if len(self.turn_order) > 1 else None
        designer_policy = self.agent_policy_mapping.get(designer_name)
        executor_policy = self.agent_policy_mapping.get(executor_name) if executor_name else designer_policy

        designer_agent = agent_group[0] if len(agent_group) > 0 else None
        executor_agent = agent_group[1] if len(agent_group) > 1 else None
        has_executor = executor_agent is not None and executor_name is not None

        # ========== Executor Phase ==========
        if has_executor:
            executor_enable_thinking = self.agent_enable_thinking.get(executor_name, False)
            executor_enable_multimodal = self.agent_enable_multimodal.get(executor_name, False)
            executor_agent.update_from_env(env, designer_code=designer_code)
            executor_prompt = executor_agent.current_prompt
        else:
            executor_name = designer_name
            executor_policy = designer_policy
            executor_agent = designer_agent
            executor_enable_thinking = self.agent_enable_thinking.get(designer_name, False)
            executor_enable_multimodal = self.agent_enable_multimodal.get(designer_name, False)
            executor_agent.update_from_env(env)
            executor_agent.designer_code = designer_code
            executor_prompt = executor_agent.current_prompt

        # Format executor prompt
        try:
            executor_format_prompt = convert_prompt_to_dpr(
                self.tokenizer_dict[executor_policy],
                self.processor_dict.get(executor_policy),
                executor_prompt,
                self.max_prompt_length,
                multi_modal=executor_enable_multimodal,
                enable_thinking=False
            )
        except Exception as e:
            print(f"[TREE DESIGN] Error formatting executor prompt for rollout {rollout_idx}: {e}")
            executor_format_prompt = None

        # Generate executor response
        executor_ppo_config = self.ppo_trainer_config_dict.get(executor_policy, None)
        executor_model_path = executor_ppo_config.actor_rollout_ref.model.path if executor_ppo_config else None
        if executor_model_path:
            executor_model_name = "/".join(str(executor_model_path).split("/")[-2:])
        else:
            executor_model_name = executor_policy

        executor_output_dpr = None
        executor_response = ""

        if not has_executor:
            executor_output_dpr = designer_output_dpr_ref
            executor_response = design_result["designer_response"]
        elif executor_format_prompt is not None:
            try:
                executor_addresses = self.server_address_dict.get(executor_policy)
                if isinstance(executor_addresses, (list, tuple)):
                    executor_address = random.choice(executor_addresses) if len(executor_addresses) > 0 else executor_addresses[0]
                else:
                    executor_address = executor_addresses

                executor_lora_id = None
                if self.lora_differ_mode and self.use_lora_for_generation and executor_name in self.agent_lora_mapping:
                    executor_lora_id = self.agent_lora_mapping[executor_name]

                executor_config = self.agent_config_dict.get(executor_name, None)
                executor_sample_num_cfg = getattr(executor_config, 'sample_num', 1) if executor_config else 1

                executor_output_dpr, executor_response = await llm_async_generate(
                    rollout_idx=rollout_idx,
                    turn_idx=1,
                    agent_idx=1,
                    prompt_dpr=executor_format_prompt,
                    ppo_trainer_config=executor_ppo_config,
                    address=executor_address,
                    model_name=executor_model_name,
                    tokenizer=self.tokenizer_dict[executor_policy],
                    enable_thinking=False,
                    application_id=str(uuid.uuid4()),
                    env_idx=env_idx,
                    policy_name=executor_policy,
                    timeout=self.generate_timeout,
                    mode=self.mode,
                    lora_id=executor_lora_id,
                    agent_config=executor_config,
                    sample_num=executor_sample_num_cfg,
                )
            except Exception as e:
                import traceback
                print(f"[TREE DESIGN] Executor generation failed for rollout {rollout_idx}: {e}")
                print(traceback.format_exc())
                executor_output_dpr = None
                executor_response = ""

        if executor_response is None:
            executor_response = ""

        if has_executor:
            executor_agent.update_from_model(executor_response)
        else:
            # Shared model mode: must call update_from_model to set current_action/generated_code
            # on this agent instance (which may differ from the one used in _generate_single_design)
            executor_agent.update_from_model(design_result["designer_response"])

        # Execute MAS code
        final_reward = 0.0
        designer_reward = 0.0
        workflow_dataproto_list = []
        mas_execution_success = False

        try:
            env_worker_id = rollout_idx % self.num_workers
            env_worker = self.env_workers[env_worker_id]

            if hasattr(env, 'state'):
                env.state.assigned_worker_id = env_worker_id
                env.state.gpu_group_id = self.gpu_group_id

            benchmark_name = getattr(self.config.env, 'benchmark', '') if hasattr(self.config, 'env') else ''
            if self.mode == "validate" and benchmark_name:
                output_base_dir = os.path.join('./tmp_auto_mas', self.experiment_name, self.mode, str(benchmark_name))
            else:
                output_base_dir = os.path.join('./tmp_auto_mas', self.experiment_name, self.mode)
            output_dir = os.path.abspath(os.path.join(output_base_dir, f'rollout_{rollout_idx}'))
            os.makedirs(output_dir, exist_ok=True)

            executor_tokenizer_path = self.tokenizer_path_dict.get(executor_policy)

            step_result = await asyncio.wait_for(
                executor_agent.step(
                    env_data=env,
                    output_dir=output_dir,
                    server_address=designer_address,
                    model_name=designer_model_name,
                    tokenizer_path=executor_tokenizer_path,
                    max_prompt_length=self.max_prompt_length,
                    max_response_length=self.max_response_length,
                    ppo_trainer_config=executor_ppo_config,
                    enable_thinking=False,
                    step_timeout=self.step_timeout,
                    env_worker=env_worker
                ),
                timeout=self.step_timeout + 60.0
            )

            workflow_dataproto_list = step_result.get("workflow_dpr", [])
            mas_execution_success = step_result.get("execution_success", False)
            final_reward = step_result.get("reward", 0.0)
            designer_reward = step_result.get("designer_reward", final_reward)

            print(f"[TREE DESIGN] rollout_idx={rollout_idx} (design={design_idx}, exec={exec_idx}): "
                  f"reward={final_reward}, designer_reward={designer_reward}, success={mas_execution_success}")

        except asyncio.TimeoutError:
            print(f"[TREE DESIGN] MAS execution timed out for rollout {rollout_idx}")
            mas_execution_success = False
            workflow_dataproto_list = []
            final_reward = 0.0
            designer_reward = 0.0
        except Exception as e:
            import traceback
            print(f"[TREE DESIGN] MAS execution failed for rollout {rollout_idx}: {e}")
            print(traceback.format_exc())
            mas_execution_success = False
            workflow_dataproto_list = []
            final_reward = 0.0
            designer_reward = 0.0
        finally:
            try:
                await self._cleanup_after_step(rollout_idx)
            except Exception:
                pass

        return {
            "reward": final_reward,
            "designer_reward": designer_reward,
            "executor_output_dpr": executor_output_dpr,
            "workflow_dataproto_list": workflow_dataproto_list,
            "designer_output_dpr_ref": designer_output_dpr_ref,
            "mas_execution_success": mas_execution_success,
            "executor_name": executor_name,
            "executor_policy": executor_policy,
            "executor_response": executor_response,
            "executor_prompt": executor_prompt,
            "has_executor": has_executor,
        }

    async def generate_tree_rollout(self, env_idx):
        """
        Tree design sampling: Generate N designs, each executed M times.
        Designer GRPO groups over N designs (reward = mean of M executions).
        Executor GRPO groups over M executions within each design.

        Args:
            env_idx: Index of the environment/problem

        Returns:
            dict mapping policy_name -> DataProto
        """
        N, M = self.design_sample_num, self.execute_sample_num
        rollout_idx_list = self.env_rollout_mapping[env_idx]
        start_time = time.perf_counter()

        trajectory_per_task_dict = {}
        for policy_name in self.tokenizer_dict.keys():
            trajectory_per_task_dict[policy_name] = DataProto()

        designer_name = self.turn_order[0] if len(self.turn_order) > 0 else "Designer"
        executor_name = self.turn_order[1] if len(self.turn_order) > 1 else None
        designer_policy = self.agent_policy_mapping.get(designer_name)
        executor_policy = self.agent_policy_mapping.get(executor_name) if executor_name else designer_policy
        has_executor = executor_name is not None

        self.multi_logger.log_async_event(
            self.mode, env_idx, -1, "tree_design_start",
            f"Starting tree design for env_idx={env_idx}: {N} designs × {M} executions",
            {"env_idx": env_idx, "design_sample_num": N, "execute_sample_num": M}
        )

        # ========== Phase 1: Generate N designs in parallel ==========
        designs = await asyncio.gather(*[
            self._generate_single_design(env_idx, d) for d in range(N)
        ], return_exceptions=True)

        # ========== Phase 2: Execute each design M times in parallel ==========
        # Limit concurrent MAS executions to avoid overwhelming vLLM server
        mas_semaphore = asyncio.Semaphore(self.mas_concurrency)

        async def _limited_execute(env_idx, ridx, d, m, design):
            async with mas_semaphore:
                return await self._execute_single_design(env_idx, ridx, d, m, design)

        exec_tasks = []
        exec_task_meta = []  # Track (design_idx, exec_idx) for each task
        for d, design in enumerate(designs):
            if isinstance(design, Exception):
                print(f"[TREE DESIGN] Design {d} for env_idx={env_idx} failed: {design}")
                continue
            for m in range(M):
                ridx = rollout_idx_list[d * M + m]
                exec_tasks.append(
                    _limited_execute(env_idx, ridx, d, m, design)
                )
                exec_task_meta.append((d, m, ridx))

        exec_results = await asyncio.gather(*exec_tasks, return_exceptions=True)

        # ========== Phase 3: Compute designer mean reward, assemble DataProtos ==========
        # Group execution results by design_idx
        design_exec_results = {}  # design_idx -> list of (exec_idx, ridx, result)
        for (d, m, ridx), result in zip(exec_task_meta, exec_results):
            if isinstance(result, Exception):
                print(f"[TREE DESIGN] Execution d={d},m={m} for env_idx={env_idx} failed: {result}")
                continue
            if d not in design_exec_results:
                design_exec_results[d] = []
            design_exec_results[d].append((m, ridx, result))

        # Collect all DataProtos per policy in lists, then concat once at the end
        # to avoid O(n^2) incremental concatenation.
        collected_dprs = {}  # policy_name -> list of DataProto

        for d in range(N):
            design = designs[d]
            if isinstance(design, Exception):
                continue

            exec_list = design_exec_results.get(d, [])
            if not exec_list:
                continue

            # Compute designer mean reward from M executions
            exec_rewards = [r["designer_reward"] for (_, _, r) in exec_list]
            designer_mean_reward = float(np.mean(exec_rewards))
            print(f"[TREE DESIGN DEBUG] env_idx={env_idx}, design={d}: "
                  f"exec_rewards={exec_rewards}, designer_mean_reward={designer_mean_reward}, "
                  f"num_exec={len(exec_list)}/{M}")

            # --- Designer DataProto: only 1 per design (not M copies) ---
            designer_output_dpr = design["designer_output_dpr"]
            if designer_output_dpr is not None:
                designer_dpr = copy.deepcopy(designer_output_dpr)
                designer_batch_size = len(designer_dpr)
                canonical_ridx = rollout_idx_list[d * M]
                designer_dpr.non_tensor_batch["reward"] = np.array([designer_mean_reward] * designer_batch_size)
                designer_dpr.non_tensor_batch["agent_name"] = np.array([designer_name] * designer_batch_size, dtype=object)
                designer_dpr.non_tensor_batch["env_final_reward"] = np.array([designer_mean_reward] * designer_batch_size)
                designer_dpr.non_tensor_batch["turn_idx"] = np.array([0] * designer_batch_size)
                designer_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * designer_batch_size)
                designer_dpr.non_tensor_batch["rollout_idx"] = np.array([canonical_ridx] * designer_batch_size)
                designer_dpr.non_tensor_batch["agent_idx"] = np.array([0] * designer_batch_size)
                designer_dpr.non_tensor_batch["design_idx"] = np.array([d] * designer_batch_size)

                if self.lora_differ_mode and designer_name in self.agent_lora_mapping:
                    designer_dpr.non_tensor_batch["lora_ids"] = np.array(
                        [self.agent_lora_mapping[designer_name]] * designer_batch_size, dtype=object
                    )

                collected_dprs.setdefault(designer_policy, []).append(designer_dpr)

            # --- Executor DataProtos: one per execution ---
            for (m, ridx, result) in exec_list:
                exec_reward = result["reward"]
                exec_executor_name = result["executor_name"]
                exec_executor_policy = result["executor_policy"]
                exec_has_executor = result["has_executor"]
                executor_output_dpr = result["executor_output_dpr"]
                workflow_dataproto_list = result["workflow_dataproto_list"]

                if executor_output_dpr is not None and exec_has_executor:
                    executor_batch_size = len(executor_output_dpr)
                    executor_output_dpr.non_tensor_batch["reward"] = np.array([exec_reward] * executor_batch_size)
                    executor_output_dpr.non_tensor_batch["agent_name"] = np.array([exec_executor_name] * executor_batch_size, dtype=object)
                    executor_output_dpr.non_tensor_batch["env_final_reward"] = np.array([exec_reward] * executor_batch_size)
                    executor_output_dpr.non_tensor_batch["turn_idx"] = np.array([1] * executor_batch_size)
                    executor_output_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * executor_batch_size)
                    executor_output_dpr.non_tensor_batch["rollout_idx"] = np.array([ridx] * executor_batch_size)
                    executor_output_dpr.non_tensor_batch["agent_idx"] = np.array([1] * executor_batch_size)
                    executor_output_dpr.non_tensor_batch["design_idx"] = np.array([d] * executor_batch_size)

                    if self.lora_differ_mode and exec_executor_name in self.agent_lora_mapping:
                        executor_output_dpr.non_tensor_batch["lora_ids"] = np.array(
                            [self.agent_lora_mapping[exec_executor_name]] * executor_batch_size, dtype=object
                        )

                    collected_dprs.setdefault(exec_executor_policy, []).append(executor_output_dpr)

                # Workflow DataProtos from MAS workflow execution
                # Use "WorkflowAgent" as agent_name to distinguish from Designer's own DataProto
                workflow_agent_name = "WorkflowAgent" if not exec_has_executor else exec_executor_name
                if workflow_dataproto_list:
                    for workflow_dpr in workflow_dataproto_list:
                        batch_size = len(workflow_dpr)
                        workflow_dpr.non_tensor_batch["reward"] = np.array([exec_reward] * batch_size)
                        workflow_dpr.non_tensor_batch["agent_name"] = np.array([workflow_agent_name] * batch_size, dtype=object)
                        workflow_dpr.non_tensor_batch["env_final_reward"] = np.array([exec_reward] * batch_size)
                        workflow_dpr.non_tensor_batch["turn_idx"] = np.array([1] * batch_size)
                        workflow_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * batch_size)
                        workflow_dpr.non_tensor_batch["rollout_idx"] = np.array([ridx] * batch_size)
                        workflow_dpr.non_tensor_batch["agent_idx"] = np.array([1] * batch_size)
                        workflow_dpr.non_tensor_batch["design_idx"] = np.array([d] * batch_size)

                        if self.lora_differ_mode and exec_executor_name in self.agent_lora_mapping:
                            workflow_dpr.non_tensor_batch["lora_ids"] = np.array(
                                [self.agent_lora_mapping[exec_executor_name]] * batch_size, dtype=object
                            )

                        collected_dprs.setdefault(exec_executor_policy, []).append(workflow_dpr)

        # Batch concat: single O(n) concatenation per policy instead of O(n^2) incremental
        for policy_name, dpr_list in collected_dprs.items():
            if not dpr_list:
                continue
            # Align keys across all collected DataProtos
            for i in range(1, len(dpr_list)):
                _align_non_tensor_batch_keys(dpr_list[0], dpr_list[i])
            merged = DataProto.concat(dpr_list) if len(dpr_list) > 1 else dpr_list[0]
            if trajectory_per_task_dict[policy_name].batch is None:
                trajectory_per_task_dict[policy_name] = merged
            else:
                _align_non_tensor_batch_keys(trajectory_per_task_dict[policy_name], merged)
                trajectory_per_task_dict[policy_name] = DataProto.concat([
                    trajectory_per_task_dict[policy_name], merged
                ])

                # Validation success tracking
                if self.mode == "validate":
                    env_obj = self.envs[ridx]
                    if exec_reward > 0.5:
                        if designer_name in self.success_rollout_idx_list_dict:
                            self.success_rollout_idx_list_dict[designer_name].append(ridx)
                        if exec_executor_name in self.success_rollout_idx_list_dict:
                            self.success_rollout_idx_list_dict[exec_executor_name].append(ridx)
                        if designer_name in self.success_ave_turn_dict:
                            self.success_ave_turn_dict[designer_name] += 1
                        if exec_executor_name in self.success_ave_turn_dict:
                            self.success_ave_turn_dict[exec_executor_name] += 1
                        env_obj.success = True
                    else:
                        env_obj.success = False

        # Record latency
        try:
            latency_s = time.perf_counter() - start_time
            self.multi_logger.log_async_event(
                self.mode, env_idx, -1, "tree_rollout_latency",
                f"Tree rollout for env_idx={env_idx} latency: {latency_s:.3f}s",
                {"latency_s": float(latency_s), "num_designs": N, "num_executions": M}
            )
        except Exception:
            pass

        # Debug output
        for policy_name, policy_data in trajectory_per_task_dict.items():
            if policy_data.batch is not None:
                rollout_len = len(policy_data)
                rewards = policy_data.non_tensor_batch.get("reward", [])
                print(f"[TREE DESIGN RETURN] env_idx={env_idx} returning {rollout_len} samples for {policy_name}, "
                      f"rewards={rewards[:5] if len(rewards) > 0 else []}")

        # Cleanup intermediate structures to free memory
        del designs, exec_results, exec_tasks, exec_task_meta, design_exec_results
        import gc
        gc.collect()

        return trajectory_per_task_dict

    async def generate_multiple_rollouts_concurrent(self, env_idx_list, rollout_mode="tree"):
        rollout_indices = []
        for env_idx in env_idx_list:
            rollout_indices.extend(self.env_rollout_mapping[env_idx])
        
        concurrent_timer = create_timer("ConcurrentRollouts")
        concurrent_timer.start(f"Starting concurrent rollouts for {len(rollout_indices)} rollouts")
        
        concurrent_timer.checkpoint("Creating async tasks")

        # Use tree design sampling when execute_sample_num > 1 and not in validate mode
        use_tree_design = (self.execute_sample_num > 1 and self.mode != "validate")

        if use_tree_design:
            print(f"[TREE DESIGN] Using tree design sampling: {self.design_sample_num} designs × {self.execute_sample_num} executions")
            tasks = [
                asyncio.create_task(
                    self.generate_tree_rollout(env_idx=env_idx)
                )
                for env_idx in env_idx_list
            ]
        else:
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
                        if policy_data.batch is not None:
                            rollout_samples = len(policy_data)
                            rollout_rewards = policy_data.non_tensor_batch.get("reward", []) if hasattr(policy_data, 'non_tensor_batch') else []
                  

                            if aggregated_results[policy_name].batch is None:
                                aggregated_results[policy_name] = policy_data
                            else:
                                aggregated_results[policy_name], policy_data = _align_non_tensor_batch_keys(
                                    aggregated_results[policy_name], policy_data
                                )
                                aggregated_results[policy_name] = DataProto.concat([
                                    aggregated_results[policy_name],
                                    policy_data
                                ])
             
                   
                       
                    # Free the rollout result after aggregation
                    del rollout_result
                    completed_count += 1

                    # Periodic GC during aggregation to prevent memory buildup
                    if completed_count % 8 == 0:
                        import gc
                        gc.collect()

                    task_pbar.update(1)
                    task_pbar.set_description(f"Rollouts ({completed_count}/{len(tasks)})")
                except Exception as e:
                    failed_count += 1
                    print(f"[PRINT DEBUG] ========== ROLLOUT TASK FAILED ==========")
                    print(f"[PRINT DEBUG] Failed count: {failed_count}")
                    print(f"[PRINT DEBUG] Exception type: {type(e).__name__}")
                    print(f"[PRINT DEBUG] Exception message: {str(e)}")
                    import traceback
                    print(f"[PRINT DEBUG] Full traceback:")
                    print(traceback.format_exc())
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
            
        # Debug: check aggregated rewards
        for policy_name, policy_data in aggregated_results.items():
            if policy_data.batch is not None and "reward" in policy_data.non_tensor_batch:
                rewards = policy_data.non_tensor_batch["reward"]
           
        
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



