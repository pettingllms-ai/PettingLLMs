# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import sys
import os
import logging

# Configure unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None

import hydra
import ray
from omegaconf import OmegaConf, DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
from pettingllms.utils.clean_up import cleanup_ray_runtime, install_cleanup_hooks
from pettingllms.utils.ray_utils import init_ray_with_temp_dirs

install_cleanup_hooks()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):   
    OmegaConf.to_yaml(config)
    run_ppo(config)


def run_ppo(config):
    try:
        # Initialize Ray with temporary directories
        init_ray_with_temp_dirs(config)
        
        # Create and execute remote trainer
        def make_trainer_remote():
            num_cpus = max(8, int(ray.cluster_resources()["CPU"] * 0.1)) 
            return ray.remote(num_cpus=num_cpus)(train_multi_agents)

        multiagent_training_engine = make_trainer_remote()
        ray.get(multiagent_training_engine.remote(config))
    finally:
        cleanup_ray_runtime()

def train_multi_agents(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer, hf_processor
    from pettingllms.verl.ray_trainer import ResourcePoolManager, Role
    from copy import deepcopy
    # Build agent_policy_mapping for validation
    agent_policy_mapping = {}
    for agent_key, agent_config in config.agent_policy_configs.agent_configs.items():
        agent_policy_mapping[agent_config.name] = agent_config.policy_name
        print(f"Agent mapping: {agent_config.name} -> {agent_config.policy_name}")
    num_base_models = len(config.base_models) if hasattr(config, 'base_models') else 0
    num_models = len(config.models) if hasattr(config, 'models') else 0
    num_agents = len(agent_policy_mapping)
    specialization = config.specialization
    # Validation 1: Check base_models and models count match (except for special case)
    if num_base_models != num_models:
        error_msg = (
            f"Configuration error: Number of base_models ({num_base_models}) does not match "
        )
        print("="*80)
        print(f"ERROR: {error_msg}")
        print("="*80)
        raise ValueError(error_msg)
    
    # Validation 2: Check specialization mode requirements
    if specialization == "prompt" or specialization == "lora":
        if num_models != 1:
            raise ValueError(
                f"For specialization={specialization}', expected exactly 1 model, but got {num_models}"
            )
    if specialization == "lora":
        # Validate LoRA configuration
        if config.lora_rank <= 0:
            raise ValueError(
                f"For specialization='lora', lora_rank must be > 0, but got {config.lora_rank}"
            )
    
    # Handle 'full' specialization with single base_model - replicate configs
    if specialization == "full" and num_base_models == 1:
        print("="*80)
        print("SPECIAL MODE: specialization='full' with single base_model detected")
        print(f"Replicating configurations to match {num_agents} agents...")
        
        base_model_config = config.base_models[list(config.base_models.keys())[0]]
        original_model_config = config.models[list(config.models.keys())[0]]
        
        # Replicate base_models
        new_base_models_dict = {
            f"base_model_{i}": deepcopy(base_model_config) 
            for i in range(num_agents)
        }
        
        # Replicate models
        new_models_dict = {
            f"model_{i}": deepcopy(original_model_config) 
            for i in range(num_agents)
        }
        
        OmegaConf.set_struct(config, False)
        config.base_models = OmegaConf.create(new_base_models_dict)
        config.models = OmegaConf.create(new_models_dict)
        OmegaConf.set_struct(config, True)


    
    n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1)
    nnodes = getattr(config.resource, 'nnodes', 1)
    OmegaConf.to_container(config, resolve=True)
    #pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    multi_modal = getattr(config, 'multi_modal', False)
    
    tokenizer_dict = {}
    tokenizer_path_dict = {}
    processor_dict = {}
    ppo_trainer_config_dict = {}
    model_num = 0



    for model_key, model_config in config.models.items():
        model_num += 1
        model_path = model_config.path
        model_name = model_config.name

        print(f"Processing model: {model_name} at path: {model_path}")

        local_path = copy_local_path_from_hdfs(model_path)

        trust_remote_code = getattr(model_config, 'trust_remote_code', False)
        if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
            trust_remote_code = config.resource.trust_remote_code

        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
        tokenizer_dict[model_name] = tokenizer
        tokenizer_path_dict[model_name] = local_path  # Store tokenizer path directly
        if multi_modal:
            processor_dict[model_name] = processor
        ppo_trainer_config_dict[model_name] = model_config.ppo_trainer_config

    n_gpus_per_model = n_gpus_per_node // model_num
    print(f"n_gpus_per_model: {n_gpus_per_model}")
    
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
    }
    
    managers = []
    for model_key, model_config in config.models.items():
        global_pool_id = f"global_pool_{model_key}"
        resource_pool_spec = {global_pool_id: [n_gpus_per_model] * nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        
        #print(f"Creating resource pool for {model_key}: {resource_pool_spec}")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        resource_pool_manager.create_resource_pool()
        managers.append(resource_pool_manager)
    
    trainer = MultiAgentsPPOTrainer(
        config=config,
        tokenizer_dict=tokenizer_dict,
        tokenizer_path_dict=tokenizer_path_dict,
        processor_dict=processor_dict,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=managers,
        ray_worker_group_cls=RayWorkerGroup,
        agent_policy_mapping=agent_policy_mapping,
    )
    trainer.init_workers()
    
    trainer.init_multi_agent_sys_execution_engine()
    
    trainer.fit()


if __name__ == "__main__":
    main()