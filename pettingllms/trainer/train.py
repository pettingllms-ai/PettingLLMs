# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
from omegaconf import OmegaConf, DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
# Local application imports
from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    OmegaConf.to_yaml(config)
    #print(config.models.model_0.ppo_trainer_config.actor_rollout_ref.model.path)
    #print(config.models.model_1.ppo_trainer_config.actor_rollout_ref.model)
    #print(config.models.model_0.ppo_trainer_config)

    run_ppo(config)


def run_ppo(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(train_multi_agents.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def train_multi_agents(config, agent_args=None, env_args=None):
    # print initial config
    from pprint import pprint

    from omegaconf import OmegaConf

    from verl.utils.fs import copy_local_path_from_hdfs
   
    

    
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    # Safely read the multi_modal flag
    multi_modal = getattr(config, 'multi_modal', False)

    # Initialize tokenizer dictionary for multiple models
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer_dict = {}
    processor_dict = {}
    ppo_trainer_config_dict = {}
    
    # Check if we have models configuration for multi-model training
    if hasattr(config, 'models') and config.models is not None:
        print("Multi-model training mode detected")
        
        # Process each model in the models configuration
        for model_key, model_config in config.models.items():
            model_path = model_config.path
            model_name = model_config.name
            
            print(f"Processing model: {model_name} at path: {model_path}")
            
            # Download the model checkpoint from hdfs
            local_path = copy_local_path_from_hdfs(model_path)
            
            # Get trust_remote_code setting from model config or use default
            trust_remote_code = getattr(model_config, 'trust_remote_code', False)
            if hasattr(config, 'resource') and hasattr(config.resource, 'trust_remote_code'):
                trust_remote_code = config.resource.trust_remote_code
            
            # Create tokenizer for this model
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
            processor = hf_processor(local_path, trust_remote_code=trust_remote_code)
            tokenizer_dict[model_name] = tokenizer
            if multi_modal:
                processor_dict[model_name] = processor
            ppo_trainer_config = model_config.ppo_trainer_config
            ppo_trainer_config_dict[model_name] = ppo_trainer_config
        
            
            

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    ray_worker_group_cls = RayWorkerGroup

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(max_concurrency=2048)(AsyncActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    
    # Access resource configuration safely
    n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if hasattr(config, 'resource') else 1
    nnodes = getattr(config.resource, 'nnodes', 1) if hasattr(config, 'resource') else 1
    
    managers = []
    for id in range(2):
        global_pool_id = f"global_pool_{id}"
        resource_pool_spec = {
           global_pool_id: [n_gpus_per_node] * nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        managers.append(resource_pool_manager)


    trainer = MultiAgentsPPOTrainer(
        config=config,
        tokenizer_dict=tokenizer_dict,
        processor_dict=processor_dict,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=managers,
        ray_worker_group_cls=ray_worker_group_cls,
    )

    trainer.init_workers()
    trainer.init_multi_agent_sys_execution_engine()
    trainer.fit()


if __name__ == "__main__":
    main()
