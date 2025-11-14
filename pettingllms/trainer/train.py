# Copyright under Agentica Project.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
import os
import json
from omegaconf import OmegaConf, DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from pettingllms.trainer.multi_agents_ppo_trainer import MultiAgentsPPOTrainer
from pettingllms.utils.clean_up import install_cleanup_hooks, register_temp_dirs
install_cleanup_hooks()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    # Set default values for lora_rank and lora_alpha if not defined
    # This prevents InterpolationKeyError when these variables are referenced in config
    if 'lora_rank' not in config or config.lora_rank is None:
        OmegaConf.set_struct(config, False)
        config.lora_rank = 0
        OmegaConf.set_struct(config, True)
    
    if 'lora_alpha' not in config or config.lora_alpha is None:
        OmegaConf.set_struct(config, False)
        config.lora_alpha = 0
        OmegaConf.set_struct(config, True)
    
    OmegaConf.to_yaml(config)
    run_ppo(config)


def run_ppo(config):
    ray_initialized_here = False
    multiagent_training_engine = None

    try:
        if not ray.is_initialized():
            ray_initialized_here = True
            # Create experiment-specific temporary directories using process ID
            pid = os.getpid()
            ray_tmp_dir = f"/tmp/verl_ray_{pid}"
            ray_spill_dir = f"/tmp/verl_spill_{pid}"
            os.makedirs(ray_tmp_dir, exist_ok=True)
            os.makedirs(ray_spill_dir, exist_ok=True)

            # Register directories for cleanup
            register_temp_dirs(ray_tmp_dir, ray_spill_dir)

            spilling_conf = {"type": "filesystem", "params": {"directory_path": [ray_spill_dir]}}
            system_config = {"object_spilling_config": json.dumps(spilling_conf)}

            # Get GPU count from config
            n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1) if hasattr(config, 'resource') else 1

            # Validate GPU availability
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if cuda_visible_devices:
                available_gpu_count = len(cuda_visible_devices.split(','))
                n_gpus_per_node = min(n_gpus_per_node, available_gpu_count)

            print(f"Initializing Ray with {n_gpus_per_node} GPUs")
            ray.init(
                num_gpus=n_gpus_per_node,
                runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
                _temp_dir=ray_tmp_dir,
                _system_config=system_config
            )

        def make_trainer_remote():
            num_cpus = max(8, int(ray.cluster_resources()["CPU"] * 0.1))
            return ray.remote(num_cpus=num_cpus)(train_multi_agents)

        multiagent_training_engine = make_trainer_remote()
        ray.get(multiagent_training_engine.remote(config))

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Clean up the remote training engine
        if multiagent_training_engine is not None:
            try:
                print("Killing remote training engine...")
                ray.kill(multiagent_training_engine)
            except Exception as e:
                print(f"Warning: Failed to kill training engine: {e}")

        # Only trigger cleanup_ray if Ray was initialized in this function
        # Otherwise, let the cleanup hooks handle it at exit
        print("\nTraining completed or interrupted. Resources will be cleaned up at exit.")

def train_multi_agents(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer, hf_processor
    from pettingllms.verl.ray_trainer import ResourcePoolManager, Role

    # Set default values for lora_rank and lora_alpha if not defined
    # This prevents InterpolationKeyError when resolving the config
    if 'lora_rank' not in config or config.lora_rank is None:
        OmegaConf.set_struct(config, False)  # Allow adding new keys
        config.lora_rank = 0
        print("lora_rank not configured, setting default value to 0 (LoRA disabled)")
        OmegaConf.set_struct(config, True)  # Re-enable struct mode

    if 'lora_alpha' not in config or config.lora_alpha is None:
        OmegaConf.set_struct(config, False)  # Allow adding new keys
        config.lora_alpha = 0
        print("lora_alpha not configured, setting default value to 0 (LoRA disabled)")
        OmegaConf.set_struct(config, True)  # Re-enable struct mode

    n_gpus_per_node = getattr(config.resource, 'n_gpus_per_node', 1)
    nnodes = getattr(config.resource, 'nnodes', 1)

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    multi_modal = getattr(config, 'multi_modal', False)

    tokenizer_dict = {}
    processor_dict = {}
    ppo_trainer_config_dict = {}
    model_num = 0

    if hasattr(config, 'models') and config.models is not None:
        print("Multi-model training mode detected")

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

        print(f"Creating resource pool for {model_key}: {resource_pool_spec}")
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        resource_pool_manager.create_resource_pool()
        managers.append(resource_pool_manager)

    trainer = MultiAgentsPPOTrainer(
        config=config,
        tokenizer_dict=tokenizer_dict,
        processor_dict=processor_dict,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=managers,
        ray_worker_group_cls=RayWorkerGroup,
    )

    try:
        trainer.init_workers()
        trainer.init_multi_agent_sys_execution_engine()
        trainer.fit()
    finally:
        # Ensure cleanup happens even if training fails
        print("\nExecuting cleanup after training...")
        try:
            trainer.cleanup()
        except Exception as e:
            print(f"Error during trainer cleanup: {e}")

        # Additional resource pool cleanup
        for manager in managers:
            try:
                if hasattr(manager, 'cleanup'):
                    manager.cleanup()
            except Exception as e:
                print(f"Error during manager cleanup: {e}")


if __name__ == "__main__":
    main()