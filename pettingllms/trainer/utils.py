
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
import uuid
import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer
from pettingllms.misc import colorful_print

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.workers.rollout.async_server import async_server_class
from pettingllms.utils.logger_config import get_multi_logger


def initialize_llm_servers(worker_group,server_class,server_config):

    if worker_group is None:
        world_size=1
        name_prefix="actor_rollout"
    else:
        world_size=worker_group.world_size
        name_prefix=worker_group.name_prefix
    rollout_tp_size = server_config.actor_rollout_ref.rollout.tensor_model_parallel_size
    rollout_dp_size = world_size // rollout_tp_size

   

    async_llm_servers = [None] * rollout_dp_size
    server_addresses = [None] * rollout_dp_size

    # Start all server instances, restart if address already in use.
    unready_dp_ranks = set(range(rollout_dp_size))
    while len(unready_dp_ranks) > 0:
        if worker_group is None:
            options_kwargs = dict(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=__import__("ray._raylet")._raylet.NodeID.from_hex(ray.nodes()[0]["NodeID"]),
                        soft=False,
                    ),
            name=f"async_llm_server",
        )
            
        # Allocate a GPU to the server actor if available, otherwise vLLM will see CUDA but cannot set device
            if torch.cuda.is_available():
                options_kwargs["num_gpus"] = 1
            server = server_class.options(**options_kwargs).remote(server_config, 1, 0, "actor_rollout")
            servers={0:server}
        else:
            register_center = ray.get_actor(f"{name_prefix}_register_center")
            workers_info = ray.get(register_center.get_worker_info.remote())
            assert len(workers_info) == world_size
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * rollout_tp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(server_config, rollout_dp_size, rollout_dp_rank, name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

        
    

        for rollout_dp_rank, server in servers.items():
    
            address = ray.get(server.get_server_address.remote())
            server_addresses[rollout_dp_rank] = address
            async_llm_servers[rollout_dp_rank] = server
            unready_dp_ranks.remove(rollout_dp_rank)
        
        ray.get([server.init_engine.remote() for server in async_llm_servers])

          
    
    return async_llm_servers, server_addresses

        # All server instances are ready, init AsyncLLM engine.
        

    

class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
        
        # 初始化多日志系统
        self.multi_logger = get_multi_logger()

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        server = self.weighted_serveres[0][1][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @rollout_trace_op
    async def generate(
        self,
        dpr_prompt:DataProto,
        sampling_params: Optional[dict[str, Any]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        image_data: Optional[list[Any]] = None,
        application_id: Optional[str] = None,
        rollout_idx: Optional[int] = None,
        policy_name: Optional[str] = None,
    ) -> DataProto:
        """Generate tokens from prompt ids or DataProto.

        Args:
            dpr_prompt: llm_inputs.batch = TensorDict({
            "input_ids":
            "attention_mask":   
            "position_ids": 
            "responses":
            prompt_ids (List[int], optional): List of prompt token ids (for legacy usage).
            sampling_params (Dict[str, Any], optional): Sampling parameters (for legacy usage).
            application_id (str, optional): Application ID for new usage.

        Returns:
            DataProto: DataProto format output consistent with Router's generate_sequences.
        """
        # 生成唯一的application_id用于追踪
        if application_id is None:
            application_id = str(uuid.uuid4())
        else:
            application_id = str(application_id)
        
        # 记录模型生成开始
        self.multi_logger.log_model_interaction(
            rollout_idx=rollout_idx if rollout_idx is not None else -1,
            policy_name=policy_name if policy_name is not None else "unknown",
            prompt="",  # 这里先留空，后面会记录详细的prompt
            response="",  # 这里先留空，后面会记录response
            extra_data={
                "event": "generate_start",
                "dpr_prompt_shape": str(dpr_prompt.batch['input_ids'].shape) if hasattr(dpr_prompt, 'batch') else None,
                "sampling_params": sampling_params,
                "application_id": application_id
            }
        )
        
        server = self._choose_server(application_id)
        
        # Ensure sampling_params is a dictionary (vLLM requires mapping, not None)
        if sampling_params is None:
            sampling_params = {}
        
        # Extract prompt_ids from DataProto and convert to list
        prompt_ids = dpr_prompt.batch['input_ids'][0].tolist() 
        
        # Remove padding tokens
        original_length = len(prompt_ids)
        while prompt_ids and prompt_ids[0] == 151643:
            prompt_ids.pop(0)
            
        self.multi_logger.log_model_interaction(
            rollout_idx=rollout_idx if rollout_idx is not None else -1,
            policy_name=policy_name if policy_name is not None else "unknown",
            prompt="",
            response="",
            extra_data={
                "event": "prompt_preprocessing",
                "original_length": original_length,
                "final_length": len(prompt_ids),
                "first_10_tokens": prompt_ids[:10] if prompt_ids else [],
                "server_selected": str(server)
            }
        )
        
        # Ensure we have valid tokens
        if not prompt_ids:
            error_msg = "No valid tokens found after removing padding"
            self.multi_logger.log_model_interaction(
                rollout_idx=rollout_idx if rollout_idx is not None else -1,
                policy_name=policy_name if policy_name is not None else "unknown",
                prompt="",
                response="",
                extra_data={
                    "event": "generate_error",
                    "error": error_msg
                }
            )
            raise ValueError(error_msg) 
        
        # Use direct await on Ray remote call - this is the correct async pattern!
        import asyncio
        
        try:
            # 记录开始生成
            self.multi_logger.log_model_interaction(
                rollout_idx=rollout_idx if rollout_idx is not None else -1,
                policy_name=policy_name if policy_name is not None else "unknown",
                prompt=str(prompt_ids[:20]) + "..." if len(prompt_ids) > 20 else str(prompt_ids),
                response="",
                extra_data={
                    "event": "server_generate_start",
                    "prompt_ids_length": len(prompt_ids),
                    "timeout": 20.0
                }
            )
            
            # Directly await the Ray remote call with timeout
            output = await asyncio.wait_for(
                server.generate.remote(
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    request_id=str(application_id),  # Convert UUID to string
                ),
                timeout=20.0
            )
        except asyncio.TimeoutError:
            error_msg = f"Generate request timed out after 20 seconds for request {application_id}"
            self.multi_logger.log_model_interaction(
                rollout_idx=rollout_idx if rollout_idx is not None else -1,
                policy_name=policy_name if policy_name is not None else "unknown",
                prompt="",
                response="",
                extra_data={
                    "event": "generate_timeout",
                    "timeout": 20.0,
                    "application_id": application_id
                }
            )
            raise TimeoutError(error_msg)
        except Exception as e:
            self.multi_logger.log_model_interaction(
                rollout_idx=rollout_idx if rollout_idx is not None else -1,
                policy_name=policy_name if policy_name is not None else "unknown",
                prompt="",
                response="",
                extra_data={
                    "event": "generate_error",
                    "error": str(e),
                    "application_id": application_id
                }
            )
            raise
        
        # 开始解码输出
        response_str = tokenizer.decode(output, skip_special_tokens=True)
        
        # 记录生成完成和响应
        self.multi_logger.log_model_interaction(
            rollout_idx=rollout_idx if rollout_idx is not None else -1,
            policy_name=policy_name if policy_name is not None else "unknown",
            prompt=str(prompt_ids[:20]) + "..." if len(prompt_ids) > 20 else str(prompt_ids),
            response=response_str,
            extra_data={
                "event": "generate_complete",
                "output_length": len(output) if output else 0,
                "response_length": len(response_str)
            }
        )

        # Transform vLLM output to DataProto
        # Response ids from vLLM (output is list[int])
        if not isinstance(output, list):
            raise TypeError(
                f"Unexpected output type from server.generate: {type(output)}; expected list[int]"
            )
        response_ids_generated = output

        # Read lengths from config with sensible fallbacks
        rollout_cfg = getattr(self.config, "actor_rollout_ref", None)
        rollout_cfg = getattr(rollout_cfg, "rollout", None)
        prompt_max_len = int(getattr(rollout_cfg, "prompt_length", len(prompt_ids)))
        response_max_len = int(getattr(rollout_cfg, "response_length", len(response_ids_generated)))

        # Truncate to fit
        prompt_ids_tail = prompt_ids[-prompt_max_len:]
        response_ids_tail = response_ids_generated[:response_max_len]

        # Build tensors: prompts left-pad, responses right-pad
        device = torch.device("cpu")
        batch_size = 1
        pad_token_id = 0

        # prompts
        prompts_tensor = torch.full((batch_size, prompt_max_len), pad_token_id, dtype=torch.long, device=device)
        if len(prompt_ids_tail) > 0:
            prompts_tensor[0, -len(prompt_ids_tail) :] = torch.tensor(
                prompt_ids_tail, dtype=torch.long, device=device
            )
        prompt_attention_mask = torch.zeros((batch_size, prompt_max_len), dtype=torch.long, device=device)
        if len(prompt_ids_tail) > 0:
            prompt_attention_mask[0, -len(prompt_ids_tail) :] = 1

        # responses
        responses_tensor = torch.full((batch_size, response_max_len), pad_token_id, dtype=torch.long, device=device)
        if len(response_ids_tail) > 0:
            responses_tensor[0, : len(response_ids_tail)] = torch.tensor(
                response_ids_tail, dtype=torch.long, device=device
            )
        response_attention_mask = torch.zeros((batch_size, response_max_len), dtype=torch.long, device=device)
        if len(response_ids_tail) > 0:
            response_attention_mask[0, : len(response_ids_tail)] = 1

        # merge
        input_ids_tensor = torch.cat([prompts_tensor, responses_tensor], dim=1)
        attention_mask_tensor = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids_full = compute_position_id_with_mask(attention_mask_tensor)

        batch_dict = {
            "prompts": prompts_tensor,
            "responses": responses_tensor,
            "response_mask": response_attention_mask,
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "position_ids": position_ids_full,
        }
       
        # 记录最终的DataProto构建结果
        self.multi_logger.log_model_interaction(
            rollout_idx=rollout_idx if rollout_idx is not None else -1,
            policy_name=policy_name if policy_name is not None else "unknown",
            prompt="",
            response="",
            extra_data={
                "event": "dataproto_build_complete",
                "batch_dict_shapes": {
                    "prompts": str(batch_dict['prompts'].shape),
                    "responses": str(batch_dict['responses'].shape),
                    "response_mask": str(batch_dict['response_mask'].shape),
                    "input_ids": str(batch_dict['input_ids'].shape),
                    "attention_mask": str(batch_dict['attention_mask'].shape),
                    "position_ids": str(batch_dict['position_ids'].shape)
                },
                "output_dpr_type": str(type(output_dpr)) if 'output_dpr' in locals() else None
            }
        )
        
        output_dpr = DataProto.from_dict(batch_dict)

        return output_dpr,response_str
            

def convert_prompt_to_dpr(tokenizer, chat_parser, processor, prompts, max_prompt_length, multi_modal=False, **kwargs):
    """
    Convert prompt dict to veRL's DataProto.
    
    Args:
        tokenizer: HF tokenizer, must support apply_chat_template and __call__ tokenization
        chat_parser: Reserved (currently unused)
        prompts: dict, {"text": str, "image": None or image path}
        max_prompt_length: Maximum prompt length (left padding)
        multi_modal: Whether multimodal (if True, should also pass processor and other necessary information)
        kwargs: Optional parameters, such as processor, meta_info, etc.
    Returns:
        DataProto: Contains tensor and non-tensor information
    """
    from verl.protocol import DataProto, union_two_dict
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import pad_sequence_to_length
    import numpy as np
    import torch

    if not isinstance(prompts, dict) or "text" not in prompts:
        raise ValueError("prompts must be a dictionary containing 'text' key: {'text': str, 'image': Optional[path]} ")

    text = prompts.get("text", "") or ""
    image_path = prompts.get("image", None)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        chat = np.array([
            {"content": text, "role": "user"}
        ])
        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = tokenizer(
            prompt_with_chat_template,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

        # Multimodal (optional): depends on externally provided processor
        multi_modal_inputs = None
        if multi_modal and image_path is not None and "processor" in kwargs:
            
            image_inputs = processor.image_processor([image_path], return_tensors="pt")
            multi_modal_inputs = {k: v for k, v in image_inputs.items()}
           

        # Pad to a unified length
        input_ids = pad_sequence_to_length(
            input_ids,
            max_seq_len=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
        )
        attention_mask = pad_sequence_to_length(
            attention_mask,
            max_seq_len=max_prompt_length,
            pad_token_id=0,
            left_pad=True,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        data = DataProto.from_dict(batch_dict)
        data.non_tensor_batch["formatted_prompts"] = np.array([prompt_with_chat_template])
        if multi_modal_inputs is not None:
            data.non_tensor_batch["multi_modal_inputs"] = multi_modal_inputs

        # Merge meta_info if provided
        meta_info = kwargs.get("meta_info")
        if meta_info:
            data.meta_info = union_two_dict(data.meta_info, meta_info)

        return data
    finally:
        tokenizer.padding_side = old_padding_side


def convert_dpr_to_response(tokenizer, chat_parser, dpr, max_prompt_length, multi_modal=False, **kwargs):
    try:
        attn = dpr.batch["attention_mask"][0, max_prompt_length :]
        tokens = dpr.batch["responses"][0]

        # Find last index where attention == 1
        non_pad_indices = (attn == 1).nonzero(as_tuple=True)[0]
        if len(non_pad_indices) == 0:
            trimmed = tokens[:0]  # empty
        else:
            last_valid_idx = non_pad_indices[-1].item()
            trimmed = tokens[: last_valid_idx + 1]  # include the last valid token

        response = tokenizer.decode(trimmed, skip_special_tokens=False)

        pad_token = tokenizer.pad_token if tokenizer.pad_token else ""
        eos_token = tokenizer.eos_token if tokenizer.eos_token else ""
        response = response.replace(pad_token, "").replace(eos_token, "")
        
        # Ensure we always return a string
        return response if response is not None else ""
    except Exception as e:
        print(f"Error in convert_dpr_to_response: {e}")
        return ""

