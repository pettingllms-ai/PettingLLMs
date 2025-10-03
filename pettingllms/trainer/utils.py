
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
import uuid
import hydra
from copy import deepcopy
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from dataclasses import dataclass
from openai.types.completion import Completion
from datetime import datetime, timedelta
import aiohttp

def _repeat_interleave(value: torch.Tensor | np.ndarray, repeats: int) -> torch.Tensor | np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)


async def poll_completions_openai(address: str, **completions_request) -> Completion:
    # Use aiohttp directly instead of AsyncOpenAI to avoid potential blocking
    # Handle address that might already contain protocol
    if address.startswith(('http://', 'https://')):
        base_url = f"{address}/v1/completions"
    else:
        base_url = f"http://{address}/v1/completions"
    # Debug: print resolved base URL
    try:
        print(f"[LLM][poll_completions_openai] address={address} -> base_url={base_url}")
    except Exception:
        pass
    headers = {
        "Content-Type": "application/json",
    }

    # Remove meta_info if present
    if "meta_info" in completions_request:
        completions_request.pop("meta_info")
    # Remove extra_headers from the payload
    if "extra_headers" in completions_request:
        completions_request.pop("extra_headers")

    max_retries = 3
    retry_delay = 1  # Initial delay in seconds

    for retry in range(max_retries):
        try:
            # Create a new session for each request to avoid blocking
            async with aiohttp.ClientSession() as session:
                async with session.post(base_url, json=completions_request, headers=headers, timeout=aiohttp.ClientTimeout(total=2700)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"API request failed with status {response.status}: {error_text}")
                    result = await response.json()
                    # Convert the raw JSON response to an OpenAI Completion object
                    return result
        except Exception as e:
            import traceback
            
            # Log the actual URL being attempted for debugging
            print(f"❌ Request failed to {base_url}: {str(e)}")
            traceback.print_exc()
            # If this is the last retry, raise the exception
            if retry == max_retries - 1:
                raise e
            # Exponential backoff
            print(f"🔄 Retrying in {retry_delay} seconds... (attempt {retry + 2}/{max_retries})")
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

    # This should never be reached due to the raise in the loop, but mypy requires it
    raise Exception("All retries failed")


async def submit_completions( address, model, prompt, **kwargs):
    # Potential blocking: network I/O can block
    return await poll_completions_openai(address=address, model=model, prompt=prompt, **kwargs)

async def postprocess_batch(batch: DataProto, response_ids: list[list[int]], n: int,pad_token_id,eos_token_id,max_response_length) -> DataProto:
    # NOTE: For Completion API, batch_completions is a list of lists of strings (not dictionaries)
    # prompts: left pad
    # responses: right pad
    # input_ids: prompt + response
    # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
    # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

    # prompts: [prompt] from input dataset
    idx = batch.batch["input_ids"]  # (bs, prompt_length)
    # left-padded attention_mask
    attention_mask = batch.batch["attention_mask"]
    position_ids = batch.batch["position_ids"]
    non_tensor_batch = deepcopy(batch.non_tensor_batch)

    # Flatten to list.
    # Flatten the list of lists of token IDs
    response = []
    for r_ids in response_ids:
        if r_ids is not None:  # Ensure we don't process None values
            for r in r_ids:
                response.append(r)
    assert len(response) == len(non_tensor_batch["formatted_prompts"]) * n
    response_tensor = pad_2d_list_to_length(response, pad_token_id, max_length=max_response_length).to(idx.device)

    if n > 1:
        idx = _repeat_interleave(idx, n)
        attention_mask = _repeat_interleave(attention_mask, n)
        position_ids = _repeat_interleave(position_ids, n)
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = _repeat_interleave(val, n)

    batch_size = len(idx)
    
    # Debug info before concatenation
    try:
        seq = torch.cat([idx, response_tensor], dim=-1)

        response_length = response_tensor.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response_tensor, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
    except Exception as e:
        print(f"[ERROR postprocess_batch] Tensor concatenation failed: {e}")
        print(f"[ERROR] Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

    output = TensorDict(
        {
            "prompts": idx,
            "responses": response_tensor,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        batch_size=batch_size,
    )
    return DataProto(batch=output, meta_info=batch.meta_info)

async def llm_async_generate(
    rollout_idx: int,
    turn_idx: int,
    agent_idx: int,
    prompt_dpr: DataProto,
    ppo_trainer_config: DictConfig,
    address: Optional[str] = None,
    model_name: Optional[str] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    enable_thinking: Optional[bool] = False,
    image_data: Optional[list[Any]] = None,
    application_id: Optional[str] = None,
    env_idx: Optional[int] = None,
    #rollout_idx: Optional[int] = None,
    policy_name: Optional[str] = None,
    timeout: Optional[float] = 60.0,    
    mode: Optional[str] = "train",
    override_temperature: Optional[float] = None
) -> DataProto:
    """Generate tokens from prompt ids or DataProto.

    Returns:
        DataProto: DataProto format output consistent with Router's generate_sequences.
    """

    if application_id is None:
        application_id = str(uuid.uuid4())
    else:
        application_id = str(application_id)

    unique_request_id = f"{application_id}_{uuid.uuid4().hex[:8]}"

    # Debug: print input address/model info
    try:
        print(f"[LLM][llm_async_generate] request_id={unique_request_id} address={address} model={model_name} mode={mode}")
    except Exception:
        pass

    if mode == "train":
        temperature = override_temperature if override_temperature is not None else ppo_trainer_config.actor_rollout_ref.rollout.temperature
        kwargs={
            "n":1,
            "temperature":temperature,
            "top_p":ppo_trainer_config.actor_rollout_ref.rollout.top_p,
            "max_tokens":ppo_trainer_config.data.max_response_length,
            "top_k":ppo_trainer_config.actor_rollout_ref.rollout.top_k,
            "logprobs":1,
        }
    if mode == "validate":
        if override_temperature is not None:
            temp = override_temperature
        elif enable_thinking:
            temp=0.6
        else:
            temp=0.0
        kwargs={
            "n":1,
            "temperature":temp,
            "top_p":1.0,
            "max_tokens":ppo_trainer_config.data.max_response_length,
            "top_k":-1,
            "logprobs":1,
        }
    batch_size = len(prompt_dpr.non_tensor_batch["formatted_prompts"])
    batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]

    tasks = []
    for batch_index, formatted_prompt in enumerate(prompt_dpr.non_tensor_batch["formatted_prompts"]):
        # For Completion API, we need to convert the conversation to a prompt string
        
        tasks.append(
            submit_completions(  # Changed from submit_chat_completions
                address=address,
                model=model_name,
                prompt=formatted_prompt,  # Changed from messages
                **kwargs,
            )
        )

    completions_list = await asyncio.gather(*tasks)
    for batch_index, completions in enumerate(completions_list):
        comps = []
        for choice in completions.get("choices", []):
            token_ids = choice.get("logprobs", {}).get("tokens", [])
            text = choice.get("text", "")
            token_ids = [int(t.split(":")[1]) for t in token_ids]
            comps.append(token_ids)
        batch_response_ids[batch_index] = comps
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    max_response_length=ppo_trainer_config.data.max_response_length
    output_dpr = await postprocess_batch(prompt_dpr, batch_response_ids, kwargs["n"], pad_token_id, eos_token_id,max_response_length)
    output_dpr.non_tensor_batch["rollout_idx"] = np.array([rollout_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["turn_idx"] = np.array([turn_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["agent_idx"] = np.array([agent_idx] * output_dpr.batch.shape[0], dtype=object)
    return output_dpr, text
    
    

@dataclass
class RequestState:
    """Track the state of a single request"""
    request_id: str
    server: ray.actor.ActorHandle
    created_time: datetime
    response_received_time: Optional[datetime] = None
    timeout_seconds: float = 10.0  
    is_timeout_cleanup_scheduled: bool = False
    is_cleaned_up: bool = False

    

def convert_prompt_to_format(tokenizer, enable_thinking, prompts,**kwargs):
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

    if not isinstance(prompts, dict) or "text" not in prompts:
        raise ValueError("prompts must be a dictionary containing 'text' key: {'text': str, 'image': Optional[path]} ")

    text = prompts.get("text", "") or ""
    image_path = prompts.get("image", None)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
  
    chat = np.array([
        {"content": text, "role": "user"}
    ])

    prompt_with_chat_template = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True
    )

    return prompt_with_chat_template
    

    

def convert_prompt_to_dpr(tokenizer, processor, prompts, max_prompt_length, multi_modal=False, enable_thinking=False, **kwargs):
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
            enable_thinking=enable_thinking
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
        if len(input_ids) >= max_prompt_length:
            return None

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

from math import prod
from typing import List, Dict, Iterator, Tuple, Optional

def compute_strides(sizes: List[int]) -> List[int]:
    """Row-major（最后一维变化最快）的stride。"""
    n = len(sizes)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return strides

def tuple_to_index(t: List[int], sizes: List[int]) -> int:
    """(t0,...,tN-1) -> rollout 索引。"""
    strides = compute_strides(sizes)
    return sum(t[i] * strides[i] for i in range(len(sizes)))

def index_to_tuple(idx: int, sizes: List[int]) -> List[int]:
    """rollout 索引 -> (t0,...,tN-1)。"""
    t = []
    strides = compute_strides(sizes)
    for i, s in enumerate(strides):
        q, idx = divmod(idx, s)
        t.append(q)
    return t

def agent_sample_rollouts(
    agent_idx: int, sample_idx: int, sizes: List[int]
) :
    strides = compute_strides(sizes)
    stride = strides[agent_idx]         # block_len
    period = sizes[agent_idx] * stride
    total = prod(sizes)
    result=[]

    for base in range(0, total, period):
        start = base + sample_idx * stride
       
        for k in range(stride):
            result.append(start + k)
    return result


def build_reverse_mapping(
    agent_names: List[str],
    sizes: List[int],
    batch_size: int,
) -> Dict[str, List[Dict[str, object]]]:
    total = prod(sizes)
    strides = compute_strides(sizes)
    out: Dict[str, List[Dict[str, object]]] = {}
    total=1
    for _ in sizes:
        total*=_

    for batch_idx in range(batch_size):
        out[batch_idx] = {}
        for i, name in enumerate(agent_names):
            stride = strides[i]
            period = sizes[i] * stride
            items = []
            for s in range(sizes[i]):
                starts = list(range(s * stride, total, period))
                
                rollouts = []
                for st in starts:
                    rollouts.extend(range(st+batch_idx*total, st+batch_idx*total + stride))
                entry= rollouts
                items.append(entry)
            out[batch_idx][name] = items
    return out
