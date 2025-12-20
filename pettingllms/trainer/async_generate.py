
# limitations under the License.
import asyncio
import heapq
import logging
import os
import random
from readline import add_history
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

_shared_session = None
_session_lock = None
_current_loop_id = None


_llm_request_semaphore = None
_semaphore_lock = None

_DEBUG_API_CALLS = False


def set_debug_api_calls(enabled: bool):
    """Enable or disable debug output for API calls"""
    global _DEBUG_API_CALLS
    _DEBUG_API_CALLS = enabled


def reset_event_loop_resources():
    """Reset all event loop bound resources when a new event loop is created."""
    global _shared_session, _session_lock, _llm_request_semaphore, _semaphore_lock, _current_loop_id
    
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = None
    
    # Check if we're in a new event loop
    if loop_id != _current_loop_id:
        # Reset all global resources without trying to close the old session
        # The old session will be garbage collected when the old loop is closed
        _shared_session = None
        _session_lock = None
        _llm_request_semaphore = None
        _semaphore_lock = None
        _current_loop_id = loop_id


async def get_shared_session() -> aiohttp.ClientSession:

    global _shared_session, _session_lock
    
    # Ensure lock is created in current event loop
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    
    async with _session_lock:
        if _shared_session is None or _shared_session.closed:
            connector = aiohttp.TCPConnector(
                limit=200,  
                limit_per_host=100,  
                ttl_dns_cache=300,  
                force_close=False,  
                enable_cleanup_closed=True  
            )
            
            timeout = aiohttp.ClientTimeout(
                total=300,  
                connect=30,  
                sock_read=300  
            )
            
            _shared_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            if _DEBUG_API_CALLS:
                print(f"[Session] Created shared aiohttp session with connection pool (limit=200, per_host=100)")
    
    return _shared_session


async def get_llm_semaphore(max_concurrent: int = 50) -> asyncio.Semaphore:

    global _llm_request_semaphore, _semaphore_lock
    
    # Ensure lock is created in current event loop
    if _semaphore_lock is None:
        _semaphore_lock = asyncio.Lock()
    
    async with _semaphore_lock:
        if _llm_request_semaphore is None:
            _llm_request_semaphore = asyncio.Semaphore(max_concurrent)
            if _DEBUG_API_CALLS:
                print(f"[Semaphore] Created global LLM request semaphore with max_concurrent={max_concurrent}")
    
    return _llm_request_semaphore


async def cleanup_shared_session():
    
    global _shared_session, _session_lock
    
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    
    async with _session_lock:
        if _shared_session is not None and not _shared_session.closed:
            await _shared_session.close()
            _shared_session = None
            if _DEBUG_API_CALLS:
                print("[Session] Closed shared aiohttp session")



async def poll_completions_openai(address: str, timeout: Optional[float] = None, **completions_request) -> Completion:
   
    session = await get_shared_session()
    semaphore = await get_llm_semaphore(max_concurrent=50)
    
    if address.startswith(('http://', 'https://')):
        base_url = f"{address}/v1/completions"
    else:
        base_url = f"http://{address}/v1/completions"
    
    headers = {
        "Content-Type": "application/json",
    }

    # Set timeout for the API call
    api_timeout = aiohttp.ClientTimeout(total=timeout) if timeout else aiohttp.ClientTimeout(total=300)  # Default 5 minutes

    completions_request.pop("meta_info", None)
    completions_request.pop("extra_headers", None)
    
    if _DEBUG_API_CALLS:
        print(f"[API][REQUEST_START] Sending request to {address} at {time.time()} with timeout={api_timeout.total}s")
    async with semaphore:
        try:
            async with session.post(
                base_url, 
                json=completions_request, 
                headers=headers,
                timeout=api_timeout
            ) as response:
                if _DEBUG_API_CALLS:
                    print(f"[API][RESPONSE_RECEIVED] Got response from {address} with status {response.status} at {time.time()}")
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                result = await response.json()
                if _DEBUG_API_CALLS:
                    print(f"[API][JSON_PARSED] JSON parsed successfully from {address} at {time.time()}")
                return result
                
        except asyncio.TimeoutError as e:
            error_msg = f"Request timeout to {address}"
            if _DEBUG_API_CALLS:
                print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
            
        except aiohttp.ClientError as e:
            error_msg = f"Client error when requesting {address}: {e}"
            if _DEBUG_API_CALLS:
                print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error when requesting {address}: {e}"
            if _DEBUG_API_CALLS:
                print(f"[ERROR] {error_msg}")
            raise


async def submit_completions(
    address: str, 
    model: str, 
    prompt: str, 
    max_retries: int = 3,
    initial_retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    **kwargs
):
    last_exception = None
    
    if _DEBUG_API_CALLS:
        print(f"[Submit][START] Starting submit_completions to {address} at {time.time()} with timeout={timeout}s")
    for attempt in range(max_retries):
        try:
            result = await poll_completions_openai(
                address=address, 
                model=model, 
                prompt=prompt,
                timeout=timeout,
                **kwargs
            )
            if attempt > 0 and _DEBUG_API_CALLS:
                print(f"[Retry] Request succeeded on attempt {attempt + 1}/{max_retries}")
            return result
            
        except Exception as e:
            last_exception = e
            # Proactively reset loop-bound resources on transient loop/session errors
            transient_msg = str(e)
            should_reset = (
                isinstance(e, asyncio.TimeoutError)
                or (hasattr(e, '__class__') and e.__class__.__name__.startswith('Client'))
                or ('Event loop is closed' in transient_msg)
            )
            if should_reset:
                if _DEBUG_API_CALLS:
                    print("[Submit][RESET] Detected transient loop/session error; resetting event-loop resources")
                reset_event_loop_resources()
            
  
            if attempt == max_retries - 1:
                if _DEBUG_API_CALLS:
                    print(f"[ERROR] All {max_retries} retry attempts failed for {address}")
                    print(f"[ERROR] Final error: {e}")
                raise e
            

            retry_delay = initial_retry_delay * (2 ** attempt)
            if _DEBUG_API_CALLS:
                print(f"[Retry] Attempt {attempt + 1}/{max_retries} failed: {e}")
                print(f"[Retry] Retrying in {retry_delay:.1f}s... (address={address})")
            
            await asyncio.sleep(retry_delay)
    
    if last_exception:
        raise last_exception
    else:
        raise Exception(f"Request failed after {max_retries} attempts")

def postprocess_batch(batch: DataProto, response_ids: list[list[int]], n: int,pad_token_id,eos_token_id,max_response_length,max_prompt_length) -> DataProto:
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
    
    # Truncate prompts if they exceed max_prompt_length
    if idx.size(1) > max_prompt_length:
        if _DEBUG_API_CALLS:
            print(f"[WARNING] Truncating prompt from {idx.size(1)} to {max_prompt_length}")
        # For left-padded prompts, keep the rightmost tokens (most recent context)
        idx = idx[:, -max_prompt_length:]
        attention_mask = attention_mask[:, -max_prompt_length:]
        position_ids = position_ids[:, -max_prompt_length:]

    # Flatten to list.
    # Flatten the list of lists of token IDs
    response = []
    for r_ids in response_ids:
        if r_ids is not None and len(r_ids) > 0:  # Ensure we don't process None or empty values
            for r in r_ids:
                # Handle empty response
                if r is None or len(r) == 0:
                    if _DEBUG_API_CALLS:
                        print(f"[WARNING] Empty response detected, using EOS token as fallback")
                    response.append([eos_token_id])
                    continue
                    
                # Truncate each response if it exceeds max_response_length
                if len(r) > max_response_length:
                    if _DEBUG_API_CALLS:
                        print(f"[WARNING] Truncating response from {len(r)} to {max_response_length}")
                    r = r[:max_response_length]
                response.append(r)
        else:
            # Fallback for None or empty r_ids
            if _DEBUG_API_CALLS:
                print(f"[WARNING] None or empty r_ids detected, using EOS token as fallback")
            for _ in range(n):  # Add n empty responses for this batch
                response.append([eos_token_id])
    
    # Ensure we have the expected number of responses
    expected_count = len(non_tensor_batch["formatted_prompts"]) * n
    if len(response) != expected_count:
        if _DEBUG_API_CALLS:
            print(f"[WARNING] Response count mismatch: expected {expected_count}, got {len(response)}")
        # Pad with empty responses if needed
        while len(response) < expected_count:
            if _DEBUG_API_CALLS:
                print(f"[WARNING] Adding fallback empty response")
            response.append([eos_token_id])
        # Truncate if too many
        if len(response) > expected_count:
            if _DEBUG_API_CALLS:
                print(f"[WARNING] Too many responses, truncating to {expected_count}")
            response = response[:expected_count]
    
    response_tensor = pad_2d_list_to_length(response, pad_token_id, max_length=max_response_length).to(idx.device)

    batch_size = len(idx)
    
    # Debug info before concatenation
    try:
        # Ensure the concatenated sequence doesn't exceed max total length
        max_total_length = max_prompt_length + max_response_length
        current_prompt_length = idx.size(1)
        current_response_length = response_tensor.size(1)
        
        # If total would exceed limit, adjust response length
        if current_prompt_length + current_response_length > max_total_length:
            new_response_length = max_total_length - current_prompt_length
            if new_response_length > 0:
                if _DEBUG_API_CALLS:
                    print(f"[WARNING] Total sequence too long ({current_prompt_length + current_response_length}), truncating response to {new_response_length}")
                response_tensor = response_tensor[:, :new_response_length]
            else:
                if _DEBUG_API_CALLS:
                    print(f"[ERROR] Prompt is already too long ({current_prompt_length}), setting response to empty")
                response_tensor = torch.full((batch_size, 1), pad_token_id, dtype=response_tensor.dtype, device=response_tensor.device)
        
        seq = torch.cat([idx, response_tensor], dim=-1)

        response_length = response_tensor.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response_tensor, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
    except Exception as e:
        if _DEBUG_API_CALLS:
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
    application_id: Optional[str] = None,
    env_idx: Optional[int] = None,
    #rollout_idx: Optional[int] = None,
    policy_name: Optional[str] = None,
    timeout: Optional[float] = 60.0,    
    mode: Optional[str] = "train",
    #override_temperature: Optional[float] = None,
    lora_id: Optional[str] = None,
    agent_config: Optional[DictConfig] = None,
    sample_num: Optional[int] = 1,  # Number of responses to generate per prompt
) -> DataProto:
    """Generate tokens from prompt ids or DataProto.
    
    Args:
        sample_num: Number of different responses to generate for each prompt (default: 1).
                   When > 1, the API will generate multiple diverse responses.

    Returns:
        DataProto: DataProto format output consistent with Router's generate_sequences.
        List[str]: List of generated text responses (length = sample_num).
    """

    if application_id is None:
        application_id = str(uuid.uuid4())
    else:
        application_id = str(application_id)

    unique_request_id = f"{application_id}_{uuid.uuid4().hex[:8]}"

    # Debug: print input address/model info
    if _DEBUG_API_CALLS:
        print(f"[LLM][llm_async_generate] request_id={unique_request_id} address={address} model={model_name} mode={mode}")

    # Ensure loop-bound resources (aiohttp session, semaphore) are bound to the current event loop
    reset_event_loop_resources()


    # Default LLM config parameters
    default_llm_config = {
        'enable_thinking': enable_thinking,
        'temperature': 0.8 if mode == "train" else 0.6,
        'top_p': 0.9 if mode == "train" else 0.95,
        'top_k': 20,
        'min_p': 0.0,
        'stop': None,
        'presence_penalty': 0.0,
        'frequency_penalty': 0.0,
        'repetition_penalty': 1.0,
        'best_of': None,
        'ignore_eos': False,
        'skip_special_tokens': True,
        'spaces_between_special_tokens': True,
    }

    # Read from agent's train_llm_config or val_llm_config, fallback to default
    llm_config = None
    if agent_config is not None:
        llm_config = getattr(agent_config, 'train_llm_config' if mode == "train" else 'val_llm_config', None)

    # Merge llm_config with defaults
    config = {**default_llm_config, **llm_config} if llm_config else default_llm_config

    # Extract parameters with type conversion
    enable_thinking = config['enable_thinking']
    temp = float(config['temperature'])
    top_p = float(config['top_p'])
    top_k = int(config['top_k'])
    min_p = float(config['min_p'])
    stop = config['stop']
    presence_penalty = float(config['presence_penalty'])
    frequency_penalty = float(config['frequency_penalty'])
    repetition_penalty = float(config['repetition_penalty'])
    best_of = config['best_of']
    ignore_eos = config['ignore_eos']
    skip_special_tokens = config['skip_special_tokens']
    spaces_between_special_tokens = config['spaces_between_special_tokens']
    
    if _DEBUG_API_CALLS:
        print(f"[LLM][llm_async_generate] enable_thinking={enable_thinking} mode={mode} temperature={temp} top_p={top_p} top_k={top_k} min_p={min_p} sample_num={sample_num}")
        print(f"[LLM][llm_async_generate] stop={stop} presence_penalty={presence_penalty} frequency_penalty={frequency_penalty} repetition_penalty={repetition_penalty}")
        print(f"[LLM][llm_async_generate] best_of={best_of} ignore_eos={ignore_eos}")

    kwargs={
        "n": sample_num,
        "temperature":temp,
        "top_p":top_p,
        "max_tokens":ppo_trainer_config.data.max_response_length,
        "top_k":top_k,
        "min_p":min_p,
        "logprobs":1,
    }
    
    if stop is not None:
        kwargs["stop"] = stop
    if presence_penalty != 0.0:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty != 0.0:
        kwargs["frequency_penalty"] = frequency_penalty
    if repetition_penalty != 1.0:
        kwargs["repetition_penalty"] = repetition_penalty
    if best_of is not None:
        kwargs["best_of"] = best_of
    if ignore_eos:
        kwargs["ignore_eos"] = ignore_eos
    if not skip_special_tokens:
        kwargs["skip_special_tokens"] = skip_special_tokens
    if not spaces_between_special_tokens:
        kwargs["spaces_between_special_tokens"] = spaces_between_special_tokens
    batch_size = len(prompt_dpr.non_tensor_batch["formatted_prompts"])
    batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]
    text_list = []  # Initialize text list for multiple samples

    # vLLM uses the 'model' parameter to specify which LoRA adapter to use
    # If lora_id is provided, use it as the model name (e.g., "lora_0", "lora_1")
    # Otherwise, use the base model name
    if lora_id is not None:
        # Convert lora_id to LoRA adapter name format
        if isinstance(lora_id, int):
            actual_model = f"lora_{lora_id}"
        elif isinstance(lora_id, str):
            if 'lora_' in lora_id:
                # Extract ID from formats like "agent_xxx_lora_0" -> "lora_0"
                lora_int_id = int(lora_id.split('lora_')[-1])
                actual_model = f"lora_{lora_int_id}"
            else:
                actual_model = lora_id
        else:
            actual_model = model_name
        if _DEBUG_API_CALLS:
            print(f"[LLM][LoRA] Using LoRA adapter: {actual_model} (from lora_id={lora_id})")
    else:
        actual_model = model_name
        if _DEBUG_API_CALLS:
            print(f"[LLM][LoRA] Using base model: {actual_model}")
    
    tasks = []
    for batch_index, formatted_prompt in enumerate(prompt_dpr.non_tensor_batch["formatted_prompts"]):
        # For Completion API, we need to convert the conversation to a prompt string
        
        # Prepare request parameters
        request_kwargs = {
            "address": address,
            "model": actual_model,  # Use LoRA adapter name if provided
            "prompt": formatted_prompt,
            "max_retries": 3,
            "initial_retry_delay": 1.0,
            "timeout": timeout,
            **kwargs,
        }
        
        tasks.append(submit_completions(**request_kwargs))


    if _DEBUG_API_CALLS:
        print(f"[LLM][batch_request] Starting {len(tasks)} requests for rollout_idx={rollout_idx}, turn_idx={turn_idx}")
    
    # Use return_exceptions=True to handle failures gracefully
    start_time = time.time()
    completions_list = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed_time = time.time() - start_time
    if _DEBUG_API_CALLS:
        print(f"[LLM][AWAIT_COMPLETE] asyncio.gather() completed at {time.time()}")
    
    success_count = sum(1 for c in completions_list if not isinstance(c, Exception))
    error_count = len(completions_list) - success_count
    
    if _DEBUG_API_CALLS:
        print(f"[LLM][DEBUG] API calls completed in {elapsed_time:.2f}s: {success_count} success, {error_count} errors for rollout_idx={rollout_idx}")
    
    for batch_index, completions in enumerate(completions_list):
        comps = []
        batch_texts = []
        
        # Handle exceptions from API calls
        if isinstance(completions, Exception):
            if _DEBUG_API_CALLS:
                print(f"[ERROR] API call failed for batch {batch_index}: {completions}")
            # Return empty token list as fallback for each sample
            for _ in range(sample_num):
                comps.append([tokenizer.eos_token_id])
                batch_texts.append("")
            batch_response_ids[batch_index] = comps
            text_list.extend(batch_texts)
            continue
            
        # Handle None or invalid responses
        if completions is None:
            if _DEBUG_API_CALLS:
                print(f"[WARNING] Got None response for batch {batch_index}")
            for _ in range(sample_num):
                comps.append([tokenizer.eos_token_id])
                batch_texts.append("")
            batch_response_ids[batch_index] = comps
            text_list.extend(batch_texts)
            continue
            
        # Normal processing
        try:
            choices = completions.get("choices", [])
            if not choices:
                if _DEBUG_API_CALLS:
                    print(f"[WARNING] No choices in response for batch {batch_index}")
                for _ in range(sample_num):
                    comps.append([tokenizer.eos_token_id])
                    batch_texts.append("")
            else:
                for choice in choices:
                    token_ids = choice.get("logprobs", {}).get("tokens", [])
                    text = choice.get("text", "")
                    batch_texts.append(text)
                    token_ids = [int(t.split(":")[1]) for t in token_ids]
                    comps.append(token_ids)
                    
        except Exception as e:
            if _DEBUG_API_CALLS:
                print(f"[ERROR] Failed to process response for batch {batch_index}: {e}")
            for _ in range(sample_num):
                comps.append([tokenizer.eos_token_id])
                batch_texts.append("")
            
        batch_response_ids[batch_index] = comps
        text_list.extend(batch_texts)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    max_response_length=ppo_trainer_config.data.max_response_length
    max_prompt_length=ppo_trainer_config.data.max_prompt_length
    output_dpr = postprocess_batch(prompt_dpr, batch_response_ids, kwargs["n"], pad_token_id, eos_token_id,max_response_length,max_prompt_length)
    output_dpr.non_tensor_batch["rollout_idx"] = np.array([rollout_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["env_idx"] = np.array([env_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["turn_idx"] = np.array([turn_idx] * output_dpr.batch.shape[0], dtype=object)
    output_dpr.non_tensor_batch["agent_idx"] = np.array([agent_idx] * output_dpr.batch.shape[0], dtype=object)

    # Fallback: if text_list is empty but we have response token ids, decode to string
    if (not text_list or all(not t for t in text_list)) and tokenizer is not None:
        # Decode all samples
        if "responses" in output_dpr.batch.keys():
            text_list = []
            for i in range(min(sample_num, output_dpr.batch["responses"].shape[0])):
                resp_ids_tensor = output_dpr.batch["responses"][i]
                # filter padding and eos
                valid_ids = [int(t) for t in resp_ids_tensor.tolist() if int(t) not in (pad_token_id, eos_token_id)]
                decoded = tokenizer.decode(valid_ids, skip_special_tokens=True)
                text_list.append(decoded or "")

    # Return format depends on sample_num:
    # - sample_num=1: return single string (backward compatible)
    # - sample_num>1: return list of strings
    if sample_num == 1:
        response = text_list[0] if text_list else ""
        # Debug empty responses
        if not response and _DEBUG_API_CALLS:
            print(f"[WARNING] Empty response for rollout_idx={rollout_idx}, turn_idx={turn_idx}, agent_idx={agent_idx}")
    else:
        response = text_list
        # Debug empty responses
        if (not response or all(not r for r in response)) and _DEBUG_API_CALLS:
            print(f"[WARNING] All responses empty for rollout_idx={rollout_idx}, turn_idx={turn_idx}, agent_idx={agent_idx}")
    
    if _DEBUG_API_CALLS:
        print(f"[LLM][FUNCTION_COMPLETE] llm_async_generate returning for rollout_idx={rollout_idx}, turn_idx={turn_idx}, agent_idx={agent_idx}")
    return output_dpr, response
    
    

    

def convert_prompt_to_format(tokenizer, enable_thinking, prompts,**kwargs):
    """
    Convert prompt dict to veRL's DataProto.
    
    Args:
        tokenizer: HF tokenizer, must support apply_chat_template and __call__ tokenization
        enable_thinking: Whether to enable thinking mode for chat template
        prompts: dict, {"text": str, "image": None, image path, or PIL.Image}
        kwargs: Optional parameters, such as meta_info, etc.
    Returns:
        DataProto: Contains tensor and non-tensor information
    """

    if not isinstance(prompts, dict) or "text" not in prompts:
        raise ValueError("prompts must be a dictionary containing 'text' key: {'text': str, 'image': Optional[path_or_image]} ")

    text = prompts.get("text", "") or ""
    image_data = prompts.get("image", None)

    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
  
    chat = np.array([
        {"content": text, "role": "user"}
    ])

    prompt_with_chat_template = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking
    )

    return prompt_with_chat_template
    

    

def convert_prompt_to_dpr(tokenizer, processor, prompts, max_prompt_length, multi_modal=False, enable_thinking=False, **kwargs):
    """
    Convert prompt dict to veRL's DataProto.
    
    Args:
        tokenizer: HF tokenizer, must support apply_chat_template and __call__ tokenization
        chat_parser: Reserved (currently unused)
        prompts: dict, {"text": str, "image": None, image path, or PIL.Image}
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
        raise ValueError("prompts must be a dictionary containing 'text' key: {'text': str, 'image': Optional[path_or_image]} ")

    text = prompts.get("text", "") or ""
    image_data = prompts.get("image", None)

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
        
        # Truncate if prompt exceeds max_prompt_length
        if input_ids.size(1) > max_prompt_length:
            if _DEBUG_API_CALLS:
                print(f"[WARNING] convert_prompt_to_dpr: Truncating prompt from {input_ids.size(1)} to {max_prompt_length}")
            # Keep the rightmost tokens (most recent context) for better quality
            input_ids = input_ids[:, -max_prompt_length:]
            attention_mask = attention_mask[:, -max_prompt_length:]
            # Regenerate the prompt string from truncated tokens to ensure consistency
            prompt_with_chat_template = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            if _DEBUG_API_CALLS:
                print(f"[WARNING] Regenerated prompt after truncation, new length: {len(prompt_with_chat_template)} chars")

        # Multimodal (optional): depends on externally provided processor
        multi_modal_inputs = None
        if multi_modal and image_data is not None and processor is not None:
            processed_image = image_data
            if isinstance(image_data, (str, os.PathLike)):
                from PIL import Image
                with Image.open(image_data) as img:
                    processed_image = img.convert("RGB")
            image_inputs = processor.image_processor([processed_image], return_tensors="pt")
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
        if _DEBUG_API_CALLS:
            print(f"Error in convert_dpr_to_response: {e}")
        return ""
