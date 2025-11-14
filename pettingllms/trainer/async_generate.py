
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


def reset_event_loop_resources():
    """Reset all event loop bound resources when a new event loop is created."""
    global _shared_session, _session_lock, _llm_request_semaphore, _semaphore_lock, _current_loop_id
    
    try:
        loop = asyncio.get_event_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = None
    
    # Check if we're in a new event loop
    if loop_id != _current_loop_id:
        # Clean up old session if it exists
        if _shared_session is not None and not _shared_session.closed:
            try:
                # Try to close the session, but don't fail if the loop is closed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_shared_session.close())
                loop.close()
            except:
                pass
        
        # Reset all global resources
        _shared_session = None
        _session_lock = None
        _llm_request_semaphore = None
        _semaphore_lock = None
        _current_loop_id = loop_id


async def get_shared_session() -> aiohttp.ClientSession:
    """Get or create a shared aiohttp session (thread-safe, event loop safe)"""
    global _shared_session

    # Quick check without lock
    if _shared_session is not None and not _shared_session.closed:
        return _shared_session

    # Import threading for thread-safe initialization
    import threading
    if not hasattr(get_shared_session, '_init_lock'):
        get_shared_session._init_lock = threading.Lock()

    # Thread-safe session creation
    with get_shared_session._init_lock:
        # Double-check after acquiring lock
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
            print(f"[Session] Created shared aiohttp session (pid={os.getpid()}) with connection pool (limit=200, per_host=100)")

    return _shared_session


async def get_llm_semaphore(max_concurrent: int = 50) -> asyncio.Semaphore:
    """Get or create a semaphore (thread-safe, event loop safe)"""
    global _llm_request_semaphore

    # Quick check without lock
    if _llm_request_semaphore is not None:
        return _llm_request_semaphore

    # Import threading for thread-safe initialization
    import threading
    if not hasattr(get_llm_semaphore, '_init_lock'):
        get_llm_semaphore._init_lock = threading.Lock()

    # Thread-safe semaphore creation
    with get_llm_semaphore._init_lock:
        # Double-check after acquiring lock
        if _llm_request_semaphore is None:
            _llm_request_semaphore = asyncio.Semaphore(max_concurrent)
            print(f"[Semaphore] Created LLM request semaphore (pid={os.getpid()}) with max_concurrent={max_concurrent}")
    
    return _llm_request_semaphore


async def cleanup_shared_session():
    
    global _shared_session, _session_lock
    
    if _session_lock is None:
        _session_lock = asyncio.Lock()
    
    async with _session_lock:
        if _shared_session is not None and not _shared_session.closed:
            await _shared_session.close()
            _shared_session = None
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
    
    print(f"[API][REQUEST_START] Sending request to {address} at {time.time()} with timeout={api_timeout.total}s")
    async with semaphore:
        try:
            async with session.post(
                base_url, 
                json=completions_request, 
                headers=headers,
                timeout=api_timeout
            ) as response:
                print(f"[API][RESPONSE_RECEIVED] Got response from {address} with status {response.status} at {time.time()}")
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed with status {response.status}: {error_text}")
                result = await response.json()
                print(f"[API][JSON_PARSED] JSON parsed successfully from {address} at {time.time()}")
                return result
                
        except asyncio.TimeoutError as e:
            error_msg = f"Request timeout to {address}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
            
        except aiohttp.ClientError as e:
            error_msg = f"Client error when requesting {address}: {e}"
            print(f"[ERROR] {error_msg}")
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error when requesting {address}: {e}"
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
    
    print(f"[Submit][START] Starting submit_completions to {address} at {time.time()} with timeout={timeout}s")
    for attempt in range(max_retries):
        try:
            print(f"[Submit][ATTEMPT] Attempt {attempt + 1}/{max_retries} calling poll_completions_openai")
            result = await poll_completions_openai(
                address=address, 
                model=model, 
                prompt=prompt,
                timeout=timeout,
                **kwargs
            )
            if attempt > 0:
                print(f"[Retry] Request succeeded on attempt {attempt + 1}/{max_retries}")
            print(f"[Submit][SUCCESS] submit_completions succeeded at {time.time()}")
            return result
            
        except Exception as e:
            last_exception = e
            try:
                # Proactively reset loop-bound resources on transient loop/session errors
                transient_msg = str(e)
                should_reset = (
                    isinstance(e, asyncio.TimeoutError)
                    or (hasattr(e, '__class__') and e.__class__.__name__.startswith('Client'))
                    or ('Event loop is closed' in transient_msg)
                )
                if should_reset:
                    print("[Submit][RESET] Detected transient loop/session error; resetting event-loop resources")
                    reset_event_loop_resources()
            except Exception:
                pass
            
  
            if attempt == max_retries - 1:
                print(f"[ERROR] All {max_retries} retry attempts failed for {address}")
                print(f"[ERROR] Final error: {e}")
                raise e
            

            retry_delay = initial_retry_delay * (2 ** attempt)
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
                    print(f"[WARNING] Empty response detected, using EOS token as fallback")
                    response.append([eos_token_id])
                    continue
                    
                # Truncate each response if it exceeds max_response_length
                if len(r) > max_response_length:
                    print(f"[WARNING] Truncating response from {len(r)} to {max_response_length}")
                    r = r[:max_response_length]
                response.append(r)
        else:
            # Fallback for None or empty r_ids
            print(f"[WARNING] None or empty r_ids detected, using EOS token as fallback")
            for _ in range(n):  # Add n empty responses for this batch
                response.append([eos_token_id])
    
    # Ensure we have the expected number of responses
    expected_count = len(non_tensor_batch["formatted_prompts"]) * n
    if len(response) != expected_count:
        print(f"[WARNING] Response count mismatch: expected {expected_count}, got {len(response)}")
        # Pad with empty responses if needed
        while len(response) < expected_count:
            print(f"[WARNING] Adding fallback empty response")
            response.append([eos_token_id])
        # Truncate if too many
        if len(response) > expected_count:
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
                print(f"[WARNING] Total sequence too long ({current_prompt_length + current_response_length}), truncating response to {new_response_length}")
                response_tensor = response_tensor[:, :new_response_length]
            else:
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
    try:
        print(f"[LLM][llm_async_generate] request_id={unique_request_id} address={address} model={model_name} mode={mode}")
    except Exception:
        pass

    # Ensure loop-bound resources (aiohttp session, semaphore) are bound to the current event loop
    try:
        reset_event_loop_resources()
    except Exception:
        pass


    if mode == "train":
        default_temp = 0.8
        default_top_p = 0.9
        default_top_k = 20
        default_min_p = 0.0
    else:  # mode == "validate" or other
        if enable_thinking:
            default_temp = 0.6
            default_top_p = 0.95
            default_top_k = 20
            default_min_p = 0.0
        else:
            default_temp = 0.7
            default_top_p = 0.8
            default_top_k = 20
            default_min_p = 0.0
    
    if agent_config is not None:
        if mode == "train":
            temp = getattr(agent_config, 'train_temperature', default_temp)
            top_p = getattr(agent_config, 'train_top_p', default_top_p)
            top_k = getattr(agent_config, 'train_top_k', default_top_k)
            min_p = getattr(agent_config, 'train_min_p', default_min_p)
        else:  # mode == "validate" or other
            temp = getattr(agent_config, 'val_temperature', default_temp)
            top_p = getattr(agent_config, 'val_top_p', default_top_p)
            top_k = getattr(agent_config, 'val_top_k', default_top_k)
            min_p = getattr(agent_config, 'val_min_p', default_min_p)
    else:
        temp = default_temp
        top_p = default_top_p
        top_k = default_top_k
        min_p = default_min_p
    
    # Validate and fix empty or invalid values
    if temp is None or temp == '' or (isinstance(temp, str) and not temp.strip()):
        temp = default_temp
    if top_p is None or top_p == '' or (isinstance(top_p, str) and not top_p.strip()):
        top_p = default_top_p
    if top_k is None or top_k == '' or (isinstance(top_k, str) and not top_k.strip()):
        top_k = default_top_k
    if min_p is None or min_p == '' or (isinstance(min_p, str) and not min_p.strip()):
        min_p = default_min_p
    
    # Convert to proper types
    temp = float(temp)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    
    try:
        print(f"[LLM][llm_async_generate] enable_thinking={enable_thinking} mode={mode} temperature={temp} top_p={top_p} top_k={top_k} min_p={min_p} sample_num={sample_num}")
    except Exception:
        pass

    kwargs={
        "n": sample_num,  # Generate sample_num responses per prompt
        "temperature":temp,
        "top_p":top_p,
        "max_tokens":ppo_trainer_config.data.max_response_length,
        "top_k":top_k,
        "min_p":min_p,
        "logprobs":1,
    }
    batch_size = len(prompt_dpr.non_tensor_batch["formatted_prompts"])
    batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]
    text_list = []  # Initialize text list for multiple samples

    # Determine which model to use: LoRA adapter or base model
    # In vllm, LoRA is selected by specifying the lora_name as the model parameter
    if lora_id is not None:
        actual_model = lora_id  # Use LoRA adapter name as model
        try:
            print(f"[LLM][llm_async_generate] Using LoRA adapter: {lora_id}")
        except Exception:
            pass
    else:
        actual_model = model_name  # Use base model name
        try:
            print(f"[LLM][llm_async_generate] Using base model: {model_name}")
        except Exception:
            pass

    tasks = []
    for batch_index, formatted_prompt in enumerate(prompt_dpr.non_tensor_batch["formatted_prompts"]):
        # For Completion API, we need to convert the conversation to a prompt string
        
        tasks.append(
            submit_completions(
                address=address,
                model=actual_model,  # Pass LoRA name or base model name
                prompt=formatted_prompt,
                max_retries=3, 
                initial_retry_delay=1.0,
                timeout=timeout,  # Pass timeout to API call
                **kwargs,
            )
        )


    try:
        print(f"[LLM][batch_request] Starting {len(tasks)} requests for rollout_idx={rollout_idx}, turn_idx={turn_idx}")
    except Exception:
        pass
    
    # Use return_exceptions=True to handle failures gracefully
    start_time = time.time()
    print(f"[LLM][DEBUG] About to await {len(tasks)} API calls for rollout_idx={rollout_idx}, turn_idx={turn_idx}, agent_idx={agent_idx}")
    print(f"[LLM][AWAIT_START] Waiting for asyncio.gather() at {time.time()}")
    completions_list = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed_time = time.time() - start_time
    print(f"[LLM][AWAIT_COMPLETE] asyncio.gather() completed at {time.time()}")
    
    success_count = sum(1 for c in completions_list if not isinstance(c, Exception))
    error_count = len(completions_list) - success_count
    
    print(f"[LLM][DEBUG] API calls completed in {elapsed_time:.2f}s: {success_count} success, {error_count} errors for rollout_idx={rollout_idx}")
    
    for batch_index, completions in enumerate(completions_list):
        comps = []
        batch_texts = []
        
        # Handle exceptions from API calls
        if isinstance(completions, Exception):
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
                print(f"[WARNING] No choices in response for batch {batch_index}")
                for _ in range(sample_num):
                    comps.append([tokenizer.eos_token_id])
                    batch_texts.append("")
            else:
                for choice in choices:
                    token_ids = choice.get("logprobs", {}).get("tokens", [])
                    text = choice.get("text", "")
                    batch_texts.append(text)
                    if token_ids:
                        token_ids = [int(t.split(":")[1]) for t in token_ids]
                        comps.append(token_ids)
                    else:
                        # Fallback: if no token_ids, add EOS
                        print(f"[WARNING] No token_ids in choice for batch {batch_index}")
                        comps.append([tokenizer.eos_token_id])
        except Exception as e:
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
    try:
        if (not text_list or all(not t for t in text_list)) and tokenizer is not None:
            # Decode all samples
            if "responses" in output_dpr.batch:
                text_list = []
                for i in range(min(sample_num, output_dpr.batch["responses"].shape[0])):
                    resp_ids_tensor = output_dpr.batch["responses"][i]
                    # filter padding and eos
                    valid_ids = [int(t) for t in resp_ids_tensor.tolist() if int(t) not in (pad_token_id, eos_token_id)]
                    decoded = tokenizer.decode(valid_ids, skip_special_tokens=True)
                    text_list.append(decoded or "")
    except Exception:
        # Keep empty text list if decoding fails
        if not text_list:
            text_list = [""] * sample_num

    # Return format depends on sample_num:
    # - sample_num=1: return single string (backward compatible)
    # - sample_num>1: return list of strings
    if sample_num == 1:
        response = text_list[0] if text_list else ""
        # Debug empty responses
        if not response:
            print(f"[WARNING] Empty response for rollout_idx={rollout_idx}, turn_idx={turn_idx}, agent_idx={agent_idx}")
    else:
        response = text_list
        # Debug empty responses
        if not response or all(not r for r in response):
            print(f"[WARNING] All responses empty for rollout_idx={rollout_idx}, turn_idx={turn_idx}, agent_idx={agent_idx}")
    
    print(f"[LLM][FUNCTION_COMPLETE] llm_async_generate returning for rollout_idx={rollout_idx}, turn_idx={turn_idx}, agent_idx={agent_idx}")
    return output_dpr, response
    
    

    

def convert_prompt_to_format(tokenizer, enable_thinking, prompts,**kwargs):
    """
    Convert prompt dict to veRL's DataProto.
    
    Args:
        tokenizer: HF tokenizer, must support apply_chat_template and __call__ tokenization
        enable_thinking: Whether to enable thinking mode for chat template
        prompts: dict, {"text": str, "image": None or image path}
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
        enable_thinking=enable_thinking
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
        
        # Truncate if prompt exceeds max_prompt_length
        if input_ids.size(1) > max_prompt_length:
            print(f"[WARNING] convert_prompt_to_dpr: Truncating prompt from {input_ids.size(1)} to {max_prompt_length}")
            # Keep the rightmost tokens (most recent context) for better quality
            input_ids = input_ids[:, -max_prompt_length:]
            attention_mask = attention_mask[:, -max_prompt_length:]
            # Regenerate the prompt string from truncated tokens to ensure consistency
            prompt_with_chat_template = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(f"[WARNING] Regenerated prompt after truncation, new length: {len(prompt_with_chat_template)} chars")

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
