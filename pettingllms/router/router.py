import asyncio
import logging
from copy import deepcopy

import aiohttp
import numpy as np
import torch
from openai.types.completion import Completion
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

logger = logging.getLogger(__name__)


def _repeat_interleave(value: torch.Tensor | np.ndarray, repeats: int) -> torch.Tensor | np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)


async def poll_completions_openai(address: str, **completions_request) -> Completion:
    # Use aiohttp directly instead of AsyncOpenAI to avoid potential blocking
    base_url = f"http://{address}/v1/completions"
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

            traceback.print_exc()
            # If this is the last retry, raise the exception
            if retry == max_retries - 1:
                raise e
            # Exponential backoff
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

    # This should never be reached due to the raise in the loop, but mypy requires it
    raise Exception("All retries failed")


class Router:
    """
    Least-used routing by address. Each model/policy has its own Router instance.
    """

    def __init__(self, config, tokenizer, addresses: list[str]):
        # A set of "ip:port" addresses
        self.addresses = addresses or []
        self.tensor_parallel_size = config.actor_rollout_ref.rollout.get("tensor_model_parallel_size", 1)
        self._lock = asyncio.Lock()
        # Track usage count for each address
        self._usage: dict[str, int] = {addr: 0 for addr in self.addresses}
        # Pin application_id to an address to improve data locality
        self._application_id_to_address: dict[str, str] = {}

        self.counter = 0
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])

    async def get_address(self, application_id: str) -> str:
        """
        Select the least-used address and increment its usage count.
        """
        async with self._lock:
            if not self._usage:
                raise RuntimeError("Router has no available addresses")

            min_address, min_usage = min(self._usage.items(), key=lambda x: x[1])
            if application_id not in self._application_id_to_address:
                self._application_id_to_address[application_id] = min_address
                self._usage[min_address] = self._usage.get(min_address, 0) + 1
            else:
                # Try to keep data locality while roughly balancing load
                cur_address = self._application_id_to_address[application_id]
                cur_usage = self._usage.get(cur_address, 0)
                if (min_usage == 0 or cur_usage - min_usage >= 4) and cur_usage > 0:
                    self._application_id_to_address[application_id] = min_address
                    self._usage[min_address] = self._usage.get(min_address, 0) + 1
                else:
                    self._usage[cur_address] = self._usage.get(cur_address, 0) + 1
        return self._application_id_to_address[application_id]

    async def release_address(self, addr: str, application_id: str) -> None:
        """ Decrement the usage count for a server address when done."""
        async with self._lock:
            if addr in self._usage:
                self._usage[addr] = max(0, self._usage.get(addr, 0) - 1)

    async def generate_sequences(self, batch: DataProto, application_id: str, **sampling_params):
        kwargs = dict(
            n=self.config.actor_rollout_ref.rollout.n,
            max_tokens=self.config.actor_rollout_ref.rollout.response_length,  # Changed from max_completion_tokens
            temperature=self.config.actor_rollout_ref.rollout.temperature,
            top_p=self.config.actor_rollout_ref.rollout.top_p,
            logprobs=1,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample or is_validate:
            kwargs["n"] = 1
            kwargs["temperature"] = 0

        if is_validate:
            kwargs.update(
                {
                    #'top_k': self.config.val_kwargs.top_k,
                    "top_p": self.config.actor_rollout_ref.rollout.val_kwargs.top_p,
                    "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        if batch.meta_info.get("max_tokens", None) is not None:
            kwargs["max_tokens"] = batch.meta_info["max_tokens"]

        if batch.meta_info.get("agent_rollout", False):
            kwargs["n"] = 1

        kwargs.update(sampling_params)

        address = await self.get_address(application_id)

        tasks = []
        # Bug: len(batch) is used later but batch might not have a __len__ method
        batch_size = len(batch.non_tensor_batch["formatted_prompts"])
        batch_response_ids: list[list[int]] = [[] for _ in range(batch_size)]

        for batch_index, formatted_prompt in enumerate(batch.non_tensor_batch["formatted_prompts"]):
            # For Completion API, we need to convert the conversation to a prompt string
            self.counter += 1
            tasks.append(
                self.submit_completions(  # Changed from submit_chat_completions
                    address=address,
                    model=self.model_name,
                    prompt=formatted_prompt,  # Changed from messages
                    **kwargs,
                )
            )

        # Potential blocking: asyncio.gather can block if any task takes too long
        logger.debug("Sending total requests: %s", self.counter)
        completions_list = await asyncio.gather(*tasks)
        await self.release_address(address, application_id)  # Release the address when done

        for batch_index, completions in enumerate(completions_list):
            comps = []
            for choice in completions.get("choices", []):
                token_ids = choice.get("logprobs", {}).get("tokens", [])
                token_ids = [int(t.split(":")[1]) for t in token_ids]
                comps.append(token_ids)
            batch_response_ids[batch_index] = comps

        return await self.postprocess_batch(batch, batch_response_ids, kwargs["n"])

    async def submit_completions(self, address, model, prompt, **kwargs):
        # Potential blocking: network I/O can block
        return await poll_completions_openai(address=address, model=model, prompt=prompt, **kwargs)

    async def postprocess_batch(self, batch: DataProto, response_ids: list[list[int]], n: int) -> DataProto:
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
        response_tensor = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.actor_rollout_ref.rollout.response_length).to(idx.device)

        if n > 1:
            idx = _repeat_interleave(idx, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            for key, val in non_tensor_batch.items():
                non_tensor_batch[key] = _repeat_interleave(val, n)

        batch_size = len(idx)
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
        response_attention_mask = get_response_mask(response_id=response_tensor, eos_token=self.eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

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


def test_router():

    router = Router(config=None, tokenizer=None, addresses=["127.0.0.1:8000"])
    batch = DataProto(batch=None, meta_info=None)
    response_ids = [[1, 2, 3], [4, 5, 6]]
    n = 1
    output = router.postprocess_batch(batch, response_ids, n)
    print(output)


if __name__ == "__main__":
    test_router()