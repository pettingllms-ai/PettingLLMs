"""Utility functions for autoevol module"""
import json
import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.model import compute_position_id_with_mask
from tensordict import TensorDict

logger = logging.getLogger(__name__)


def load_and_tokenize_jsonl(
    file_path: str,
    tokenizer,
    max_prompt_length: int,
    max_response_length: int
) -> List[Tuple[DataProto, str]]:
    """
    Load JSONL trajectory file and convert to DataProto format.

    Args:
        file_path: Path to the JSONL file (ShareGPT format)
        tokenizer: HuggingFace tokenizer
        max_prompt_length: Maximum prompt length
        max_response_length: Maximum response length

    Returns:
        List of (output_dpr, response_text) tuples matching llm_async_generate format
    """
    all_trajectories = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse the ShareGPT format JSON
                sharegpt_data = json.loads(line)

                # Check if token_ids are already included in the data
                has_token_ids = _check_has_token_ids(sharegpt_data)

                if has_token_ids:
                    logger.info("Found token_ids in trajectory data, using them directly")
                    # Use token_ids directly without re-tokenization
                    trajectories = _convert_sharegpt_with_token_ids_to_dataproto(
                        sharegpt_data,
                        tokenizer,
                        max_prompt_length,
                        max_response_length
                    )
                else:
                    logger.info("No token_ids found in trajectory data, falling back to tokenization")
                    # Tokenize this conversation
                    trajectories = _tokenize_sharegpt_to_dataproto(
                        sharegpt_data,
                        tokenizer,
                        max_prompt_length,
                        max_response_length
                    )

                all_trajectories.extend(trajectories)

    except Exception as e:
        logger.error(f"Error loading and processing trajectory file {file_path}: {e}")
        return []

    return all_trajectories


def _check_has_token_ids(sharegpt_data: dict) -> bool:
    """Check if sharegpt_data contains token_ids for responses"""
    conversations = sharegpt_data.get("conversations", [])
    for msg in conversations:
        if msg.get("from") == "gpt" and "token_ids" in msg:
            return True
    return False


def _convert_sharegpt_with_token_ids_to_dataproto(
    sharegpt_data: dict,
    tokenizer,
    max_prompt_length: int,
    max_response_length: int
) -> List[Tuple[DataProto, str]]:
    """
    Convert ShareGPT format data with pre-computed token_ids to DataProto format.
    This avoids re-tokenization by using token_ids directly from the saved data.

    Args:
        sharegpt_data: ShareGPT format data with token_ids included
        tokenizer: HuggingFace tokenizer
        max_prompt_length: Maximum prompt length
        max_response_length: Maximum response length

    Returns:
        List of (output_dpr, response_text) tuples matching llm_async_generate format
    """
    trajectories = []
    conversations = sharegpt_data.get("conversations", [])

    # Parse conversations into prompt-response pairs
    system_message = ""
    conversation_pairs = []
    current_prompt_messages = []

    for msg in conversations:
        role = msg["from"]
        content = msg["value"]

        if role == "system":
            system_message = content
        elif role == "human":
            current_prompt_messages.append({"role": "user", "content": content})
        elif role == "gpt":
            # This is a response with token_ids
            if current_prompt_messages:
                conversation_pairs.append({
                    "prompt_messages": current_prompt_messages.copy(),
                    "response": content,
                    "token_ids": msg.get("token_ids", [])
                })
                current_prompt_messages.append({"role": "assistant", "content": content})
            else:
                logger.warning("Found GPT response without preceding prompt, skipping")

    # Process each prompt-response pair
    for pair in conversation_pairs:
        prompt_messages = pair["prompt_messages"]
        response_text = pair["response"]
        response_token_ids = pair["token_ids"]

        # Filter out padding tokens and special tokens if needed
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id

        # Remove padding and EOS tokens from token_ids
        response_token_ids = [tid for tid in response_token_ids
                             if tid not in (pad_token_id, eos_token_id, 0)]

        # Build the prompt (still need to tokenize prompt)
        messages_for_prompt = []
        if system_message:
            messages_for_prompt.append({"role": "system", "content": system_message})
        messages_for_prompt.extend(prompt_messages)

        old_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"

        try:
            # Format and tokenize prompt
            prompt_with_template = tokenizer.apply_chat_template(
                messages_for_prompt,
                add_generation_prompt=True,
                tokenize=False
            )

            prompt_inputs = tokenizer(
                prompt_with_template,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False
            )

            prompt_ids = prompt_inputs["input_ids"]
            prompt_attention_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_ids))

            # Truncate prompt if needed
            if prompt_ids.size(1) > max_prompt_length:
                logger.warning(f"Truncating prompt from {prompt_ids.size(1)} to {max_prompt_length}")
                prompt_ids = prompt_ids[:, -max_prompt_length:]
                prompt_attention_mask = prompt_attention_mask[:, -max_prompt_length:]

            # Convert response token_ids to tensor
            response_ids = torch.tensor([response_token_ids], dtype=torch.long)

            # Truncate response if needed
            if response_ids.size(1) > max_response_length:
                logger.warning(f"Truncating response from {response_ids.size(1)} to {max_response_length}")
                response_ids = response_ids[:, :max_response_length]

            # Pad prompt (left padding)
            prompt_ids_padded = pad_sequence_to_length(
                prompt_ids,
                max_seq_len=max_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True
            )
            prompt_attention_mask_padded = pad_sequence_to_length(
                prompt_attention_mask,
                max_seq_len=max_prompt_length,
                pad_token_id=0,
                left_pad=True
            )
            prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask_padded)

            # Pad response (right padding)
            response_ids_padded = pad_sequence_to_length(
                response_ids,
                max_seq_len=max_response_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=False
            )

            # Create response attention mask
            response_attention_mask = (response_ids_padded != tokenizer.pad_token_id).long()

            # Concatenate prompt and response
            input_ids = torch.cat([prompt_ids_padded, response_ids_padded], dim=-1)
            attention_mask = torch.cat([prompt_attention_mask_padded, response_attention_mask], dim=-1)

            # Create position ids for response
            response_length = response_ids_padded.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0)
            response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

            # Create DataProto
            batch_dict = {
                "prompts": prompt_ids_padded,
                "responses": response_ids_padded,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids
            }

            output_dpr = DataProto(
                batch=TensorDict(batch_dict, batch_size=1),
                non_tensor_batch={
                    "formatted_prompts": np.array([prompt_with_template])
                }
            )

            trajectories.append((output_dpr, response_text))

        finally:
            tokenizer.padding_side = old_padding_side

    return trajectories


def _tokenize_sharegpt_to_dataproto(
    sharegpt_data: dict,
    tokenizer,
    max_prompt_length: int,
    max_response_length: int
) -> List[Tuple[DataProto, str]]:
    """
    Tokenize ShareGPT format conversation data and convert to DataProto format
    matching llm_async_generate output.

    Args:
        sharegpt_data: ShareGPT format data with {"id": str, "conversations": [{"from": str, "value": str}]}
        tokenizer: HuggingFace tokenizer
        max_prompt_length: Maximum prompt length
        max_response_length: Maximum response length

    Returns:
        List of (output_dpr, response_text) tuples matching llm_async_generate format
    """
    trajectories = []
    conversations = sharegpt_data.get("conversations", [])

    # Parse conversations into prompt-response pairs
    system_message = ""
    conversation_pairs = []
    current_prompt_messages = []

    for msg in conversations:
        role = msg["from"]
        content = msg["value"]

        if role == "system":
            system_message = content
        elif role == "human":
            current_prompt_messages.append({"role": "user", "content": content})
        elif role == "gpt":
            # This is a response - create a pair
            if current_prompt_messages:
                conversation_pairs.append({
                    "prompt_messages": current_prompt_messages.copy(),
                    "response": content
                })
                # Add this response to the history for the next turn
                current_prompt_messages.append({"role": "assistant", "content": content})
            else:
                logger.warning("Found GPT response without preceding prompt, skipping")

    # Tokenize each prompt-response pair
    for pair in conversation_pairs:
        prompt_messages = pair["prompt_messages"]
        response_text = pair["response"]

        # Build the prompt with chat template
        messages_for_prompt = []
        if system_message:
            messages_for_prompt.append({"role": "system", "content": system_message})
        messages_for_prompt.extend(prompt_messages)

        old_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"

        try:
            # Format prompt
            prompt_with_template = tokenizer.apply_chat_template(
                messages_for_prompt,
                add_generation_prompt=True,
                tokenize=False
            )

            # Tokenize prompt
            prompt_inputs = tokenizer(
                prompt_with_template,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False
            )

            prompt_ids = prompt_inputs["input_ids"]
            prompt_attention_mask = prompt_inputs.get("attention_mask", torch.ones_like(prompt_ids))

            # Truncate prompt if needed
            if prompt_ids.size(1) > max_prompt_length:
                logger.warning(f"Truncating prompt from {prompt_ids.size(1)} to {max_prompt_length}")
                prompt_ids = prompt_ids[:, -max_prompt_length:]
                prompt_attention_mask = prompt_attention_mask[:, -max_prompt_length:]
                prompt_with_template = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)

            # Tokenize response
            response_inputs = tokenizer(
                response_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False
            )

            response_ids = response_inputs["input_ids"]

            # Truncate response if needed
            if response_ids.size(1) > max_response_length:
                logger.warning(f"Truncating response from {response_ids.size(1)} to {max_response_length}")
                response_ids = response_ids[:, :max_response_length]

            # Pad prompt (left padding)
            prompt_ids_padded = pad_sequence_to_length(
                prompt_ids,
                max_seq_len=max_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True
            )
            prompt_attention_mask_padded = pad_sequence_to_length(
                prompt_attention_mask,
                max_seq_len=max_prompt_length,
                pad_token_id=0,
                left_pad=True
            )
            prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask_padded)

            # Pad response (right padding)
            response_ids_padded = pad_sequence_to_length(
                response_ids,
                max_seq_len=max_response_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=False
            )

            # Create response attention mask
            response_attention_mask = (response_ids_padded != tokenizer.pad_token_id).long()

            # Concatenate prompt and response
            input_ids = torch.cat([prompt_ids_padded, response_ids_padded], dim=-1)
            attention_mask = torch.cat([prompt_attention_mask_padded, response_attention_mask], dim=-1)

            # Create position ids for response
            response_length = response_ids_padded.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0)
            response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
            position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)

            # Create DataProto
            batch_dict = {
                "prompts": prompt_ids_padded,
                "responses": response_ids_padded,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids
            }

            output_dpr = DataProto(
                batch=TensorDict(batch_dict, batch_size=1),
                non_tensor_batch={
                    "formatted_prompts": np.array([prompt_with_template])
                }
            )

            trajectories.append((output_dpr, response_text))

        finally:
            tokenizer.padding_side = old_padding_side

    return trajectories
