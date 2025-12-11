#!/usr/bin/env python3
"""
Test script to verify the sample_num fix for multi-sampling in agents.
"""

import torch
import numpy as np
from tensordict import TensorDict

# Mock the DataProto class for testing
class DataProto:
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info

def test_postprocess_batch_with_multiple_samples():
    """Test that postprocess_batch correctly handles n > 1"""
    print("Testing postprocess_batch with sample_num > 1...")
    
    # Create mock input batch with batch_size=2
    batch_size = 2
    prompt_length = 10
    
    # Create mock prompts (left-padded)
    prompts = torch.randint(0, 1000, (batch_size, prompt_length))
    attention_mask = torch.ones((batch_size, prompt_length))
    position_ids = torch.arange(prompt_length).unsqueeze(0).expand(batch_size, -1)
    
    mock_batch = TensorDict({
        "input_ids": prompts,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, batch_size=batch_size)
    
    mock_non_tensor = {
        "formatted_prompts": ["prompt1", "prompt2"]
    }
    
    input_dpr = DataProto(batch=mock_batch, non_tensor_batch=mock_non_tensor)
    
    # Create mock response_ids: 2 batches, each with 3 samples (n=3)
    n = 3
    response_ids = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # 3 responses for prompt1
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]]  # 3 responses for prompt2
    ]
    
    pad_token_id = 0
    eos_token_id = 2
    max_response_length = 10
    max_prompt_length = 20
    
    # Import the actual function
    import sys
    sys.path.insert(0, '/home/lah003/workspace/verl_efficient')
    from pettingllms.trainer.async_generate import postprocess_batch
    
    # Run postprocess_batch
    output_dpr = postprocess_batch(
        input_dpr, 
        response_ids, 
        n, 
        pad_token_id, 
        eos_token_id,
        max_response_length,
        max_prompt_length
    )
    
    # Verify output shape
    expected_output_batch_size = batch_size * n  # 2 * 3 = 6
    actual_batch_size = output_dpr.batch["input_ids"].shape[0]
    
    print(f"✓ Input batch size: {batch_size}")
    print(f"✓ Sample num (n): {n}")
    print(f"✓ Expected output batch size: {expected_output_batch_size}")
    print(f"✓ Actual output batch size: {actual_batch_size}")
    
    assert actual_batch_size == expected_output_batch_size, \
        f"Batch size mismatch: expected {expected_output_batch_size}, got {actual_batch_size}"
    
    # Verify prompts are replicated correctly
    prompts_output = output_dpr.batch["prompts"]
    print(f"✓ Output prompts shape: {prompts_output.shape}")
    
    # Each original prompt should appear n times consecutively
    for i in range(batch_size):
        for j in range(n):
            idx = i * n + j
            assert torch.equal(prompts_output[idx], prompts[i]), \
                f"Prompt replication error at position {idx}"
    
    print("✓ Prompts replicated correctly")
    
    # Verify responses shape
    responses_output = output_dpr.batch["responses"]
    print(f"✓ Output responses shape: {responses_output.shape}")
    assert responses_output.shape[0] == expected_output_batch_size
    
    # Verify non_tensor_batch is replicated
    if "formatted_prompts" in output_dpr.non_tensor_batch:
        formatted_prompts_output = output_dpr.non_tensor_batch["formatted_prompts"]
        print(f"✓ Formatted prompts length: {len(formatted_prompts_output)}")
        assert len(formatted_prompts_output) == expected_output_batch_size, \
            f"Non-tensor batch size mismatch: expected {expected_output_batch_size}, got {len(formatted_prompts_output)}"
        print("✓ Non-tensor batch replicated correctly")
    
    print("\n✅ All tests passed! The fix is working correctly.\n")

if __name__ == "__main__":
    test_postprocess_batch_with_multiple_samples()
