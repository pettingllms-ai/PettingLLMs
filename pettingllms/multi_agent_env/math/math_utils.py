"""
Utility functions for mathematical problem solving and evaluation.

This module contains utilities for loading math datasets, evaluating solutions,
and computing metrics for mathematical problem solving tasks.
"""

import os
import json
import random
import asyncio
import re

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union
from datasets import load_dataset as hf_load_dataset


def extract_reasoning_steps(response: str):
    """
    Extract reasoning steps from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        Extracted reasoning steps
    """
    match = re.search(r"\*\*Reasoning Steps:\*\*\s*```(.*?)```", response, re.DOTALL)
    if not match:
        return []
    
    steps_block = match.group(1).strip()
    
    steps = [line.strip() for line in steps_block.split("\n") if line.strip()]
    return steps

def extract_code(response: str) -> str:
    """
    Extract code from agent response.
    
    Args:
        response: Agent response string
        
    Returns:
        Extracted code string
    """
    if response is None:
        return ""
    python_pattern = r'```python\s*(.*?)```'
    matches = re.findall(python_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    code_pattern = r'```\s*(.*?)```'
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[-1].strip()
    
    return response.strip()




def load_math_problem_batch(
    env_indices: List[int],
    mode: str = "train",
    dataset_name: str = "polaris",
    config: dict = None,
    benchmark_name: str = "AIME24"
) -> List[Dict[str, Any]]:
    

    current_dir = Path(__file__).parent.parent.parent.parent
    local_datasets_dir = current_dir / "data" / "math"
    
    if mode == "train":
        parquet_file = local_datasets_dir /"train" /f"{dataset_name}.parquet"
    else:
        parquet_file = local_datasets_dir /"test" / f"{benchmark_name}.parquet"
    

    if not parquet_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {parquet_file}")
    
    print(f"Loading dataset from: {parquet_file}")
    ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
    print(f"Dataset loaded: {len(ds)} samples")
    
    batch_results = []
    
    if mode == "train":
        if len(ds) < len(env_indices):
            raise ValueError(f"Dataset has {len(ds)} samples, but {len(env_indices)} requested")
        
        indices = random.sample(range(len(ds)), len(env_indices))
        for idx in indices:
            problem_dict = _format_math_problem(ds[idx], idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
    else:
        for i, example in enumerate(ds):
            problem_dict = _format_math_problem(example, i, mode="validate")
            if problem_dict:
                batch_results.append(problem_dict)
    
    print(f"Returning {len(batch_results)} samples")
    return batch_results


def _format_math_problem(example: Dict, index: int, mode: str = "train") -> Optional[Dict]:
    """
    Format a math problem example into a standardized dictionary.
    
    Args:
        example: Raw example from dataset (expected format: question/solution)
        index: Index of the example
        mode: "train" or "validate"
        
    Returns:
        Formatted problem dictionary or None if invalid
    """
    question = example.get("question", "")
    solution = example.get("solution", "")
    answer = solution
    
    if not question:
        print(f"Warning: Skipping example {index}: missing question field")
        return None
    
    return {
        "question": question,
        "solution": answer
    }