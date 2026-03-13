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
import subprocess
import sys

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




def _ensure_math_datasets(datasets_dir: Path, dataset_names: List[str], mode: str = "train") -> None:
    """Run load_math.py if any required parquet files are missing."""
    subdir = "train" if mode == "train" else "test"
    missing = [
        name for name in dataset_names
        if not (datasets_dir / subdir / f"{name}.parquet").exists()
    ]
    if not missing:
        return
    print(f"[math_utils] Missing dataset(s): {missing}. Running load_math.py ...")
    script = Path(__file__).parent.parent.parent.parent / "scripts" / "dataprocess" / "load_math.py"
    if not script.exists():
        raise FileNotFoundError(f"load_math.py not found at: {script}")
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    result = subprocess.run([sys.executable, str(script)], check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"load_math.py failed with return code {result.returncode}")
    still_missing = [
        name for name in dataset_names
        if not (datasets_dir / subdir / f"{name}.parquet").exists()
    ]
    if still_missing:
        raise FileNotFoundError(f"Dataset(s) still missing after running load_math.py: {still_missing}")
    print("[math_utils] Datasets ready.")


def _load_single_dataset_train(parquet_file: Path, n: int) -> List[Dict[str, Any]]:
    """Load n samples from a single parquet file (train mode)."""
    if not parquet_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {parquet_file}")
    print(f"Loading dataset from: {parquet_file}")
    ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
    print(f"Dataset loaded: {len(ds)} samples")
    if len(ds) < n:
        print(f"Warning: Dataset has {len(ds)} samples, but {n} requested. Will sample with replacement.")
        indices = [random.randint(0, len(ds) - 1) for _ in range(n)]
    else:
        indices = random.sample(range(len(ds)), n)
    results = []
    for idx in indices:
        problem_dict = _format_math_problem(ds[idx], idx, mode="train")
        if problem_dict:
            results.append(problem_dict)
    return results


_UNSET = object()  # sentinel to detect un-passed dataset_name


def load_math_problem_batch(
    env_indices: List[int],
    mode: str = "train",
    dataset_name=_UNSET,
    config: dict = None,
    benchmark_name: str = "AIME24"
) -> List[Dict[str, Any]]:

    # Only fall back to config.env.dataset when the caller did NOT pass dataset_name
    # explicitly. If called from mixed_env.py with dataset_name=dataset_math, that
    # explicit value is kept and config.env.dataset is ignored.
    if dataset_name is _UNSET:
        if config is not None and hasattr(config, "env") and hasattr(config.env, "dataset"):
            dataset_name = config.env.dataset
        else:
            dataset_name = "polaris"

    current_dir = Path(__file__).parent.parent.parent.parent
    local_datasets_dir = current_dir / "data" / "math"

    # OmegaConf ListConfig is not a plain list/tuple — normalise to List[str] robustly
    def _to_str_list(v) -> List[str]:
        if isinstance(v, str):
            return [v]
        try:
            return [str(x) for x in v]
        except TypeError:
            return [str(v)]

    if mode != "train":
        benchmark_list = _to_str_list(benchmark_name)
        _ensure_math_datasets(local_datasets_dir, benchmark_list, mode="test")
        batch_results = []
        for bname in benchmark_list:
            parquet_file = local_datasets_dir / "test" / f"{bname}.parquet"
            if not parquet_file.exists():
                raise FileNotFoundError(f"Dataset file not found: {parquet_file}")
            print(f"Loading dataset from: {parquet_file}")
            ds = hf_load_dataset("parquet", data_files=str(parquet_file), split="train")
            print(f"Dataset loaded: {len(ds)} samples from {bname}")
            for i, example in enumerate(ds):
                problem_dict = _format_math_problem(example, i, mode="validate")
                if problem_dict:
                    batch_results.append(problem_dict)
        print(f"Returning {len(batch_results)} samples from benchmarks: {benchmark_list}")
        return batch_results

    # Train mode: support single dataset name or list for mixed training
    n_total = len(env_indices)
    train_names = _to_str_list(dataset_name)
    _ensure_math_datasets(local_datasets_dir, train_names, mode="train")
    if len(train_names) > 1:
        # Multi-dataset: split evenly (half-half for 2 datasets)
        n_datasets = len(train_names)
        base = n_total // n_datasets
        counts = [base] * n_datasets
        for i in range(n_total % n_datasets):
            counts[i] += 1

        batch_results = []
        for ds_name, n in zip(train_names, counts):
            parquet_file = local_datasets_dir / "train" / f"{ds_name}.parquet"
            samples = _load_single_dataset_train(parquet_file, n)
            print(f"Sampled {len(samples)} from {ds_name}")
            batch_results.extend(samples)

        random.shuffle(batch_results)
        print(f"Returning {len(batch_results)} mixed samples from {train_names}")
        return batch_results
    else:
        parquet_file = local_datasets_dir / "train" / f"{train_names[0]}.parquet"
        batch_results = _load_single_dataset_train(parquet_file, n_total)
        print(f"Returning {len(batch_results)} samples from {train_names[0]}")
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