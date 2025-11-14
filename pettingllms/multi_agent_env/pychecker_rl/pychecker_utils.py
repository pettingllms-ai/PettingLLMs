"""
Utility functions for PyChecker RL environment.

This module contains utilities for code parsing, workflow management,
and simulation execution for hardware verification tasks.
"""

import os
import json
import logging
import subprocess
import tempfile
import re
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

logger = logging.getLogger(__name__)


# ============================================================================
# Code Extraction and Parsing Utilities
# ============================================================================

def extract_code_from_response(response: str, code_type: str = "python") -> str:
    """
    Extract code from LLM response with flexible parsing.

    Args:
        response: LLM response string
        code_type: Type of code to extract ("python", "verilog", etc.)

    Returns:
        Extracted code string, or empty string if no code block found
    """
    if not response:
        return ""

    # Try specific language pattern first
    if code_type == "python":
        pattern = r'```python\s*(.*?)```'
    elif code_type == "verilog":
        pattern = r'```(?:verilog|systemverilog|sv)\s*(.*?)```'
    else:
        pattern = f'```{code_type}\\s*(.*?)```'

    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Try generic code block pattern
    generic_pattern = r'```\s*(.*?)```'
    matches = re.findall(generic_pattern, response, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # If no code blocks found, return empty string
    logger.warning(f"No code block found in response for code_type={code_type}")
    return ""


# ============================================================================
# Task Folder Management
# ============================================================================

def create_task_folder(base_dir: str, env_idx: int, rollout_idx: int, turn_idx: int, worker_id: int = None) -> str:
    """
    Create task-specific folder for storing artifacts

    Args:
        base_dir: Base directory for task folders
        env_idx: Environment index
        rollout_idx: Rollout index
        turn_idx: Turn index
        worker_id: Worker ID for path isolation (optional, will be computed if not provided)

    Returns:
        Path to task folder
    """
    if worker_id is None:
        # Fallback: compute worker_id deterministically
        worker_id = rollout_idx % 512  # default num_workers

    # Include worker_id in path for isolation between concurrent tasks
    task_folder = os.path.join(base_dir, f"worker_{worker_id}", f"env_{env_idx}", f"rollout_{rollout_idx}", f"turn_{turn_idx}")
    os.makedirs(task_folder, exist_ok=True)
    return task_folder


# ============================================================================
# Simulation Result Analysis
# ============================================================================

def check_compile_success(log_content: str) -> bool:
    """Check if simulation succeeded (can be simulated, has Unpass info)"""
    # Check if log has "Unpass:" which means simulation ran successfully
    if re.search(r'Unpass:', log_content):
        return True
    
    # If simulation started and ran, it's successful
    if re.search(r'========== Test Vector \d+ ==========', log_content):
        return True
    if re.search(r'sim finished', log_content):
        return True
    
    return False


def check_simulation_pass(log_content: str) -> bool:
    """Check if simulation passes (Unpass: 0 means all tests passed)"""
    # Check for "Unpass: 0" which means all tests passed
    unpass_match = re.search(r'Unpass:\s*(\d+)', log_content)
    if unpass_match:
        unpass_count = int(unpass_match.group(1))
        return unpass_count == 0
    
    # Fallback: check for error indicators
    if re.search(r'\bERROR\b', log_content):
        return False
    if re.search(r'\berror\b', log_content):
        return False
    if re.search(r'\bmismatch\b', log_content, re.IGNORECASE):
        return False
    if re.search(r'\bFAIL(?:ED)?\b', log_content, re.IGNORECASE):
        return False
    if re.search(r'Total failures:\s*[1-9]', log_content):
        return False

    returncode_match = re.search(r'Return code:\s*(\d+)', log_content)
    if returncode_match:
        if int(returncode_match.group(1)) != 0:
            return False
    
    return True


# ============================================================================
# Python Execution Utilities
# ============================================================================

# ============================================================================
# Verilog Simulation Utilities
# ============================================================================



# ============================================================================
# Dataset Loading Utilities
# ============================================================================

def load_pychecker_problem_batch(
    env_indices: List[int],
    mode: str = "train",
    dataset_name: str = "pychecker",
    config: dict = None
) -> List[Dict[str, Any]]:
    """
    Load a batch of PyChecker problems from dataset

    Args:
        env_indices: List of environment indices (used to determine sample count for train mode)
        mode: "train" or "validate"
        dataset_name: Name of the dataset
        config: Configuration dictionary

    Returns:
        List of problem dictionaries
    """
    import random

    current_dir = Path(__file__).parent.parent.parent.parent
    local_datasets_dir = current_dir / "data" / "pychecker_rl"

    if mode == "train":
        dataset_file = local_datasets_dir / "dataset" / "train.jsonl"
    else:
        dataset_file = local_datasets_dir / "dataset" / "test.jsonl"

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    print(f"Loading dataset from: {dataset_file}")

    # Load JSONL file
    all_problems = []
    with open(dataset_file, 'r') as f:
        for line in f:
            if line.strip():
                all_problems.append(json.loads(line.strip()))

    print(f"Dataset loaded: {len(all_problems)} samples")

    batch_results = []

    if mode == "train":
        # For train mode: randomly sample N problems where N = len(env_indices)
        sample_num = len(env_indices)
        if sample_num > len(all_problems):
            raise ValueError(f"Dataset has {len(all_problems)} samples, but {sample_num} requested")

        # Randomly sample without replacement
        indices = random.sample(range(len(all_problems)), sample_num)
        for idx in indices:
            problem_dict = _format_pychecker_problem(all_problems[idx], idx, mode="train")
            if problem_dict:
                batch_results.append(problem_dict)
    else:
        # For validate mode: load all problems
        for i, problem in enumerate(all_problems):
            problem_dict = _format_pychecker_problem(problem, i, mode="validate")
            if problem_dict:
                batch_results.append(problem_dict)

    print(f"Returning {len(batch_results)} samples")
    return batch_results


def _format_pychecker_problem(example: Dict, index: int, mode: str = "train") -> Optional[Dict]:
    """
    Format a PyChecker problem example into a standardized dictionary.
    
    Args:
        example: Raw example from dataset
        index: Index of the example
        mode: "train" or "validate"
        
    Returns:
        Formatted problem dictionary or None if invalid
    """
    problem_input = example.get("problem_input", "")
    golden_output = example.get("golden_output", "")
    circuit_type = example.get("circuit_type", "CMB")
    
    if not problem_input or not golden_output:
        print(f"Warning: Skipping example {index}: missing required fields")
        return None
    
    return {
        "problem_input": problem_input,
        "golden_output": golden_output,
        "circuit_type": circuit_type
    }


def calculate_reward(success: bool, results: Dict[str, Any]) -> float:
    """
    Calculate reward based on execution results.
    
    Reward scheme:
    - 0.0: Python execution failed or simulation failed to compile
    - 0.3: Simulation compiled successfully but tests failed
    - 1.0: All tests passed
    
    Args:
        success: Whether the workflow succeeded
        results: Results dictionary from simulation
        
    Returns:
        Reward value (0.0, 0.3, or 1.0)
    """
    if not success:
        return 0.0
    
    compile_success = results.get("compile_success", False)
    all_tests_passed = results.get("all_tests_passed", False)
    
    if not compile_success:
        return 0.0
    
    if all_tests_passed:
        return 1.0
    else:
        return 0.0

