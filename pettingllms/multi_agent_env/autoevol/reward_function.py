"""
Reward functions for evaluating MAS performance on different task types.

Each reward function takes the result summary from the MAS execution
and the environment data, and returns a reward score.
"""

import re
import logging
from typing import Any
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


def extract_answer_from_summary(summary: str) -> str:
    """
    Extract answer from MAS summary output.

    Looks for common patterns like:
    - "Exact Answer: ..."
    - "Final Answer: ..."
    - "Answer: ..."
    - "... is <number>."
    - "... is <number>"

    Args:
        summary: The summary text from MAS execution

    Returns:
        Extracted answer string
    """
    # Try to find "Exact Answer:" pattern
    match = re.search(r"Output:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try "Final Answer:" pattern
    match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try "Answer:" pattern
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", summary, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to find pattern like "... is <number>." or "... is <number>"
    # This matches sentences ending with "is <number>" (with optional period)
    match = re.search(r'is\s+(-?\d+(?:\.\d+)?|\\boxed\{[^}]+\})\s*\.?\s*$', summary, re.MULTILINE | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # If it's a boxed answer, extract the content
        boxed_match = re.match(r'\\boxed\{([^}]+)\}', answer)
        if boxed_match:
            return boxed_match.group(1).strip()
        return answer

    # Try to find \boxed{} anywhere in the text
    match = re.search(r'\\boxed\{([^}]+)\}', summary)
    if match:
        return match.group(1).strip()

    # Fallback: return the last line
    lines = [line.strip() for line in summary.split('\n') if line.strip()]
    return lines[-1] if lines else summary.strip()


def math_reward_function(summary: str, env_data: Env) -> float:
    """
    Calculate reward for math tasks by comparing predicted answer with ground truth.

    Args:
        summary: The result summary from MAS execution
        env_data: Environment data containing the ground truth answer

    Returns:
        Reward score (1.0 if correct, 0.0 if incorrect)
    """
    try:
        from math_verify import parse, verify
    except ImportError:
        logger.error("math_verify module not found. Please install it for math verification.")
        return 0.0

    # Extract predicted answer from summary
    predicted_answer = extract_answer_from_summary(summary)

    # Get ground truth answer from env_data
    ground_truth = getattr(env_data, 'ground_truth_answer', None)
    if ground_truth is None:
        logger.warning("No ground truth answer found in env_data")
        return 0.0

    try:
        # Parse both answers
        parsed_pred = parse(predicted_answer)
        parsed_gt = parse(str(ground_truth))

        # Verify if they match
        is_correct = verify(parsed_pred, parsed_gt)

        reward = 1.0 if is_correct else 0.0
        logger.info(f"Math verification: pred={predicted_answer}, gt={ground_truth}, correct={is_correct}")

        return reward

    except Exception as e:
        logger.error(f"Error in math verification: {e}")
        # Fallback to simple string comparison
        return 1.0 if predicted_answer.strip().lower() == str(ground_truth).strip().lower() else 0.0


def code_reward_function(summary: str, env_data: Env) -> float:
    """
    Calculate reward for code generation tasks.

    Args:
        summary: The result summary from MAS execution
        env_data: Environment data containing test cases or expected output

    Returns:
        Reward score based on code correctness
    """
    # TODO: Implement code verification logic
    # This would typically involve:
    # 1. Extract generated code from summary
    # 2. Run test cases from env_data
    # 3. Calculate pass rate

    logger.warning("code_reward_function not fully implemented yet")
    return 0.0


def qa_reward_function(summary: str, env_data: Env) -> float:
    """
    Calculate reward for QA tasks by comparing answer similarity.

    Args:
        summary: The result summary from MAS execution
        env_data: Environment data containing the ground truth answer

    Returns:
        Reward score based on answer correctness
    """
    # Extract predicted answer
    predicted_answer = extract_answer_from_summary(summary)

    # Get ground truth answer
    ground_truth = getattr(env_data, 'answer', None)
    if ground_truth is None:
        logger.warning("No ground truth answer found in env_data")
        return 0.0

    # Simple exact match for now
    # Could be enhanced with fuzzy matching or semantic similarity
    pred_normalized = predicted_answer.strip().lower()
    gt_normalized = str(ground_truth).strip().lower()

    reward = 1.0 if pred_normalized == gt_normalized else 0.0
    logger.info(f"QA verification: pred={predicted_answer}, gt={ground_truth}, correct={reward > 0.5}")

    return reward


# Registry of reward functions by task type
REWARD_FUNCTIONS = {
    "math": math_reward_function,
    "code": code_reward_function,
    "qa": qa_reward_function,
}


def get_reward_function(task_type: str):
    """
    Get the reward function for a specific task type.

    Args:
        task_type: The type of task (e.g., "math", "code", "qa")

    Returns:
        The corresponding reward function, or None if not found
    """
    return REWARD_FUNCTIONS.get(task_type.lower())