"""
Reward functions for evaluating MAS performance on different task types.

Each reward function takes the result summary from the MAS execution
and the environment data, and returns a reward score.
"""

import re
import logging
from typing import Any
from pettingllms.multi_agent_env.base.env import Env

# Suppress AutoGen/AG2 logging warnings
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
logging.getLogger("autogen").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def extract_answer_from_summary(summary: str) -> str:
    """
    Extract answer from MAS summary output between WORKFLOW_SUMMARY_START and WORKFLOW_SUMMARY_END.

    This function simply extracts the last consecutive number string (including fractions and decimals)
    from the summary section.

    Args:
        summary: The summary text from MAS execution (could be full output or just the summary section)

    Returns:
        Extracted answer string (the last number found in the summary)
    """
    # First, try to extract content between WORKFLOW_SUMMARY_START and WORKFLOW_SUMMARY_END markers
    workflow_match = re.search(
        r'WORKFLOW_SUMMARY_START\s*(.*?)\s*WORKFLOW_SUMMARY_END',
        summary,
        re.DOTALL | re.IGNORECASE
    )

    if workflow_match:
        # Use only the content between the markers
        summary_content = workflow_match.group(1).strip()
    else:
        # If markers not found, use the entire summary
        summary_content = summary

    # Find all consecutive number strings (including fractions and decimals)
    # Pattern matches: "33", "3.14", "-5", "25/8"
    number_pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
    numbers = re.findall(number_pattern, summary_content)

    if numbers:
        # Return the last number found
        last_number = numbers[-1]
        logger.info(f"Extracted last number from summary: {last_number} (found {len(numbers)} total)")
        return last_number

    # Fallback: return the last non-empty line
    lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
    return lines[-1] if lines else summary_content.strip()


def math_reward_function(summary: str, env_data: Env) -> float:
    """
    Calculate reward for math tasks by comparing predicted answer with ground truth.

    Args:
        summary: The result summary from MAS execution
        env_data: Environment data containing the ground truth answer

    Returns:
        Reward score (1.0 if correct, 0.0 if incorrect)
    """
    from math_verify import parse, verify

    # Extract predicted answer from summary
    predicted_answer = extract_answer_from_summary(summary)

    # Get ground truth answer from env_data.state
    ground_truth = None
    ground_truth = env_data.state.ground_truth_answer

    # Parse both answers
    parsed_gt = parse(str(ground_truth))
    # Verify if they match
    is_correct = verify(predicted_answer, parsed_gt)

    reward = 1.0 if is_correct else 0.0
    logger.info(f"Math verification: pred={predicted_answer}, gt={ground_truth}, correct={is_correct}")

    return reward


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