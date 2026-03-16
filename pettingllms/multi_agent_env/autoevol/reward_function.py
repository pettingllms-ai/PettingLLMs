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

    The summary parameter is actually the already-extracted final answer from _calculate_reward.

    Args:
        summary: The extracted final answer (not the full summary)
        env_data: Environment data containing the ground truth answer

    Returns:
        Reward score (1.0 if correct, 0.0 if incorrect)
    """
    from math_verify import parse, verify

    # The summary is actually the already-extracted answer
    predicted_answer = summary.strip()

    # Get ground truth answer from env_data.state
    ground_truth = env_data.state.ground_truth_answer
    gt_str = str(ground_truth).strip()

    # Helper: strip common non-mathematical decorations so that e.g.
    # "8:32 AM" matches "8:32", "42 minutes" matches "42", "90°" matches "90".
    def strip_decorations(s):
        s = s.strip()
        # Remove trailing AM/PM (case-insensitive)
        s = re.sub(r'\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)\s*$', '', s)
        # Remove trailing unit words
        s = re.sub(
            r'\s*(?:minutes?|mins?|hours?|hrs?|seconds?|secs?|degrees?|'
            r'meters?|centimeters?|cm|km|dollars?|USD|\$|%|percent)\s*$',
            '', s, flags=re.IGNORECASE,
        )
        # Remove trailing ° symbol and \circ
        s = re.sub(r'\s*(?:°|\\circ)\s*$', '', s)
        # Remove wrapping \text{...}
        s = re.sub(r'^\\text\{(.*)\}$', r'\1', s)
        return s.strip()

    # Helper: normalize LaTeX for comparison
    def normalize_latex(s):
        s = re.sub(r'\\\\', r'\\', s)  # Double backslash -> single
        s = re.sub(r'\\dfrac', r'\\frac', s)  # dfrac -> frac
        s = re.sub(r'\\left|\\right', '', s)  # Remove \left \right
        s = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', s)  # Remove \mathrm{}
        s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)  # Remove \text{}
        s = re.sub(r'\s+', '', s)  # Remove whitespace
        return s

    # Helper: convert LaTeX fractions to plain fractions (\frac{1}{43} -> 1/43)
    def latex_frac_to_plain(s):
        # Iteratively replace \frac{a}{b} with (a)/(b)
        prev = None
        while prev != s:
            prev = s
            s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
        # Clean up unnecessary parentheses for simple numerator/denominator
        s = re.sub(r'\((\w+)\)/\((\w+)\)', r'\1/\2', s)
        return s

    # Helper: numeric comparison with tolerance (handles rounding like 53.53 vs 53.50)
    def numeric_equal(a, b, rel_tol=0.005):
        try:
            def to_float(s):
                s = s.strip()
                if '/' in s and not s.startswith('\\'):
                    parts = s.split('/')
                    if len(parts) == 2:
                        return float(parts[0].strip()) / float(parts[1].strip())
                return float(s)
            fa, fb = to_float(a), to_float(b)
            if fa == fb:
                return True
            denom = max(abs(fa), abs(fb), 1e-10)
            return abs(fa - fb) / denom <= rel_tol
        except (ValueError, ZeroDivisionError):
            return False

    # Helper: extract multiple choice letter (A-E)
    def extract_choice_letter(s):
        """Extract choice letter from formats like 'A', '(A)', '\mathrm{(A)}', 'A.', 'A)'"""
        s = s.strip()
        # Pattern: \mathrm{(X)} or \textbf{(X)}
        match = re.search(r'\\(?:mathrm|textbf)\{\(([A-Ea-e])\)', s)
        if match:
            return match.group(1).upper()
        # Pattern: (X) at start or standalone
        match = re.search(r'^\(([A-Ea-e])\)', s)
        if match:
            return match.group(1).upper()
        # Pattern: single letter at start (with optional punctuation after)
        match = re.match(r'^([A-Ea-e])(?:[.\)\s:]|$)', s)
        if match:
            return match.group(1).upper()
        return None

    # 1. Direct exact match (case-insensitive)
    pred_lower = predicted_answer.lower().strip()
    gt_lower = gt_str.lower().strip()
    if pred_lower == gt_lower:
        logger.info(f"Math verification (exact match): pred={predicted_answer}, gt={ground_truth}, correct=True")
        return 1.0

    # 1.5. Decoration-stripped match (handles "8:32 AM" vs "8:32",
    #       "90°" vs "90", "42 minutes" vs "42", etc.)
    pred_stripped = strip_decorations(predicted_answer)
    gt_stripped = strip_decorations(gt_str)
    if pred_stripped and gt_stripped and pred_stripped.lower() == gt_stripped.lower():
        logger.info(f"Math verification (decoration-stripped): pred={predicted_answer}, gt={ground_truth}, correct=True")
        return 1.0

    # 2. Normalized LaTeX match
    pred_latex = normalize_latex(predicted_answer)
    gt_latex = normalize_latex(gt_str)
    if pred_latex == gt_latex:
        logger.info(f"Math verification (latex normalized): pred={predicted_answer}, gt={ground_truth}, correct=True")
        return 1.0

    # 2.5. Decoration-stripped + LaTeX normalized
    pred_stripped_latex = normalize_latex(pred_stripped)
    gt_stripped_latex = normalize_latex(gt_stripped)
    if pred_stripped_latex and gt_stripped_latex and pred_stripped_latex == gt_stripped_latex:
        logger.info(f"Math verification (stripped+latex): pred={predicted_answer}, gt={ground_truth}, correct=True")
        return 1.0

    # 2.6. LaTeX fraction to plain fraction (\frac{1}{43} -> 1/43)
    pred_plain_frac = latex_frac_to_plain(normalize_latex(predicted_answer))
    gt_plain_frac = latex_frac_to_plain(normalize_latex(gt_str))
    if pred_plain_frac and gt_plain_frac and pred_plain_frac == gt_plain_frac:
        logger.info(f"Math verification (frac_to_plain): pred={predicted_answer}, gt={ground_truth}, correct=True")
        return 1.0

    # 2.7. Numeric tolerance (handles rounding: 53.53 vs 53.50)
    for a, b, label in [
        (predicted_answer, gt_str, "numeric"),
        (pred_stripped, gt_stripped, "numeric stripped"),
        (pred_plain_frac, gt_plain_frac, "numeric frac"),
    ]:
        if numeric_equal(a, b):
            logger.info(f"Math verification ({label}): pred={predicted_answer}, gt={ground_truth}, correct=True")
            return 1.0

    # 3. Multiple choice: compare extracted letters
    pred_letter = extract_choice_letter(predicted_answer)
    gt_letter = extract_choice_letter(gt_str)
    if pred_letter and gt_letter and pred_letter == gt_letter:
        logger.info(f"Math verification (choice letter): pred={pred_letter}, gt={gt_letter}, correct=True")
        return 1.0

    # 4. Try math_verify for mathematical expressions
    for pred_v, gt_v, label in [
        (predicted_answer, gt_str, "math_verify"),
        (pred_latex, gt_latex, "math_verify normalized"),
        (pred_stripped, gt_stripped, "math_verify stripped"),
        (pred_plain_frac, gt_plain_frac, "math_verify frac"),
    ]:
        try:
            parsed_gt = parse(gt_v)
            is_correct = verify(pred_v, parsed_gt)
            if is_correct:
                logger.info(f"Math verification ({label}): pred={predicted_answer}, gt={ground_truth}, correct=True")
                return 1.0
        except Exception as e:
            logger.debug(f"{label} failed: {e}")
        # Also try parsing both sides (handles \dfrac{1}{43} vs 1/43)
        try:
            parsed_pred = parse(pred_v)
            parsed_gt = parse(gt_v)
            is_correct = verify(parsed_pred, parsed_gt)
            if is_correct:
                logger.info(f"Math verification ({label} both-parsed): pred={predicted_answer}, gt={ground_truth}, correct=True")
                return 1.0
        except Exception as e:
            logger.debug(f"{label} both-parsed failed: {e}")

    logger.info(f"Math verification: pred={predicted_answer}, gt={ground_truth}, correct=False")
    return 0.0


def code_reward_function(summary: str, env_data: Env) -> float:
    """
    Calculate reward for code generation tasks.

    Extracts code from the MAS summary, runs it against ground-truth test
    cases stored in ``env_data.state``, and returns the pass ratio.

    Args:
        summary: The result summary / final answer from MAS execution
        env_data: Environment data containing ``state.ground_truth_test_input``
                  and ``state.ground_truth_test_output``

    Returns:
        Reward score (pass ratio across test cases, 0.0 – 1.0)
    """
    import subprocess
    import tempfile
    import os

    # --- 1. Extract code from summary ---
    code = _extract_code_block(summary)
    if not code:
        logger.info("code_reward_function: no code block found in summary")
        return 0.0

    # --- 2. Get ground-truth test cases ---
    test_inputs = getattr(env_data.state, "ground_truth_test_input", None) or []
    test_outputs = getattr(env_data.state, "ground_truth_test_output", None) or []

    if not test_inputs or not test_outputs:
        logger.warning("code_reward_function: no test cases available, returning 0.0")
        return 0.0

    n_tests = min(len(test_inputs), len(test_outputs))
    passed = 0
    timeout_per_case = 10  # seconds

    for i in range(n_tests):
        try:
            # Write code to a temp file and execute with subprocess
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                ["python3", tmp_path],
                input=str(test_inputs[i]),
                capture_output=True,
                text=True,
                timeout=timeout_per_case,
            )
            actual = result.stdout.strip()
            expected = str(test_outputs[i]).strip()

            # Whitespace-normalized comparison
            if " ".join(actual.split()) == " ".join(expected.split()):
                passed += 1
            else:
                logger.debug(
                    f"Test {i} failed: expected={expected!r}, got={actual!r}"
                )
        except subprocess.TimeoutExpired:
            logger.debug(f"Test {i} timed out after {timeout_per_case}s")
        except Exception as e:
            logger.debug(f"Test {i} execution error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    reward = passed / n_tests if n_tests > 0 else 0.0
    logger.info(
        f"code_reward_function: {passed}/{n_tests} tests passed, reward={reward:.2f}"
    )
    return reward


def _extract_code_block(text: str) -> str:
    """Extract code from text, trying multiple formats.

    Priority:
    0. If text contains [AGENT RESPONSE]: markers (noisy subprocess stdout),
       extract the last clean agent response and apply the rest of the logic on it.
    1. <solution>...</solution> tags
    2. <tool_call> execute_code JSON "code" field (model uses tool call without <solution>)
    2b. <code>```python...```</code> tags
    3. ```python ... ``` blocks
    4. Generic ``` ... ``` blocks
    5. Raw text if it looks like code
    """
    # 0. Extract from noisy subprocess stdout if [AGENT RESPONSE]: markers present
    if "[AGENT RESPONSE]:" in text:
        # Split on [AGENT RESPONSE]: and take the last section
        parts = text.split("[AGENT RESPONSE]:")
        if len(parts) > 1:
            last_response = parts[-1]
            # Trim at [TOKENS]: marker if present (end of agent response section)
            tokens_idx = last_response.find("[TOKENS]:")
            if tokens_idx != -1:
                last_response = last_response[:tokens_idx]
            # Also trim at next ========== AGENT NODE line if present
            node_idx = last_response.find("==========")
            if node_idx != -1:
                last_response = last_response[:node_idx]
            last_response = last_response.strip()
            if last_response:
                text = last_response

    # 1. <solution>...</solution> tags (may contain ```python inside)
    solution_matches = re.findall(
        r"<solution>\s*(.*?)\s*</solution>", text, re.DOTALL
    )
    if solution_matches:
        inner = solution_matches[-1].strip()
        # Strip inner ```python fences if present
        inner_code = re.findall(r"```python\s*(.*?)```", inner, re.DOTALL)
        if inner_code:
            return inner_code[-1].strip()
        inner_code = re.findall(r"```\s*(.*?)```", inner, re.DOTALL)
        if inner_code:
            return inner_code[-1].strip()
        return inner

    # 2. execute_code JSON — extract "code" field from tool_call / tool_response blocks
    # Handles multiple format variants:
    #   <tool_call>{"name": "execute_code", "parameters": {"code": "..."}}</tool_call>
    #   <tool_response>{"name":"execute_code","parameters":{"code":"..."}}</tool_response>
    # Uses JSON-aware pattern ((?:[^"\\]|\\.)*) to handle escaped chars inside strings.
    tc_code_matches = re.findall(
        r'"name"\s*:\s*"execute_code".*?"code"\s*:\s*"((?:[^"\\]|\\.)*)"',
        text, re.DOTALL
    )
    if tc_code_matches:
        # Filter out template placeholders and pick the longest valid code block
        # (placeholders like "your_python_code_here" are short and contain no real code)
        _PLACEHOLDER_PATTERNS = ('your_python_code_here', 'your_test_code_here', 'your_code_here')
        valid_matches = []
        for m in tc_code_matches:
            raw = m.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
            stripped = raw.strip()
            if not stripped:
                continue
            if any(p in stripped for p in _PLACEHOLDER_PATTERNS):
                continue
            # Require at least one Python keyword to avoid log-contaminated blocks
            if not any(kw in stripped for kw in ('def ', 'import ', 'print(', 'for ', 'while ', 'class ', 'if ')):
                continue
            valid_matches.append(stripped)
        if valid_matches:
            # Take the longest valid match (most likely the final complete solution)
            return max(valid_matches, key=len)

    # 2b. <code>...</code> tags
    code_tag_matches = re.findall(
        r"<code>\s*(.*?)\s*</code>", text, re.DOTALL
    )
    if code_tag_matches:
        inner = code_tag_matches[-1].strip()
        inner_code = re.findall(r"```python\s*(.*?)```", inner, re.DOTALL)
        if inner_code:
            return inner_code[-1].strip()
        inner_code = re.findall(r"```\s*(.*?)```", inner, re.DOTALL)
        if inner_code:
            return inner_code[-1].strip()
        return inner

    # 3. ```python ... ``` blocks
    python_matches = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if python_matches:
        return python_matches[-1].strip()

    # 4. Generic ``` ... ``` blocks
    generic_matches = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if generic_matches:
        return generic_matches[-1].strip()

    # 5. If the whole text looks like code, use it directly
    if any(kw in text for kw in ("def ", "import ", "print(", "for ", "while ")):
        return text.strip()

    return ""


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