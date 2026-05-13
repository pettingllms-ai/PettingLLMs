from typing import List, Dict, Any, Optional
import json
import logging
import re
import os
import subprocess
import asyncio
import warnings
import pickle
import numpy as np
import torch
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.model import compute_position_id_with_mask
from tensordict import TensorDict
# Suppress AutoGen/AG2 logging warnings and aiohttp resource warnings
logging.getLogger("autogen.oai.client").setLevel(logging.ERROR)
logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)
# Suppress ResourceWarning for unclosed aiohttp sessions
warnings.filterwarnings("ignore", category=ResourceWarning, message=".*unclosed.*")

logger = logging.getLogger(__name__)
from pettingllms.multi_agent_env.autoevol.reward_function import REWARD_FUNCTIONS


def _detect_gibberish(text: str, ngram_size: int = 3, threshold: float = 0.4) -> float:
    """Detect repetitive gibberish in text by measuring n-gram repetition ratio.

    Returns a penalty in [-0.5, 0.0].  The more repetitive the text, the
    larger the (negative) penalty.

    Args:
        text: The text to check.
        ngram_size: Size of character n-grams to use.
        threshold: Repetition ratio above which we start penalising.

    Returns:
        Float penalty: 0.0 (clean) to -0.5 (extremely repetitive).
    """
    if not text or len(text) < 100:
        return 0.0

    # Use last 2000 chars (where gibberish usually appears)
    sample = text[-2000:]

    # Character-level n-gram repetition
    ngrams = [sample[i:i + ngram_size] for i in range(len(sample) - ngram_size + 1)]
    if not ngrams:
        return 0.0
    unique_ratio = len(set(ngrams)) / len(ngrams)
    repetition_ratio = 1.0 - unique_ratio  # 0 = all unique, 1 = all same

    if repetition_ratio <= threshold:
        return 0.0

    # Scale: threshold→0, 1.0→-0.5
    penalty = -0.5 * (repetition_ratio - threshold) / (1.0 - threshold)
    return max(penalty, -0.5)
from pettingllms.multi_agent_env.autoevol.data_utils import load_and_tokenize_jsonl


def _extract_unwrapped_workflow_code(response: str) -> str:
    """Recover MAS code when the model emits raw workflow Python without fences."""
    start_positions = [
        response.find(marker)
        for marker in (
            "from workflow import",
            "from pettingllms.multi_agent_env.autoevol.workflow import",
        )
        if response.find(marker) >= 0
    ]
    if not start_positions:
        return ""

    start = min(start_positions)
    candidate = response[start:].strip()
    if "workflow = Workflow" not in candidate or "workflow.run" not in candidate:
        return ""

    run_index = candidate.find("workflow.run")
    stop_markers = (
        "\nWait,",
        "\nBut ",
        "\nHowever,",
        "\nAlternatively,",
        "\nSo ",
        "\nThe workflow",
        "\nThis workflow",
    )
    stop_positions = [
        candidate.find(marker, run_index)
        for marker in stop_markers
        if candidate.find(marker, run_index) > run_index
    ]
    if stop_positions:
        candidate = candidate[: min(stop_positions)].rstrip()

    lines = candidate.splitlines()
    while lines and lines[-1].strip() in {"```", "</code>"}:
        lines.pop()
    return "\n".join(lines).strip()


class MASGenerator(Agent):
    """MAS Designer Agent - designs multi-agent systems"""

    def __init__(self, task_type: str = "math", rollout_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.task_type = task_type.lower()
        self.rollout_idx = rollout_idx

        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)


    def update_from_env(self, env_data: Env):
        """Update agent from environment data and generate prompt"""
        self.env_data = env_data
        user_prompt_text = "Design Multi Agent System for the Question: " + env_data.state.problem
        system_prompt_text = "You are an expert Python developer specializing in multi-agent workflow systems."

        self.current_prompt = {"text": user_prompt_text, "image": None, "system": system_prompt_text}



    def update_from_model(self, response: str):
        code = ""
        self.current_response = response

        # Strategy 1: Try <code>...</code> tags first
        code_match = re.search(r"<code>\s*(.*?)\s*</code>", response, re.DOTALL)
        if code_match:
            code_content = code_match.group(1).strip()
            # Check if the content inside <code> tags contains ```python...``` blocks
            python_match = re.search(r"```python\s*(.*?)\s*```", code_content, re.DOTALL)
            if python_match:
                code = python_match.group(1).strip()
            else:
                # Try just ``` blocks without language specifier
                generic_match = re.search(r"```\s*(.*?)\s*```", code_content, re.DOTALL)
                if generic_match:
                    code = generic_match.group(1).strip()
                else:
                    # No code blocks inside, use the content as-is
                    code = code_content

        # Strategy 2: Fallback to ```python...``` blocks in the full response
        if not code:
            python_matches = re.findall(r"```python\s*(.*?)\s*```", response, re.DOTALL)
            if python_matches:
                code = python_matches[-1].strip()

        # Strategy 3: Fallback to generic ```...``` blocks
        if not code:
            generic_matches = re.findall(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if generic_matches:
                code = generic_matches[-1].strip()

        # Strategy 4: Fallback to raw workflow Python emitted without tags/fences.
        if not code:
            code = _extract_unwrapped_workflow_code(response)

        self.generated_code = code
        self.current_action = code

        return self.current_action
    async def step(self, env_data: Env, output_dir: str = None,
                   server_address: str = None, model_name: str = None, 
                   tokenizer_path: Optional[str] = None,
                   max_prompt_length: int = 2048, max_response_length: int = 2048,
                   ppo_trainer_config: Any = None, enable_thinking: bool = False,
                   step_timeout: float = 600.0, env_worker: Any = None):
        """
        Generate mas.py, execute it, collect results, and compute reward.

        Args:
            env_data: Environment data
            output_dir: Output directory for generated code
            server_address: vLLM server address for verl integration
            model_name: Model name for verl integration
            tokenizer_path: Path to tokenizer (preferred over extracting from tokenizer object)
            max_prompt_length: Max prompt length
            max_response_length: Max response length
            ppo_trainer_config: PPO trainer config for verl integration
            enable_thinking: Whether to enable thinking mode
            step_timeout: Timeout for executing mas.py
            env_worker: Ray worker for code execution (optional)

        Returns:
            dict: {
                "workflow_dpr": List of DataProto from workflow execution,
                "execution_success": bool indicating if execution succeeded,
                "reward": float reward score,
                "trajectory": List of DataProto (same as workflow_dpr)
            }
        """

        # Check if current_action is empty
        if not hasattr(self, 'current_action') or not self.current_action or not self.current_action.strip():
            logger.warning("current_action is empty, skipping execution")
            return {
                "workflow_dpr": [],
                "execution_success": False,
                "reward": 0.0,
                "designer_reward": 0.0,
                "correctness_reward": 0.0,
                "delivery_reward": 0.0,
                "solution_reward": 0.0,
                "trajectory": []
            }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        response_txt_path = os.path.join(output_dir, "response.txt")
        with open(response_txt_path, 'w') as f:
            f.write(self.current_response if hasattr(self, 'current_response') else '')
        # Save generated code to mas.py
        mas_py_path = os.path.join(output_dir, "mas.py")
        dataproto_pkl_path = os.path.abspath(os.path.join(output_dir, "dataproto.pkl"))
        output_txt_path = os.path.join(output_dir, "output.txt")
        # Log the final tokenizer_path for debugging
        logger.info(f"Using tokenizer_path for mas.py generation: {tokenizer_path}")

        # Add necessary imports and AIClient setup before the generated code
        setup_code = f"""
import sys
import os

# Resolve paths at runtime to avoid hardcoded absolute paths
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))

sys.path.insert(0, here)
sys.path.insert(0, repo_root)

# Suppress warnings
import logging
import warnings
logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Set task type for agent delivery instruction
os.environ["TASK_TYPE"] = "{self.task_type}"

# Import necessary modules for verl integration
from pettingllms.multi_agent_env.autoevol.utils.BaseOpenAI import AIClient

# Debug: Print tokenizer_path for verification
_tokenizer_path = {repr(tokenizer_path)}
print(f"[DEBUG] mas.py: tokenizer_path = {{_tokenizer_path}}")
print(f"[DEBUG] mas.py: tokenizer_path type = {{type(_tokenizer_path)}}")

# Create AIClient with verl integration
_server_addr = "{server_address}"
_api_base = _server_addr if _server_addr.startswith('http') else ("http://" + _server_addr)
print(f"[AICLIENT SETUP] Creating AIClient with:")
print(f"[AICLIENT SETUP]   server_address = {{_server_addr}}")
print(f"[AICLIENT SETUP]   model_name = {{'{model_name or 'default'}'}}")
print(f"[AICLIENT SETUP]   tokenizer_path = {{_tokenizer_path}}")
print(f"[AICLIENT SETUP]   max_prompt_length = {max_prompt_length}")
print(f"[AICLIENT SETUP]   max_response_length = {max_response_length}")
print(f"[AICLIENT SETUP]   enable_thinking = {enable_thinking}")
ai_client = AIClient(
    api_base=_api_base,
    api_key="dummy",
    chat_model="{model_name or 'default'}",
    max_answer_tokens={max_response_length},
    tokenizer_path=_tokenizer_path,
    server_address=_server_addr,
    max_prompt_length={max_prompt_length},
    max_response_length={max_response_length},
    enable_thinking={str(enable_thinking)},
    workflow=None  # Will be set after workflow creation
)
print(f"[AICLIENT SETUP] AIClient created successfully")

"""

        # Patch the generated code to fix imports and add verl parameters
        patched_code = self._patch_imports(self.generated_code)
        patched_code = self._patch_string_escapes(patched_code)
        patched_code = self._patch_workflow_init(
            patched_code,
            server_address,
            enable_thinking
        )

        # Add DataProto saving code at the end
        dataproto_save_code = f"""


# Save workflow DataProto for training
try:
    import pickle

    if 'workflow' in globals() and hasattr(workflow, 'dataproto_list') and workflow.dataproto_list:
        dataproto_file = r'{dataproto_pkl_path}'
        os.makedirs(os.path.dirname(dataproto_file), exist_ok=True)

        with open(dataproto_file, 'wb') as f:
            pickle.dump(workflow.dataproto_list, f)
        print("[SUCCESSSAVED]")

except Exception as e:
    print(f"\\n[ERROR] Failed to save DataProto: {{e}}")
    import traceback
    traceback.print_exc()
"""

        # Combine all parts
        full_code = setup_code + "\n" + patched_code + "\n" + dataproto_save_code

        # Save to file
        with open(mas_py_path, 'w') as f:
            f.write(full_code)

        logger.info(f"Generated mas.py with verl integration: {mas_py_path}")

        # Execute mas.py and collect results
        execution_success = False
        workflow_dataproto_list = []
        final_answer = ""
        reward = 0.0
        designer_reward = 0.0
        correctness_reward = 0.0
        delivery_reward = 0.0
        solution_reward = 0.0

        try:
            # Use Ray worker for execution (better resource management)
            with open(mas_py_path, 'r') as f:
                script_content = f.read()
            
            from pettingllms.multi_agent_env.math.math_worker import _await_ray_object_ref
            import ray

            timeout_buffer = 60
            total_timeout = step_timeout + timeout_buffer

            obj_ref = env_worker.run.remote(script_content, step_timeout)
            try:
                output_text = await _await_ray_object_ref(obj_ref, total_timeout)
            finally:
                del obj_ref  # Release Ray object reference to free object store memory

            # Print executor output summary (set EXECUTOR_VERBOSE_DEBUG=1 for full output)
            import os as _os
            if _os.environ.get("EXECUTOR_VERBOSE_DEBUG", "0") == "1":
                print("=" * 80)
                print(f"[EXECUTOR OUTPUT] Rollout execution output (length: {len(output_text)}):")
                print("=" * 80)
                print(output_text)
                print("=" * 80)
                print(f"[EXECUTOR OUTPUT END]")
            else:
                print(f"[EXECUTOR OUTPUT] length={len(output_text)}, success={'[SUCCESSSAVED]' in output_text}")
            
            # Save output to file
            with open(output_txt_path, 'w') as f:
                f.write(output_text)
            
            if "[SUCCESSSAVED]" in output_text:
                execution_success = True
                print(f"[EXECUTOR SUCCESS] MAS execution succeeded, DataProto saved")
            else:
                print(f"[EXECUTOR WARNING] MAS execution may have failed - [SUCCESSSAVED] not found in output")

            # Load DataProto from file if exists
            if os.path.exists(dataproto_pkl_path):
                with open(dataproto_pkl_path, 'rb') as f:
                    workflow_dataproto_list = pickle.load(f)
                logger.info(f"Loaded {len(workflow_dataproto_list)} DataProto entries from {dataproto_pkl_path}")

            # Extract final answer from output
            final_answer = self._extract_final_answer(output_text)

            # For code tasks, pass full output_text so code_reward_function can
            # extract code blocks; for math/other tasks use extracted answer.
            is_code_task = self.task_type.lower() == "code"
            if is_code_task:
                from pettingllms.multi_agent_env.autoevol.reward_function import _extract_code_block as _ecb
                extracted_code = _ecb(output_text)
                print(f"[EXECUTOR RESULT] Extracted code (first 100 chars): {extracted_code[:100]!r}")
                correctness_reward = self._calculate_reward(output_text, env_data)
                print(f"[EXECUTOR RESULT] Code task - correctness_reward: {correctness_reward}")
            else:
                print(f"[EXECUTOR RESULT] Ground truth answer: {env_data.state.ground_truth_answer}")
                correctness_reward = self._calculate_reward(final_answer, env_data)

            # --- Format rewards (split: delivery + solution) ---
            import re as _re
            import ast as _ast
            delivery_reward = 0.0
            solution_reward = 0.0

            # Delivery reward: +0.4 if agent used <delivery>...</delivery> tags
            _disable_delivery = _os.environ.get("DISABLE_DELIVERY_REWARD", "0") == "1"
            has_delivery_tag = bool(_re.search(r'<delivery>.*?</delivery>', output_text, _re.DOTALL))
            if has_delivery_tag and not _disable_delivery:
                delivery_reward = 0.4

            if is_code_task:
                # Solution reward: +1.0 if <solution> present AND parseable Python
                has_solution_tag = bool(_re.search(r'<solution>.*?</solution>', output_text, _re.DOTALL))
                if has_solution_tag:
                    _code = (extracted_code or '').strip()
                    try:
                        _ast.parse(_code)
                        solution_reward = 0.4 if len(_code) >= 10 else 0.0
                    except SyntaxError:
                        solution_reward = 0.0
            else:
                # Solution reward: +1.0 if model used \boxed{} (math only)
                has_boxed = '\\boxed{' in output_text
                _is_placeholder = bool(_re.match(
                    r'^[\{\}]*[a-zA-Z_][a-zA-Z_0-9]*[\{\}]*$', final_answer.strip()
                )) if final_answer.strip() else True
                solution_reward = 0.4 if (has_boxed and not _is_placeholder) else 0.0

            format_reward = delivery_reward + solution_reward

            # --- Length penalty: disabled ---
            length_penalty = 0.0

            # Designer reward: only correctness + format (no penalties)
            designer_reward = correctness_reward + format_reward
            # Agent node reward: correctness + format
            reward = correctness_reward + format_reward + length_penalty
            print(f"[EXECUTOR RESULT] designer_reward: {designer_reward} (correctness={correctness_reward}, delivery={delivery_reward}, solution={solution_reward})")
            print(f"[EXECUTOR RESULT] agent_reward: {reward} (correctness={correctness_reward}, delivery={delivery_reward}, solution={solution_reward})")
            logger.info(f"Calculated reward: designer={designer_reward}, agent={reward} for final_answer: {final_answer}")

        except asyncio.TimeoutError:
            logger.warning(f"MAS execution timed out after {step_timeout}s")
        except Exception as e:
            logger.warning(f"MAS execution error: {e}")
            import traceback
            logger.warning(traceback.format_exc())

        # Return results
        return {
            "workflow_dpr": workflow_dataproto_list,
            "execution_success": execution_success,
            "reward": reward,
            "designer_reward": designer_reward,
            "correctness_reward": correctness_reward,
            "delivery_reward": delivery_reward,
            "solution_reward": solution_reward,
            "trajectory": workflow_dataproto_list
        }

    def _patch_imports(self, code: str) -> str:
        """
        Patch import statements to use full module paths.
        
        Converts:
        - from workflow import ... -> from pettingllms.multi_agent_env.autoevol.workflow import ...
        - from utils import ... -> from pettingllms.multi_agent_env.autoevol.utils import ...
        
        Args:
            code: Generated code with relative imports
            
        Returns:
            Patched code with absolute imports
        """
        import re
        
        # Pattern 1: from workflow import ...
        # Pattern 2: from workflow.xxx import ...
        # Pattern 3: from utils import ...
        # Pattern 4: from utils.xxx import ...
        
        # Replace workflow imports
        code = re.sub(
            r'\bfrom\s+workflow\s+import\s+',
            'from pettingllms.multi_agent_env.autoevol.workflow import ',
            code
        )
        code = re.sub(
            r'\bfrom\s+workflow\.',
            'from pettingllms.multi_agent_env.autoevol.workflow.',
            code
        )
        
        # Replace utils imports
        code = re.sub(
            r'\bfrom\s+utils\s+import\s+',
            'from pettingllms.multi_agent_env.autoevol.utils import ',
            code
        )
        code = re.sub(
            r'\bfrom\s+utils\.',
            'from pettingllms.multi_agent_env.autoevol.utils.',
            code
        )
        
        return code

    def _patch_string_escapes(self, code: str) -> str:
        """
        Patch string literals to handle backslashes correctly.
        
        Converts strings containing backslashes (like LaTeX commands) to raw strings
        to avoid unicode escape errors.
        
        Args:
            code: Generated code that may contain strings with backslashes
            
        Returns:
            Patched code with raw strings where needed
        """
        import re
        
        def needs_raw_string(content):
            if '\\' not in content:
                return False
            
            problematic_patterns = [
                r'\\u[0-9a-fA-F]', r'\\U[0-9a-fA-F]', r'\\N\{', r'\\x[0-9a-fA-F]',
                r'\\textit', r'\\textbf', r'\\underline', r'\\overline',
                r'\\sqrt', r'\\frac', r'\\geq', r'\\leq', r'\\neq',
                r'\\sum', r'\\prod', r'\\int', r'\\lim',
                r'\\left', r'\\right', r'\\begin', r'\\end'
            ]
            
            for pattern in problematic_patterns:
                if re.search(pattern, content):
                    return True
            
            return False
        
        def replace_string(match):
            full_match = match.group(0)
            var_name = match.group(1)
            equals_space = match.group(2)
            raw_prefix = match.group(3) or ''
            quote_type = match.group(4)
            content = match.group(5)
            
            if raw_prefix.lower() == 'r':
                return full_match
            
            if needs_raw_string(content):
                result = var_name + equals_space + 'r' + quote_type + content + quote_type
                return result
            
            return full_match
        
        pattern = r'(\w+)(\s*=\s*)(r|R)?("""|\'\'\')(.+?)\4'
        patched = re.sub(pattern, replace_string, code, flags=re.DOTALL)
        
        return patched

    def _patch_workflow_init(self, code: str, server_address: str, enable_thinking: bool) -> str:
        """
        Patch workflow = Workflow(...) to add verl integration parameters.

        Args:
            code: Generated code containing Workflow initialization
            server_address: vLLM server address
            enable_thinking: Whether to enable thinking mode

        Returns:
            Patched code with verl parameters added to Workflow
        """
        

        # Find workflow = Workflow(...) pattern and the line after it
        # Match: workflow = Workflow(name="...", other_params)
        pattern = r'(workflow\s*=\s*Workflow\s*\([^)]*)\)'

        def add_verl_params(match):
            original = match.group(1)
            # Add verl parameters before the closing parenthesis
            verl_params = f"""
    ai_client=ai_client"""

            # Check if original ends with comma
            if original.rstrip().endswith(','):
                workflow_line = original + verl_params + "\n)"
            else:
                workflow_line = original + "," + verl_params + "\n)"
            
            # Add ai_client.workflow assignment immediately after workflow creation
            workflow_line += "\nai_client.workflow = workflow"
            
            return workflow_line

        # Apply the patch
        patched = re.sub(pattern, add_verl_params, code, flags=re.MULTILINE | re.DOTALL)

        if patched == code:
            logger.warning("Could not find 'workflow = Workflow(...)' pattern to patch. Code unchanged.")

        return patched
    def _extract_final_answer(self, output_text: str) -> str:
        """
        Extract final answer from MAS execution output.

        Priority:
        1. \boxed{...} format (preferred)
        2. **Final Answer** markers
        3. WORKFLOW_SUMMARY markers
        4. Fallback to last content

        Args:
            output_text: Output from MAS execution

        Returns:
            Extracted final answer string
        """
        import re

        # Strategy 0 (HIGHEST PRIORITY): Look for \boxed{...} format
        # Use brace-counting to handle nested braces like \dfrac{2}{9}
        def extract_boxed_content(text):
            """Extract all \boxed{...} contents using brace counting.

            Handles both \boxed{} and \\boxed{} (escaped backslash).
            """
            results = []
            # Try both single and double backslash patterns
            for pattern in ['\\boxed{', '\\\\boxed{']:
                i = 0
                while i < len(text):
                    boxed_start = text.find(pattern, i)
                    if boxed_start == -1:
                        break
                    # Find matching closing brace
                    content_start = boxed_start + len(pattern)
                    brace_count = 1
                    j = content_start
                    while j < len(text) and brace_count > 0:
                        if text[j] == '{':
                            brace_count += 1
                        elif text[j] == '}':
                            brace_count -= 1
                        j += 1
                    if brace_count == 0:
                        content = text[content_start:j-1]
                        # Skip if we already found this content (avoid duplicates from overlapping patterns)
                        if content not in [r for r in results]:
                            results.append(content)
                    i = j if brace_count == 0 else boxed_start + 1
            return results

        boxed_matches = extract_boxed_content(output_text)
        if boxed_matches:
            # Return the LAST boxed answer (final answer)
            return boxed_matches[-1].strip()

        # Strategy 1: Look for **Final Answer** or FINAL ANSWER: format
        # Try pattern with stars first: **Final Answer** or **Final Answer**: xxx
        final_answer_match = re.search(
            r'\*\*Final\s*Answer\s*\*{0,2}[:\s]*(.*?)(?:={3,}|\Z)',
            output_text,
            re.DOTALL | re.IGNORECASE
        )

        # If not found, try pattern without stars: FINAL ANSWER: xxx or FINAL ANSWER:\nxxx
        if not final_answer_match:
            final_answer_match = re.search(
                r'(?:^|\n)\s*FINAL\s+ANSWER\s*:?\s*\n?\s*(.*?)(?:\n\n|\n===|\n\s*\n|$|\Z)',
                output_text,
                re.MULTILINE | re.IGNORECASE | re.DOTALL
            )

        # Also try case-insensitive pattern: Final Answer: xxx or Final Answer:\nxxx
        if not final_answer_match:
            final_answer_match = re.search(
                r'(?:^|\n)\s*Final\s+Answer\s*:?\s*\n?\s*(.*?)(?:\n\n|\n===|\n\s*\n|$|\Z)',
                output_text,
                re.MULTILINE | re.IGNORECASE | re.DOTALL
            )

        if final_answer_match:
            final_answer_section = final_answer_match.group(1).strip()

            # First check for boxed in this section using brace counting
            boxed_in_section = extract_boxed_content(final_answer_section)
            if boxed_in_section:
                return boxed_in_section[-1].strip()

            # Return cleaned text if no boxed found (for text answers)
            # Remove markdown formatting
            cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', final_answer_section)
            cleaned = re.sub(r'\\\[|\\\]|\\\(|\\\)', '', cleaned)  # Remove LaTeX delimiters
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) < 200:  # Sanity check
                # Try to extract a clean answer from the section
                # Look for patterns like "= X" or ": X" at the end
                eq_match = re.search(r'[=:]\s*([^\n=:]+?)(?:\s*[.。]?\s*)?$', cleaned)
                if eq_match:
                    return eq_match.group(1).strip()
                return cleaned

        # Strategy 2: Look for WORKFLOW_SUMMARY_START/END markers
        workflow_match = re.search(
            r'WORKFLOW_SUMMARY_START\s*(.*?)\s*WORKFLOW_SUMMARY_END',
            output_text,
            re.DOTALL | re.IGNORECASE
        )

        if workflow_match:
            summary_content = workflow_match.group(1).strip()

            # Check for boxed in summary using brace counting
            boxed_in_summary = extract_boxed_content(summary_content)
            if boxed_in_summary:
                return boxed_in_summary[-1].strip()

            # Fallback to last non-empty line
            lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
            if lines:
                return lines[-1]

        # Strategy 3: Last resort - look for boxed anywhere in last part of output
        # Only search in the last 3000 characters
        search_text = output_text[-3000:] if len(output_text) > 3000 else output_text
        boxed_in_end = extract_boxed_content(search_text)
        if boxed_in_end:
            return boxed_in_end[-1].strip()

        # Fallback: return last non-empty line (avoid returning numbers that could be token counts)
        lines = [line.strip() for line in output_text.split('\n') if line.strip()]
        # Filter out lines that look like debug output or token counts
        filtered_lines = [l for l in lines if not re.match(r'^[\d,]+$', l) and 'token' not in l.lower() and 'max_' not in l.lower()]
        return filtered_lines[-1] if filtered_lines else (lines[-1] if lines else "")
    
    def _calculate_reward(self, final_answer: str, env_data: Env) -> float:
        """
        Calculate reward based on task type and answer correctness.
        
        Args:
            final_answer: Extracted final answer from MAS execution
            env_data: Environment data containing ground truth
            
        Returns:
            Reward score (typically 1.0 for correct, 0.0 for incorrect)
        """
        reward_function = REWARD_FUNCTIONS.get(self.task_type.lower())
        
        if reward_function is None:
            logger.warning(f"No reward function found for task_type: {self.task_type}, defaulting to 0.0")
            return 0.0
        
        try:
            # Pass final_answer as summary to reward function
            reward = reward_function(final_answer, env_data)
            return float(reward)
        except Exception as e:
            logger.warning(f"Error calculating reward: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return 0.0


class MASExecutor(Agent):
    """MAS Executor Agent - executes multi-agent systems designed by Designer"""

    def __init__(self, task_type: str = "math", rollout_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.task_type = task_type.lower()
        self.rollout_idx = rollout_idx

        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, env_data: Env, designer_code: str = None):
        """Update agent from environment data and designer's generated code"""
        self.env_data = env_data
        # Use designer's code as the prompt, but without thinking
        # The prompt should be the code itself, ready to execute
        if designer_code:
            user_prompt_text = designer_code
            system_prompt_text = "You are an expert in executing Multi-Agent System workflows. Execute the provided workflow code."
        else:
            # Fallback if no designer code is provided
            user_prompt_text = "Execute the Multi Agent System workflow."
            system_prompt_text = "You are an expert in executing Multi-Agent System workflows."

        self.current_prompt = {"text": user_prompt_text, "image": None, "system": system_prompt_text}
        self.designer_code = designer_code

    def update_from_model(self, response: str):
        """For executor, we don't need to extract code from response - we already have the code from designer"""
        self.current_response = response
        # Executor doesn't generate new code, it executes the designer's code
        # So we keep the designer's code as current_action
        if hasattr(self, 'designer_code') and self.designer_code:
            self.current_action = self.designer_code
        else:
            self.current_action = ""
        return self.current_action

    async def step(self, env_data: Env, output_dir: str = None,
                   server_address: str = None, model_name: str = None, 
                   tokenizer_path: Optional[str] = None,
                   max_prompt_length: int = 2048, max_response_length: int = 2048,
                   ppo_trainer_config: Any = None, enable_thinking: bool = False,
                   step_timeout: float = 600.0, env_worker: Any = None):
        """
        Execute the MAS code designed by Designer, collect results, and compute reward.

        Args:
            env_data: Environment data
            output_dir: Output directory for generated code
            server_address: vLLM server address for verl integration
            model_name: Model name for verl integration
            tokenizer_path: Path to tokenizer (preferred over extracting from tokenizer object)
            max_prompt_length: Max prompt length
            max_response_length: Max response length
            ppo_trainer_config: PPO trainer config for verl integration
            enable_thinking: Whether to enable thinking mode (should be False for executor)
            step_timeout: Timeout for executing mas.py
            env_worker: Ray worker for code execution (optional)

        Returns:
            dict: {
                "workflow_dpr": List of DataProto from workflow execution,
                "execution_success": bool indicating if execution succeeded,
                "reward": float reward score,
                "trajectory": List of DataProto (same as workflow_dpr)
            }
        """
        # Use designer's code for execution
        code = self.designer_code if hasattr(self, 'designer_code') and self.designer_code else ""
        
        if not code or not code.strip():
            logger.warning("No designer code available for execution")
            return {
                "workflow_dpr": [],
                "execution_success": False,
                "reward": 0.0,
                "designer_reward": 0.0,
                "correctness_reward": 0.0,
                "delivery_reward": 0.0,
                "solution_reward": 0.0,
                "trajectory": []
            }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        response_txt_path = os.path.join(output_dir, "executor_response.txt")
        with open(response_txt_path, 'w') as f:
            f.write(self.current_response if hasattr(self, 'current_response') else '')
        
        # Save generated code to mas.py
        mas_py_path = os.path.join(output_dir, "mas.py")
        dataproto_pkl_path = os.path.abspath(os.path.join(output_dir, "dataproto.pkl"))
        output_txt_path = os.path.join(output_dir, "output.txt")
        
        logger.info(f"Using tokenizer_path for mas.py generation: {tokenizer_path}")

        # Add necessary imports and AIClient setup before the generated code
        setup_code = f"""
import sys
import os

# Resolve paths at runtime to avoid hardcoded absolute paths
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))

sys.path.insert(0, here)
sys.path.insert(0, repo_root)

# Suppress warnings
import logging
import warnings
logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Set task type for agent delivery instruction
os.environ["TASK_TYPE"] = "{self.task_type}"

# Import necessary modules for verl integration
from pettingllms.multi_agent_env.autoevol.utils.BaseOpenAI import AIClient

# Debug: Print tokenizer_path for verification
_tokenizer_path = {repr(tokenizer_path)}
print(f"[DEBUG] mas.py: tokenizer_path = {{_tokenizer_path}}")
print(f"[DEBUG] mas.py: tokenizer_path type = {{type(_tokenizer_path)}}")

# Create AIClient with verl integration
_server_addr = "{server_address}"
_api_base = _server_addr if _server_addr.startswith('http') else ("http://" + _server_addr)
ai_client = AIClient(
    api_base=_api_base,
    api_key="dummy",
    chat_model="{model_name or 'default'}",
    max_answer_tokens={max_response_length},
    tokenizer_path=_tokenizer_path,
    server_address=_server_addr,
    max_prompt_length={max_prompt_length},
    max_response_length={max_response_length},
    enable_thinking={str(False)},  # Executor should not use thinking
    workflow=None  # Will be set after workflow creation
)

"""

        # Patch the generated code to fix imports and add verl parameters
        patched_code = self._patch_imports(code)
        patched_code = self._patch_string_escapes(patched_code)
        patched_code = self._patch_workflow_init(
            patched_code,
            server_address,
            False  # Executor should not use thinking
        )

        # Add DataProto saving code at the end
        dataproto_save_code = f"""


# Save workflow DataProto for training
try:
    import pickle

    if 'workflow' in globals() and hasattr(workflow, 'dataproto_list') and workflow.dataproto_list:
        dataproto_file = r'{dataproto_pkl_path}'
        os.makedirs(os.path.dirname(dataproto_file), exist_ok=True)

        with open(dataproto_file, 'wb') as f:
            pickle.dump(workflow.dataproto_list, f)
        print("[SUCCESSSAVED]")

except Exception as e:
    print(f"\\n[ERROR] Failed to save DataProto: {{e}}")
    import traceback
    traceback.print_exc()
"""

        # Combine all parts
        full_code = setup_code + "\n" + patched_code + "\n" + dataproto_save_code

        # Save to file
        with open(mas_py_path, 'w') as f:
            f.write(full_code)

        logger.info(f"Generated mas.py with verl integration: {mas_py_path}")

        # Execute mas.py and collect results
        execution_success = False
        workflow_dataproto_list = []
        final_answer = ""
        reward = 0.0
        designer_reward = 0.0
        correctness_reward = 0.0
        delivery_reward = 0.0
        solution_reward = 0.0

        try:
            # Use Ray worker for execution (better resource management)
            with open(mas_py_path, 'r') as f:
                script_content = f.read()
            
            from pettingllms.multi_agent_env.math.math_worker import _await_ray_object_ref
            import ray

            timeout_buffer = 60
            total_timeout = step_timeout + timeout_buffer

            obj_ref = env_worker.run.remote(script_content, step_timeout)
            try:
                output_text = await _await_ray_object_ref(obj_ref, total_timeout)
            finally:
                del obj_ref  # Release Ray object reference to free object store memory

            # Print executor output summary (set EXECUTOR_VERBOSE_DEBUG=1 for full output)
            import os as _os
            if _os.environ.get("EXECUTOR_VERBOSE_DEBUG", "0") == "1":
                print("=" * 80)
                print(f"[EXECUTOR OUTPUT] Rollout execution output (length: {len(output_text)}):")
                print("=" * 80)
                print(output_text)
                print("=" * 80)
                print(f"[EXECUTOR OUTPUT END]")
            else:
                print(f"[EXECUTOR OUTPUT] length={len(output_text)}, success={'[SUCCESSSAVED]' in output_text}")
            
            # Save output to file
            with open(output_txt_path, 'w') as f:
                f.write(output_text)
            
            if "[SUCCESSSAVED]" in output_text:
                execution_success = True
                print(f"[EXECUTOR SUCCESS] MAS execution succeeded, DataProto saved")
            else:
                print(f"[EXECUTOR WARNING] MAS execution may have failed - [SUCCESSSAVED] not found in output")

            # Load DataProto from file if exists
            if os.path.exists(dataproto_pkl_path):
                with open(dataproto_pkl_path, 'rb') as f:
                    workflow_dataproto_list = pickle.load(f)
                logger.info(f"Loaded {len(workflow_dataproto_list)} DataProto entries from {dataproto_pkl_path}")

            # Extract final answer from output
            final_answer = self._extract_final_answer(output_text)

            # For code tasks, pass full output_text so code_reward_function can
            # extract code blocks; for math/other tasks use extracted answer.
            is_code_task = self.task_type.lower() == "code"
            if is_code_task:
                from pettingllms.multi_agent_env.autoevol.reward_function import _extract_code_block as _ecb
                extracted_code = _ecb(output_text)
                print(f"[EXECUTOR RESULT] Extracted code (first 100 chars): {extracted_code[:100]!r}")
                correctness_reward = self._calculate_reward(output_text, env_data)
                print(f"[EXECUTOR RESULT] Code task - correctness_reward: {correctness_reward}")
            else:
                print(f"[EXECUTOR RESULT] Ground truth answer: {env_data.state.ground_truth_answer}")
                correctness_reward = self._calculate_reward(final_answer, env_data)

            # --- Format rewards (split: delivery + solution) ---
            import re as _re
            import ast as _ast
            delivery_reward = 0.0
            solution_reward = 0.0

            # Delivery reward: +0.4 if agent used <delivery>...</delivery> tags
            _disable_delivery = _os.environ.get("DISABLE_DELIVERY_REWARD", "0") == "1"
            has_delivery_tag = bool(_re.search(r'<delivery>.*?</delivery>', output_text, _re.DOTALL))
            if has_delivery_tag and not _disable_delivery:
                delivery_reward = 0.4

            if is_code_task:
                # Solution reward: +1.0 if <solution> present AND parseable Python
                has_solution_tag = bool(_re.search(r'<solution>.*?</solution>', output_text, _re.DOTALL))
                if has_solution_tag:
                    _code = (extracted_code or '').strip()
                    try:
                        _ast.parse(_code)
                        solution_reward = 0.4 if len(_code) >= 10 else 0.0
                    except SyntaxError:
                        solution_reward = 0.0
            else:
                # Solution reward: +1.0 if model used \boxed{} (math only)
                has_boxed = '\\boxed{' in output_text
                _is_placeholder = bool(_re.match(
                    r'^[\{\}]*[a-zA-Z_][a-zA-Z_0-9]*[\{\}]*$', final_answer.strip()
                )) if final_answer.strip() else True
                solution_reward = 0.4 if (has_boxed and not _is_placeholder) else 0.0

            format_reward = delivery_reward + solution_reward

            # --- Length penalty: disabled ---
            length_penalty = 0.0

            # Designer reward: only correctness + format (no penalties)
            designer_reward = correctness_reward + format_reward
            # Agent node reward: correctness + format
            reward = correctness_reward + format_reward + length_penalty
            print(f"[EXECUTOR RESULT] designer_reward: {designer_reward} (correctness={correctness_reward}, delivery={delivery_reward}, solution={solution_reward})")
            print(f"[EXECUTOR RESULT] agent_reward: {reward} (correctness={correctness_reward}, delivery={delivery_reward}, solution={solution_reward})")
            logger.info(f"Calculated reward: designer={designer_reward}, agent={reward} for final_answer: {final_answer}")

        except asyncio.TimeoutError:
            logger.warning(f"MAS execution timed out after {step_timeout}s")
        except Exception as e:
            logger.warning(f"MAS execution error: {e}")
            import traceback
            logger.warning(traceback.format_exc())

        # Return results
        return {
            "workflow_dpr": workflow_dataproto_list,
            "execution_success": execution_success,
            "reward": reward,
            "designer_reward": designer_reward,
            "correctness_reward": correctness_reward,
            "delivery_reward": delivery_reward,
            "solution_reward": solution_reward,
            "trajectory": workflow_dataproto_list
        }

    def _patch_imports(self, code: str) -> str:
        """Patch import statements to use full module paths."""
        import re
        
        # Replace workflow imports
        code = re.sub(
            r'\bfrom\s+workflow\s+import\s+',
            'from pettingllms.multi_agent_env.autoevol.workflow import ',
            code
        )
        code = re.sub(
            r'\bfrom\s+workflow\.',
            'from pettingllms.multi_agent_env.autoevol.workflow.',
            code
        )
        
        # Replace utils imports
        code = re.sub(
            r'\bfrom\s+utils\s+import\s+',
            'from pettingllms.multi_agent_env.autoevol.utils import ',
            code
        )
        code = re.sub(
            r'\bfrom\s+utils\.',
            'from pettingllms.multi_agent_env.autoevol.utils.',
            code
        )
        
        return code

    def _patch_string_escapes(self, code: str) -> str:
        """Patch string literals to handle backslashes correctly."""
        import re
        
        def needs_raw_string(content):
            if '\\' not in content:
                return False
            
            problematic_patterns = [
                r'\\u[0-9a-fA-F]', r'\\U[0-9a-fA-F]', r'\\N\{', r'\\x[0-9a-fA-F]',
                r'\\textit', r'\\textbf', r'\\underline', r'\\overline',
                r'\\sqrt', r'\\frac', r'\\geq', r'\\leq', r'\\neq',
                r'\\sum', r'\\prod', r'\\int', r'\\lim',
                r'\\left', r'\\right', r'\\begin', r'\\end'
            ]
            
            for pattern in problematic_patterns:
                if re.search(pattern, content):
                    return True
            
            return False
        
        def replace_string(match):
            full_match = match.group(0)
            var_name = match.group(1)
            equals_space = match.group(2)
            raw_prefix = match.group(3) or ''
            quote_type = match.group(4)
            content = match.group(5)
            
            if raw_prefix.lower() == 'r':
                return full_match
            
            if needs_raw_string(content):
                result = var_name + equals_space + 'r' + quote_type + content + quote_type
                return result
            
            return full_match
        
        pattern = r'(\w+)(\s*=\s*)(r|R)?("""|\'\'\')(.+?)\4'
        patched = re.sub(pattern, replace_string, code, flags=re.DOTALL)
        
        return patched

    def _patch_workflow_init(self, code: str, server_address: str, enable_thinking: bool) -> str:
        """Patch workflow = Workflow(...) to add verl integration parameters."""
        import re

        pattern = r'(workflow\s*=\s*Workflow\s*\([^)]*)\)'

        def add_verl_params(match):
            original = match.group(1)
            verl_params = f"""
    ai_client=ai_client"""

            if original.rstrip().endswith(','):
                workflow_line = original + verl_params + "\n)"
            else:
                workflow_line = original + "," + verl_params + "\n)"
            
            workflow_line += "\nai_client.workflow = workflow"
            
            return workflow_line

        patched = re.sub(pattern, add_verl_params, code, flags=re.MULTILINE | re.DOTALL)

        if patched == code:
            logger.warning("Could not find 'workflow = Workflow(...)' pattern to patch. Code unchanged.")

        return patched

    def _extract_final_answer(self, output_text: str) -> str:
        """
        Extract final answer from MAS execution output.

        Priority:
        1. \boxed{...} format (preferred)
        2. **Final Answer** markers
        3. WORKFLOW_SUMMARY markers
        4. Fallback to last content
        """
        import re

        # Strategy 0 (HIGHEST PRIORITY): Look for \boxed{...} format
        # Use brace-counting to handle nested braces like \dfrac{2}{9}
        def extract_boxed_content(text):
            """Extract all \boxed{...} contents using brace counting.

            Handles both \boxed{} and \\boxed{} (escaped backslash).
            """
            results = []
            # Try both single and double backslash patterns
            for pattern in ['\\boxed{', '\\\\boxed{']:
                i = 0
                while i < len(text):
                    boxed_start = text.find(pattern, i)
                    if boxed_start == -1:
                        break
                    # Find matching closing brace
                    content_start = boxed_start + len(pattern)
                    brace_count = 1
                    j = content_start
                    while j < len(text) and brace_count > 0:
                        if text[j] == '{':
                            brace_count += 1
                        elif text[j] == '}':
                            brace_count -= 1
                        j += 1
                    if brace_count == 0:
                        content = text[content_start:j-1]
                        # Skip if we already found this content (avoid duplicates from overlapping patterns)
                        if content not in [r for r in results]:
                            results.append(content)
                    i = j if brace_count == 0 else boxed_start + 1
            return results

        boxed_matches = extract_boxed_content(output_text)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # Strategy 1: Look for **Final Answer** or FINAL ANSWER: format
        final_answer_match = re.search(
            r'\*\*Final\s*Answer\s*\*{0,2}[:\s]*(.*?)(?:={3,}|\Z)',
            output_text,
            re.DOTALL | re.IGNORECASE
        )

        if not final_answer_match:
            final_answer_match = re.search(
                r'(?:^|\n)\s*FINAL\s+ANSWER\s*:?\s*\n?\s*(.*?)(?:\n\n|\n===|\n\s*\n|$|\Z)',
                output_text,
                re.MULTILINE | re.IGNORECASE | re.DOTALL
            )

        if not final_answer_match:
            final_answer_match = re.search(
                r'(?:^|\n)\s*Final\s+Answer\s*:?\s*\n?\s*(.*?)(?:\n\n|\n===|\n\s*\n|$|\Z)',
                output_text,
                re.MULTILINE | re.IGNORECASE | re.DOTALL
            )

        if final_answer_match:
            final_answer_section = final_answer_match.group(1).strip()

            # Check for boxed in section using brace counting
            boxed_in_section = extract_boxed_content(final_answer_section)
            if boxed_in_section:
                return boxed_in_section[-1].strip()

            # Return cleaned text for text answers
            cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', final_answer_section)
            cleaned = re.sub(r'\\\[|\\\]|\\\(|\\\)', '', cleaned)  # Remove LaTeX delimiters
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) < 200:
                # Try to extract a clean answer from the section
                eq_match = re.search(r'[=:]\s*([^\n=:]+?)(?:\s*[.。]?\s*)?$', cleaned)
                if eq_match:
                    return eq_match.group(1).strip()
                return cleaned

        # Strategy 2: Look for WORKFLOW_SUMMARY_START/END markers
        workflow_match = re.search(
            r'WORKFLOW_SUMMARY_START\s*(.*?)\s*WORKFLOW_SUMMARY_END',
            output_text,
            re.DOTALL | re.IGNORECASE
        )

        if workflow_match:
            summary_content = workflow_match.group(1).strip()

            # Check for boxed in summary using brace counting
            boxed_in_summary = extract_boxed_content(summary_content)
            if boxed_in_summary:
                return boxed_in_summary[-1].strip()

            lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
            if lines:
                return lines[-1]

        # Strategy 3: Last resort - look for boxed anywhere in last part of output
        search_text = output_text[-3000:] if len(output_text) > 3000 else output_text
        boxed_in_end = extract_boxed_content(search_text)
        if boxed_in_end:
            return boxed_in_end[-1].strip()

        # Fallback: return last non-empty line (avoid returning numbers that could be token counts)
        lines = [line.strip() for line in output_text.split('\n') if line.strip()]
        filtered_lines = [l for l in lines if not re.match(r'^[\d,]+$', l) and 'token' not in l.lower() and 'max_' not in l.lower()]
        return filtered_lines[-1] if filtered_lines else (lines[-1] if lines else "")

    def _calculate_reward(self, final_answer: str, env_data: Env) -> float:
        """Calculate reward based on task type and answer correctness."""
        reward_function = REWARD_FUNCTIONS.get(self.task_type.lower())
        
        if reward_function is None:
            logger.warning(f"No reward function found for task_type: {self.task_type}, defaulting to 0.0")
            return 0.0
        
        try:
            reward = reward_function(final_answer, env_data)
            return float(reward)
        except Exception as e:
            logger.warning(f"Error calculating reward: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return 0.0
