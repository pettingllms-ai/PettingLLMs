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
from pettingllms.multi_agent_env.autoevol.data_utils import load_and_tokenize_jsonl

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
        user_prompt_text = "Design Multi Agent System for the Question:" + env_data.state.problem
        system_prompt_text = "You are an expert in designing Multi-Agent System workflows."

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

# Import necessary modules for verl integration
from pettingllms.multi_agent_env.autoevol.utils.BaseOpenAI import AIClient

# Debug: Print tokenizer_path for verification
_tokenizer_path = {repr(tokenizer_path)}
print(f"[DEBUG] mas.py: tokenizer_path = {{_tokenizer_path}}")
print(f"[DEBUG] mas.py: tokenizer_path type = {{type(_tokenizer_path)}}")

# Create AIClient with verl integration
_server_addr = "{server_address}"
_api_base = _server_addr if _server_addr.startswith('http') else f"http://{{{{_server_addr}}}}"
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

        try:
            # Use Ray worker for execution (better resource management)
            with open(mas_py_path, 'r') as f:
                script_content = f.read()
            
            from pettingllms.multi_agent_env.math.math_worker import _await_ray_object_ref
            import ray
            
            timeout_buffer = 20
            total_timeout = step_timeout + timeout_buffer
            
            obj_ref = env_worker.run.remote(script_content, step_timeout)
            output_text = await _await_ray_object_ref(obj_ref, total_timeout)
            
            # Save output to file
            with open(output_txt_path, 'w') as f:
                f.write(output_text)
            
            if "[SUCCESSSAVED]" in output_text:
                execution_success = True

            # Load DataProto from file if exists
            if os.path.exists(dataproto_pkl_path):
                with open(dataproto_pkl_path, 'rb') as f:
                    workflow_dataproto_list = pickle.load(f)
                logger.info(f"Loaded {len(workflow_dataproto_list)} DataProto entries from {dataproto_pkl_path}")

            # Extract final answer from output
            final_answer = self._extract_final_answer(output_text)

            # Calculate reward based on task type
            reward = self._calculate_reward(final_answer, env_data)
            logger.info(f"Calculated reward: {reward} for final_answer: {final_answer} and golden answer: {env_data.state.ground_truth_answer}")

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
        
        Looks for **Final Answer** marker and extracts the last number.
        
        Args:
            output_text: Output from MAS execution
            
        Returns:
            Extracted final answer string
        """
        import re
        
        # Strategy 1: Look for **Final Answer** and extract last number
        final_answer_match = re.search(
            r'\*\*Final\s*Answer\s*\*{0,2}[:\s]*(.*?)(?:={3,}|\Z)',
            output_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if final_answer_match:
            final_answer_section = final_answer_match.group(1).strip()
            
            # Extract last number from final answer section
            number_pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
            numbers = re.findall(number_pattern, final_answer_section)
            if numbers:
                return numbers[-1]
        
        # Strategy 2: Look for WORKFLOW_SUMMARY_START/END markers
        workflow_match = re.search(
            r'WORKFLOW_SUMMARY_START\s*(.*?)\s*WORKFLOW_SUMMARY_END',
            output_text,
            re.DOTALL | re.IGNORECASE
        )
        
        if workflow_match:
            summary_content = workflow_match.group(1).strip()
            
            # Try to extract the last number from summary
            number_pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
            numbers = re.findall(number_pattern, summary_content)
            if numbers:
                return numbers[-1]
            
            # Fallback to last non-empty line
            lines = [line.strip() for line in summary_content.split('\n') if line.strip()]
            if lines:
                return lines[-1]
        
        # Strategy 3: Extract last number from entire output
        number_pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
        numbers = re.findall(number_pattern, output_text)
        if numbers:
            return numbers[-1]
        
        # Fallback: return last non-empty line
        lines = [line.strip() for line in output_text.split('\n') if line.strip()]
        return lines[-1] if lines else ""
    
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