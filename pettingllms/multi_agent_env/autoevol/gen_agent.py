from typing import List, Dict, Any, Optional
import json
import logging
import re
import os
import subprocess
import asyncio
import numpy as np
import torch
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from verl.protocol import DataProto
from verl.utils.torch_functional import pad_sequence_to_length
from verl.utils.model import compute_position_id_with_mask
from tensordict import TensorDict
logger = logging.getLogger(__name__)
from pettingllms.multi_agent_env.autoevol.reward_function import REWARD_FUNCTIONS
from pettingllms.multi_agent_env.autoevol.utils import load_and_tokenize_jsonl

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
        """Update agent from environment data and generate Qwen-formatted prompt"""
        self.env_data = env_data

        # Get code generation prompt
        user_prompt_text = env_data.state.problem
        system_prompt_text = "You are an expert in designing Multi-Agent System workflows."

        # Format with Qwen chat template
        prompt_text = (
            f"<|im_start|>system\n{system_prompt_text}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        self.current_prompt = {"text": prompt_text, "image": None}



    def update_from_model(self, response: str):
        code = ""

        code_match = re.search(r"<code>\s*(.*?)\s*</code>", response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
            if matches:
                code = matches[-1].strip()
            else:

                code = "# Error: Could not extract code from the model response."
                logger.warning("Failed to extract code from model response")


        self.generated_code = code


        self.current_action = code

        return self.current_action

    def _replace_llm_config_in_code(self, code: str, llm_config: dict) -> str:
        """
        Replace the llm_config dictionary in generated mas.py code with actual LLM configuration.

        Args:
            code: The generated Python code containing llm_config
            llm_config: Dictionary with keys: server_address, model_name, api_key, temperature

        Returns:
            Modified code with replaced llm_config
        """
        # Extract configuration values
        server_address = llm_config.get("server_address", "")
        model_name = llm_config.get("model_name", "gpt-4")
        api_key = llm_config.get("api_key", "")
        temperature = llm_config.get("temperature", 0.2)

        # Ensure server_address has http:// prefix and /v1 suffix
        if server_address and not server_address.startswith(('http://', 'https://')):
            server_address = f"http://{server_address}"

        # Add /v1 suffix if not present (required for OpenAI SDK compatibility)
        if server_address and not server_address.endswith('/v1'):
            server_address = f"{server_address}/v1"

        # Build the replacement llm_config string
        new_llm_config = f'''llm_config = {{
    "config_list": [{{
        "model": "{model_name}",
        "api_key": "{api_key}",
        "base_url": "{server_address}"
    }}],
    "temperature": {temperature},
}}'''

        # Pattern to match llm_config = { ... } including nested braces
        # This pattern matches from 'llm_config =' to the closing '}' that matches the opening '{'
        pattern = r'llm_config\s*=\s*\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'

        # Replace the llm_config
        modified_code = re.sub(pattern, new_llm_config, code, count=1, flags=re.DOTALL)

        if modified_code == code:
            logger.warning("Failed to replace llm_config in generated code - pattern not found")
        else:
            logger.info(f"Replaced llm_config with: model={model_name}, base_url={server_address}")

        return modified_code

    def _replace_question_in_code(self, code: str, question: str) -> str:
        """
        Replace hardcoded question/problem in the generated mas.py code with the actual question.

        This function looks for common patterns where questions appear in system messages:
        - After "Problem:" or "problem:"
        - In long text blocks within system_message
        - Between specific markers

        Args:
            code: The generated Python code containing hardcoded question
            question: The actual question from env_data.state.problem

        Returns:
            Modified code with replaced question
        """
        # Escape backslashes in the question to avoid Python string escape issues
        # This is important for LaTeX math expressions like \{, \cdots, \tfrac, etc.
        escaped_question = question.replace('\\', '\\\\')

        # Strategy 1: Try to find and replace content after "Problem:" in system_message
        # This pattern matches: system_message="""...\n\nProblem: <old question>\n\n..."""
        pattern1 = r'(system_message\s*=\s*"""[^"]*?)(Problem:\s*)(.*?)(\n\n|\n""")'

        def replace_after_problem(match):
            prefix = match.group(1)  # Everything before "Problem:"
            problem_marker = match.group(2)  # "Problem:" with whitespace
            # Skip the old question text (group 3)
            suffix = match.group(4)  # The ending (newlines or end of string)
            return f"{prefix}{problem_marker}{escaped_question}{suffix}"

        modified_code = re.sub(pattern1, replace_after_problem, code, flags=re.DOTALL)

        # Strategy 2: If no "Problem:" marker found, try to replace entire long text in system_message
        # Only if Strategy 1 didn't change anything
        if modified_code == code:
            # Find system_message with multi-line content and replace with question
            pattern2 = r'(system_message\s*=\s*""")(.*?)(""")'

            def replace_system_message(match):
                prefix = match.group(1)
                old_content = match.group(2).strip()
                suffix = match.group(3)

                # Only replace if the old content looks like a problem (long text, not a simple instruction)
                if len(old_content) > 100 and ('solve' in old_content.lower() or 'find' in old_content.lower()):
                    # Keep the first line if it's a role description
                    lines = old_content.split('\n')
                    if len(lines) > 0 and len(lines[0]) < 100 and 'You are' in lines[0]:
                        role_line = lines[0]
                        new_content = f"{role_line}\n\n{escaped_question}"
                    else:
                        new_content = escaped_question
                    return f"{prefix}{new_content}{suffix}"
                return match.group(0)  # Return unchanged if doesn't match criteria

            modified_code = re.sub(pattern2, replace_system_message, code, flags=re.DOTALL)

        if modified_code != code:
            logger.info(f"Replaced question in generated code")
        else:
            logger.warning("Failed to replace question in generated code - no suitable pattern found")

        return modified_code

    async def step(self, env_data: Env, env_worker: Any = None, output_dir: str = None,
                   server_address: str = None, model_name: str = None, tokenizer=None,
                   max_prompt_length: int = 2048, max_response_length: int = 2048,
                   llm_config_for_mas: dict = None):
        """
        Execute MAS Designer step: generate mas.py, run it with vLLM access, calculate reward.

        Returns:
            Tuple[List[Tuple[DataProto, str]], float]:
                - tokenized_trajectories: List of (DataProto, response_text) tuples
                - final_reward: Reward score from task-specific reward function
        """
        

        # Ensure output directory is provided
        if output_dir is None:
            raise ValueError("output_dir must be provided to step()")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Prepare the code with necessary imports and path setup
        dyevolve_dir = os.path.dirname(os.path.abspath(__file__))

        # Add environment setup at the beginning of the code



        # Combine all parts
        

        # Save generated code to mas.py
        mas_py_path = os.path.join(output_dir, "mas.py")
        # Use absolute path for trajectory file to avoid path resolution issues
        self.trajectory_json_path = os.path.abspath(os.path.join(output_dir, "traj.json"))

        trajectory_output_code = f"""
# Automatically save executor conversations after workflow execution
try:
    from ag2_tracer import get_global_tracker
    tracker = get_global_tracker()
    if tracker.agent_conversations:  # Only save if there are conversations
        import os
        from datetime import datetime

        # Use absolute path to avoid cwd-related issues
        trajectory_file = r'{self.trajectory_json_path}'

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(trajectory_file), exist_ok=True)

        tracker.save_all(filepath=trajectory_file, append=False)
        print(f"\\n[Conversation data saved to {{trajectory_file}}]")
except Exception as e:
    # Silently fail - don't interrupt workflow execution
    print(f"\\n[Warning: Failed to save executor conversations: {{e}}]")
    pass
"""
        full_code = self.generated_code + "\n" + trajectory_output_code

        # Replace llm_config with actual LLM configuration
        if llm_config_for_mas is not None:
            full_code = self._replace_llm_config_in_code(full_code, llm_config_for_mas)

        # Replace hardcoded question with actual question from env_data
        if env_data and hasattr(env_data.state, 'problem'):
            actual_question = env_data.state.problem
            full_code = self._replace_question_in_code(full_code, actual_question)

        with open(mas_py_path, 'w') as f:
            f.write(full_code)

        logger.info(f"Saved MAS code to {mas_py_path}")

        # Run the mas.py file in Ray Docker Worker environment
        try:
            # Read and execute the generated MAS code
            with open(mas_py_path, 'r') as f:
                mas_code = f.read()

            execution_timeout = 300.0

            # Execute code using Ray worker or subprocess
            if env_worker is not None:
                logger.info(f"Executing MAS code in Ray Docker Worker for rollout {self.rollout_idx}")
                from pettingllms.multi_agent_env.math.math_worker import get_code_execution_output

                stdout = await get_code_execution_output(code=mas_code, timeout=execution_timeout, ray_actor=env_worker)
                stderr = ""

                # Check for Ray execution errors
                if isinstance(stdout, str):
                    if stdout.startswith("error:"):
                        logger.error(f"Ray execution failed: {stdout}")
                        stderr, stdout = stdout, ""
                    elif stdout == "timeout":
                        logger.error(f"Ray execution timed out for rollout {self.rollout_idx}")
                        raise subprocess.TimeoutExpired(mas_py_path, execution_timeout)
            else:
                logger.warning("env_worker is None, falling back to subprocess execution")

                # Setup environment with virtual environment and PYTHONPATH
                env = os.environ.copy()

                # Get paths
                workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
                venv_python = os.path.join(workspace_root, "pettingllms_venv/bin/python")
                autoevol_dir = os.path.dirname(os.path.abspath(__file__))

                # Add autoevol directory to PYTHONPATH for ag2_tools and ag2_tracer
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{autoevol_dir}:{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = autoevol_dir

                # Use venv python if available, otherwise use system python
                python_executable = venv_python if os.path.exists(venv_python) else 'python'

                result = subprocess.run(
                    [python_executable, mas_py_path],
                    capture_output=True,
                    text=True,
                    timeout=execution_timeout,
                    cwd=output_dir,
                    env=env
                )
                stdout, stderr = result.stdout, result.stderr

            # Save execution output to file (both stdout and stderr)
            output_txt_path = os.path.join(output_dir, "output.txt")
            with open(output_txt_path, 'w') as f:
                if stdout:
                    f.write("=== STDOUT ===\n")
                    f.write(stdout)
                if stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(stderr)
                if not stdout and not stderr:
                    f.write("(No output captured)\n")
            logger.info(f"Saved execution output to {output_txt_path}")

            # Extract summary and trajectory from output
            summary = self._extract_summary(stdout)
            trajectory_store = self._extract_trajectory_from_stdout(stdout)
            self.trajectory_store = trajectory_store if trajectory_store else {}

            if trajectory_store:
                logger.info(f"Extracted {len(trajectory_store)} trajectory entries from stdout")
            else:
                logger.warning("No trajectory data found in stdout")

            # Load and tokenize trajectory data from saved JSONL file if tokenizer provided
            tokenized_trajectories = []
            if tokenizer is not None:
                trajectory_file = self.trajectory_json_path
                if not os.path.exists(trajectory_file):
                    logger.warning(f"Trajectory file {trajectory_file} not found, skipping tokenization")
                    self.tokenized_trajectories = []
                else:
                    # Use the new load_and_tokenize_jsonl function from utils
                    tokenized_trajectories = load_and_tokenize_jsonl(
                        trajectory_file, tokenizer, max_prompt_length, max_response_length
                    )
                    if tokenized_trajectories:
                        logger.info(f"Tokenized {len(tokenized_trajectories)} trajectory turns")
                        # Store tokenized trajectories in a new attribute
                        self.tokenized_trajectories = tokenized_trajectories
                    else:
                        logger.warning("No tokenized trajectories generated")
                        self.tokenized_trajectories = []

            # Log stderr if there were errors
            if stderr:
                logger.warning(f"MAS stderr output: {stderr[:500]}")

            # Calculate reward using task-specific reward function
            final_reward = 0.0
            if self.task_type in REWARD_FUNCTIONS:
                reward_func = REWARD_FUNCTIONS[self.task_type]
                final_reward = reward_func(summary, env_data)
                logger.info(f"Rollout {self.rollout_idx}: final_reward={final_reward}")
            else:
                logger.warning(f"No reward function found for task_type={self.task_type}, defaulting to 0.0")
                final_reward = 0.0

            self.agent_reward = final_reward
            self.reward_history.append(final_reward)

            # Return tokenized trajectories and final reward
            return tokenized_trajectories, final_reward

        except subprocess.TimeoutExpired:
            logger.error(f"MAS execution timed out for rollout {self.rollout_idx}")
            return [], 0.0
        except Exception as e:
            logger.error(f"Error executing MAS: {e}")
            return [], 0.0



    def _extract_summary(self, stdout: str) -> str:
        """Extract summary from workflow output"""
        start_marker = "WORKFLOW_SUMMARY_START"
        end_marker = "WORKFLOW_SUMMARY_END"

        if start_marker in stdout and end_marker in stdout:
            start_idx = stdout.find(start_marker) + len(start_marker)
            end_idx = stdout.find(end_marker)
            summary = stdout[start_idx:end_idx].strip()
            return summary
        else:
            lines = [line.strip() for line in stdout.split('\n') if line.strip()]
            return lines[-1] if lines else ""
    
    def _extract_trajectory_from_stdout(self, stdout: str) -> dict:
        """Extract trajectory data from subprocess stdout"""
        import pickle
        import base64

        start_marker = "TRAJECTORY_DATA_START"
        end_marker = "TRAJECTORY_DATA_END"

        if start_marker in stdout and end_marker in stdout:
            start_idx = stdout.find(start_marker) + len(start_marker)
            end_idx = stdout.find(end_marker)
            trajectory_b64 = stdout[start_idx:end_idx].strip()

            trajectory_bytes = base64.b64decode(trajectory_b64)
            trajectory_store = pickle.loads(trajectory_bytes)
            return trajectory_store
        else:
            return {}

