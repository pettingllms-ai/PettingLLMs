import copy
import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from typing import List
from pettingllms.multi_agent_env.code.code_utils import (
        evaluate_code_against_tests,
    )
logger = logging.getLogger(__name__)


def truncatefn(s, length=600):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[:length//2] + "\n\n...(code truncated)...\n\n" + s[-length//2:]


class CodeGenerationAgent(Agent):
    """
    Agent specialized for generating code to solve programming problems.
    Can see previous code history and decide when to terminate.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Code Generation Agent's data.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        # Save environment data
        self.env_data = env_data

        # Support passing either the raw environment (with state) or a wrapped Env
        state = getattr(env_data, "state", None)
        
        problem = getattr(state, "problem", None)
        generated_code_history = getattr(state, "generated_code_history", None)
        test_results_history = getattr(state, "test_results_history", None)
        
        if turn_idx == 0:
            # First turn: just present the problem
            formatted_prompt = (
                f"You are a helpful assistant that generates Python code to solve programming problems.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Please think step by step and generate code to solve this problem.\n"
                f"You can continue refining your code in multiple turns. In each turn, either:\n"
                f"1. Generate or refine your code, OR\n"
                f"2. When you are confident your code is correct, output your final code and then output <TERMINATE> to end.\n\n"
                f"Important Requirements:\n"
                f"- Your solution MUST read input using input() and write output using print().\n"
                f"- Do NOT hardcode or generate inputs yourself.\n"
                f"- First think about how many and what type of inputs you need.\n"
                f"- Write like: x = int(input()), b = int(input())\n"
                f"- Then generate the function to solve the problem.\n"
                f"- Finally print the result.\n\n"
                f"Respond in the format:\n"
                f"**Code:**\n```python\n# your code here\n```\n\n"
                f"When done, add:\n<TERMINATE>\n\n"
            )
        else:
            # Subsequent turns: show previous code attempts and test results
            prompt_for_history = "Here is the history of your previous code attempts:\n"
            if generated_code_history is not None:
                for i in range(len(generated_code_history)):
                    prompt_for_history += f"\n--- Turn {i+1} ---\n"
                    prompt_for_history += f"Code:\n{generated_code_history[i]}\n"
                    if test_results_history and i < len(test_results_history) and test_results_history[i]:
                        test_result = test_results_history[i]
                        prompt_for_history += f"Test Pass Ratio: {test_result.get('pass_ratio', 0.0):.2%}\n"
                        if test_result.get('failed_cases'):
                            prompt_for_history += f"Failed Test Cases:\n"
                            for case in test_result['failed_cases'][:3]:
                                prompt_for_history += f"  Input: {case.get('test_input', 'N/A')}\n"
                                prompt_for_history += f"  Expected: {case.get('expected_output', 'N/A')}\n"
                                prompt_for_history += f"  Got: {case.get('code_execution_output', 'N/A')}\n"
            
            formatted_prompt = (
                f"You are a helpful assistant that generates Python code to solve programming problems.\n\n"
                f"Problem:\n{problem}\n\n"
                f"{prompt_for_history}\n"
                f"--- Current Turn ---\n"
                f"Based on your previous attempts above, you can now:\n"
                f"1. Refine your code to fix the issues and pass all tests, OR\n"
                f"2. If you are confident your code is correct, output your final code and then output <TERMINATE> to end.\n\n"
                f"Important Requirements:\n"
                f"- Your solution MUST read input using input() and write output using print().\n"
                f"- Do NOT hardcode or generate inputs yourself.\n"
                f"- Analyze the failed test cases and fix the bugs.\n\n"
                f"Respond in the format:\n"
                f"**Code:**\n```python\n# your corrected code here\n```\n\n"
                f"When done, add:\n<TERMINATE>\n\n"
            )
        
        self.current_prompt = {"text": formatted_prompt, "image": None}
        
    
    def update_from_model(self, response: str):
        # Parse the response and update agent_data
        import re
        
        # Parse code
        code = ""
        
        # Try to match the code block in our prompt format
        matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
        if matches:
            code = matches[-1].strip()
        else:
            code = "Failed to extract code from response."
        
        # Update the agent's current action (environment expects a raw code string)
        self.current_action = code
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Process the generated code and check for termination token.
        The agent terminates when it outputs <TERMINATE> token.
        """
        # Store the full code
        gen_code = self.current_action
        env_data.state.generated_code = truncatefn(gen_code)
        
        # Initialize history lists if they don't exist
        if not hasattr(env_data.state, 'generated_code_history'):
            env_data.state.generated_code_history = []
        if not hasattr(env_data.state, 'test_results_history'):
            env_data.state.test_results_history = []
        
        env_data.state.generated_code_history.append(env_data.state.generated_code)
        self.action_history.append(gen_code)
        
        # Evaluate generated code against ground truth tests (if exists)
        ground_truth_test_input = env_data.state.ground_truth_test_input or []
        ground_truth_test_output = env_data.state.ground_truth_test_output or []
        passed_ratio = 0.0
        
        if isinstance(ground_truth_test_input, list) and isinstance(ground_truth_test_output, list) and ground_truth_test_input and ground_truth_test_output:
            try:
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    gen_code, ground_truth_test_input, ground_truth_test_output, timeout=30.0, ray_actor=env_worker, rollout_idx=self.rollout_idx
                )
            except Exception as e:
                logger.warning(f"Failed to evaluate code against tests: {e}")
                passed_ratio, passed_cases, failed_cases = 0.0, [], [{"error": str(e)}]
            
            env_data.state.ground_truth_test_vs_generated_code_match_cases = passed_cases
            env_data.state.ground_truth_test_vs_generated_code_mismatch_cases = failed_cases
            env_data.state.ground_truth_test_vs_generated_code_match_ratio = passed_ratio
            
            # Store test results in history
            test_result = {
                'pass_ratio': passed_ratio,
                'passed_cases': passed_cases,
                'failed_cases': failed_cases
            }
            env_data.state.test_results_history.append(test_result)
        else:
            env_data.state.test_results_history.append(None)
        
        # Check if agent has produced the termination token <TERMINATE>
        has_terminate_token = "<TERMINATE>" in self.current_action or "<TERMINATE>" in getattr(self, 'current_response', '')
        
        
            
        # Evaluate success based on test pass ratio
        if passed_ratio >= 1.0 and len(ground_truth_test_input) > 0:
            self.success = True
            env_data.state.success = True
            env_data.state.code_is_correct = True
        else:
            self.success = False
            env_data.state.success = False
            env_data.state.code_is_correct = False

        if has_terminate_token:
            # Agent has explicitly signaled completion with <TERMINATE>
            env_data.done = True

        
    
    def calculate_reward(self, env_data: Env):
        """
        Calculate reward based on test pass ratio.
        """
        # Reward is based on the test pass ratio
        self.agent_reward = env_data.state.ground_truth_test_vs_generated_code_match_ratio or 0.0
        self.reward_history.append(self.agent_reward)

    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.current_reward = None
        self.current_info = None
        self.current_action = None
        self.current_prompt = None
        self.current_response = None


