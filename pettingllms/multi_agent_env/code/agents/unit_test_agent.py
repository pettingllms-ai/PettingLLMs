import copy
import logging
from typing import Any

from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env

logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


class UnitTestGenerationAgent(Agent):
    """
    Agent specialized for generating unit test cases.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Unit Test Generation Agent.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, env_data: Env):
        """
        Update the agent's internal prompt after an environment step.
        Rules:
        - If either state.current_code or state.current_test_input is None/empty, prompt to generate test cases.
        - Otherwise, refine or correct tests based on existing code and test cases.
        """
        # Save environment data
        self.env_data = env_data

        state = getattr(env_data, "state", None)
        agent_obs = getattr(env_data, "agent_observations", None)

        def as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join([str(v) for v in value])
            return str(value)

        if state is not None:
            question = as_text(getattr(state, "problem", ""))
            current_code = getattr(state, "current_code", None)
            current_test_input = getattr(state, "current_test_input", None)
            current_code_output = getattr(state, "current_code_output", None)
            current_test_output = getattr(state, "current_test_output", None)
            mismatch_testcases = getattr(state, "mismatch_testcases", None)
        elif agent_obs is not None:
            question = as_text(agent_obs.get("question", ""))
            current_code = agent_obs.get("current_code", None)
            current_test_input = agent_obs.get("current_test_input", None)
            current_code_output = agent_obs.get("current_code_output", None)
            current_test_output = agent_obs.get("current_test_output", None)
            mismatch_testcases = agent_obs.get("mismatch_testcases", None)
        else:
            question = ""
            current_code = None
            current_test_input = None
            current_code_output = None
            current_test_output = None
            mismatch_testcases = None

        need_generate = current_code in (None, "") or current_test_input in (None, "")

        if need_generate:
            # Test-case generation mode
            formatted_prompt = (
                f"You are a helpful assistant that generates test examples for coding tasks.\n\n"
                f"Problem:\n{question}\n\n"
                f"Please generate diverse, accurate, and discriminative test cases (inputs and outputs).\n"
                f"Cover normal, edge, and special cases.\n\n"
                f"Respond in the format:\n\n"
                f"**Test Input:**\n```\ninput here\n```\n\n"
                f"**Test Output:**\n```\noutput here\n```\n\n"
                f"**Explanation:**\nreasoning here."
            )
        else:
            # Test-case refinement mode
            formatted_prompt = (
                f"You are a helpful assistant that refines or corrects test examples for coding tasks.\n\n"
                f"Problem:\n{question}\n\n"
                f"Current code output:\n{as_text(current_code_output)}\n\n"
                f"Current tests (inputs):\n{as_text(current_test_input)}\n\n"
                f"Current tests (expected outputs):\n{as_text(current_test_output)}\n\n"
                f"Mismatch summary (if any):\n{as_text(mismatch_testcases)}\n\n"
                f"Please provide corrected or more discriminative tests while keeping format consistent.\n\n"
                f"Respond in the format:\n\n"
                f"**Test Input:**\n```\ninput here\n```\n\n"
                f"**Test Output:**\n```\noutput here\n```\n\n"
                f"**Explanation:**\nexplanation here."
            )

        self.agent_data.current_prompt = {"text": formatted_prompt, "image": None}
        
        
        
    def update_from_model(self, response: str):
        # Parse the response and update agent_data
        import re
        
        # Parse test cases
        test_cases = []
        
        # First, extract with backticked input/output/explanation per our prompt format
        pattern_input = r'\*\*Test Input:\*\*\s*```([\s\S]*?)```'
        pattern_output = r'\*\*Test Output:\*\*\s*```([\s\S]*?)```'
        pattern_explanation = r'\*\*Explanation:\*\*\s*([\s\S]*?)(?=\*\*Test Input:\*\*|$)'
        inputs = re.findall(pattern_input, response, re.DOTALL)
        outputs = re.findall(pattern_output, response, re.DOTALL)
        explanations = re.findall(pattern_explanation, response, re.DOTALL)

        # Fallback: extract without backticks using anchors
        if not inputs:
            inputs = re.findall(r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*|$)', response, re.DOTALL)
        if not outputs:
            outputs = re.findall(r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:\*\*|$)', response, re.DOTALL)

        # If counts mismatch, truncate to shortest length
        pair_count = min(len(inputs), len(outputs))
        for i in range(pair_count):
            test_case = {
                "input": inputs[i].strip(),
                "output": outputs[i].strip(),
                "explanation": (explanations[i].strip() if i < len(explanations) else "")
            }
            test_cases.append(test_case)

        # Final fallback: return raw text for environment handling/logging
        if not test_cases:
            test_cases = [{"input": "", "output": "", "explanation": response.strip()}]

        self.agent_data.current_action = test_cases
        return self.agent_data.current_action
    
    def calculate_reward(self, env_data: Env, mode: str = "sum") -> float:
        """
        Compute reward based on environment state, supporting three modes:
        - generated: use generated_pass_ratio (prefer generated_test_vs_generated_code_match_ratio, fallback to generated_test_vs_golden_code_match_ratio)
        - golden: use golden_pass_ratio (golden_test_vs_generated_code_match_ratio)
        - sum/both/others: sum of both
        """
        state = getattr(env_data, "state", None)
        generated_pass_ratio = 0.0
        golden_pass_ratio = 0.0

        if state is not None:
            # Generated tests vs generated code
            gen_vs_gen = getattr(state, "generated_test_vs_generated_code_match_ratio", None)
            # Generated tests vs golden code (as fallback)
            gen_vs_gold = getattr(state, "generated_test_vs_golden_code_match_ratio", None)
            # Golden tests vs generated code
            gold_vs_gen = getattr(state, "golden_test_vs_generated_code_match_ratio", None)

            if isinstance(gen_vs_gen, (int, float)):
                generated_pass_ratio = float(gen_vs_gen)
            elif isinstance(gen_vs_gold, (int, float)):
                generated_pass_ratio = float(gen_vs_gold)

            if isinstance(gold_vs_gen, (int, float)):
                golden_pass_ratio = float(gold_vs_gen)

        m = (mode or "sum").lower()
        if m in ("generated", "gen"):
            reward = generated_pass_ratio
        elif m in ("golden", "gold"):
            reward = golden_pass_ratio
        else:
            reward = generated_pass_ratio + golden_pass_ratio

        # Record and return
        self.agent_data.agent_reward = reward
        if self.agent_data.info is None:
            self.agent_data.info = {}
        self.agent_data.info.update({
            "generated_pass_ratio": generated_pass_ratio,
            "golden_pass_ratio": golden_pass_ratio,
            "reward_mode": m,
        })
        return reward

    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.agent_data = AgentData()