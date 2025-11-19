import copy
import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from typing import List
from math_verify import parse, verify
logger = logging.getLogger(__name__)


def truncatefn(s, length=600):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[:length//2] + "\n\n...(reasoning steps truncated)...\n\n" + s[-length//2:]


class ReasoningAgent(Agent):
    """
    Agent specialized for solving mathematical problems.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Math Solving Agent's data.
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
        reasoning_generated_solution_history = getattr(state, "reasoning_generated_solution_history", None)
        reasoning_extracted_answer_history = getattr(state, "reasoning_extracted_answer_history", None)
        
        if turn_idx == 0:
            # First turn: just present the problem
            formatted_prompt = (
                f"Problem:\n{problem}\n\n"
                f"Please think step by step and solve this problem.\n"
                f"You can continue reasoning in multiple turns.\n"
                f"When you have your answer, output it in \\boxed{{}} format.\n"
                f"Example: \\boxed{{123}}\n\n"
            )
        else:
            # Subsequent turns: show previous attempts and ask to continue or refine
            prompt_for_history = "Here is the history of your previous reasoning:\n"
            if reasoning_generated_solution_history is not None:
                for i in range(len(reasoning_generated_solution_history)):
                    prompt_for_history += f"\n--- Turn {i+1} ---\n{reasoning_generated_solution_history[i]}\n"
                    if reasoning_extracted_answer_history[i] is not None:
                        prompt_for_history += f"Extracted answer: {reasoning_extracted_answer_history[i]}\n"
            
            formatted_prompt = (
                f"Problem:\n{problem}\n\n"
                f"{prompt_for_history}\n"
                f"--- Current Turn ---\n"
                f"Based on your previous reasoning above, continue reasoning and refining your solution.\n"
                f"When you have your answer, output it in \\boxed{{}} format.\n"
                f"Example: \\boxed{{123}}\n\n"
            )
        
        self.current_prompt = {"text": formatted_prompt, "image": None}
        
    
    def update_from_model(self, response: str):
        
        self.current_action = response
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Process the generated reasoning solution and check for termination condition.
        The agent terminates when the extracted answer is the same as the previous turn.
        """
        # Store the full solution
        env_data.state.reasoning_generated_solution = truncatefn(self.current_action)
        env_data.state.reasoning_generated_solution_history.append(env_data.state.reasoning_generated_solution)
        self.action_history.append(self.current_action)
        
        # Try to extract answer from the current action
        extracted_answer = parse(self.current_action)
        env_data.state.reasoning_extracted_answer = extracted_answer
        
        # Check if answer is the same as previous turn (termination condition)
        should_terminate = False
        if len(env_data.state.reasoning_extracted_answer_history) > 0:
            previous_answer = env_data.state.reasoning_extracted_answer_history[-1]
            if extracted_answer is not None and previous_answer is not None:
                if extracted_answer == previous_answer:
                    should_terminate = True
        
        env_data.state.reasoning_extracted_answer_history.append(extracted_answer)
        self.answer_history.append(extracted_answer)
        
        # Evaluate correctness against ground truth if an answer was extracted
        if extracted_answer is not None:
            ground_truth_answer = env_data.state.ground_truth_answer
            if ground_truth_answer is not None:
                is_correct = verify(extracted_answer, parse(ground_truth_answer))
                env_data.state.reasoning_is_correct = is_correct
                env_data.state.success = is_correct
                self.success = is_correct
            else:
                # No ground truth available
                env_data.state.reasoning_is_correct = False
                env_data.state.success = False
                self.success = False
        else:
            # No valid answer extracted
            env_data.state.reasoning_is_correct = False
            env_data.state.success = False
            self.success = False
        
        # Terminate if answer is same as previous turn
        if should_terminate:
            env_data.done = True

        
    
    def calculate_reward(self, env_data: Env):
        """
        Calculate reward based on whether the final answer matches the ground truth.
        Reward is 1 if correct, 0 if incorrect.
        """
        # Reward is based on correctness of the answer compared to golden result
        if env_data.state.reasoning_is_correct is True:
            self.agent_reward = 1.0
        else:
            self.agent_reward = 0.0
        
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
