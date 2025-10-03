import copy
import logging
from typing import Any
from pettingllms.multi_agent_env.base.agent import Agent, AgentData
from pettingllms.multi_agent_env.base.env import Env
from typing import List
from pettingllms.multi_agent_env.math.math_utils import extract_reasoning_steps
from pettingllms.multi_agent_env.math.math_utils import evaluate_math_solution
from math_verify import parse, verify
logger = logging.getLogger(__name__)


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[:500] + "\n\n...(reasoning steps truncated)...\n\n" + s[-500:]


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
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        # Save environment data
        self.env_data = env_data

        # Support passing either the raw environment (with state) or a wrapped Env
        state = getattr(env_data, "state", None)
        agent_obs = getattr(env_data, "agent_observations", None)

        def as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join([str(v) for v in value])
            return str(value)
        

        problem = getattr(state, "problem", None)
        reasoning_solution = getattr(state, "reasoning_generated_solution", None)
        reasoning_extracted_answer = getattr(state, "reasoning_extracted_answer", None)
        code_solution = getattr(state, "code_generated_solution", None)
        code_extracted_answer = getattr(state, "code_extracted_answer", None)
        
        
        if turn_idx == 0:
           formatted_prompt = (
        f"Problem:\n{problem}\n\n"
        f"Please think step by step and output the final answer in \\boxed{{}} format.\n"
        f"Example: \\boxed{{123}}\n\n"
    )
        else:
            formatted_prompt = (
                f"You are a helpful assistant that refines mathematical solutions through reasoning.\n\n"
                f"Problem:\n{problem}\n\n"
                f"Your previous reasoning solution:\n{truncatefn(as_text(reasoning_solution), 1000)}\n\n And your extracted answer is {reasoning_extracted_answer}.\n"
                f"Another LLM using python code to solve the problem and excuted the script.\n"
                f"The code agent's script is {truncatefn(as_text(code_solution), 1000)}\n"
                f"The code agent's execution result is {code_extracted_answer}\n"
                f"Please firstly refer the code agent's execution result to judge whose answer is correct. And then solve the problem again.\n"
                f"The code agent's answer is possible to be correct and possible to be incorrect.\n"
                f"Then solve the problem again.\n"
                
            )
            
            formatted_prompt += (
               f"Before giving the full reasoning, please summarize the key reasoning steps clearly.\n"
                f"Output them in the following format:\n\n"
                f"**Reasoning Steps:**\n```reasoning steps here```\n\n"
                f"Put your final answer in the format of \\boxed{{<answer>}}\n"
                f"For the final answer, only output the answer after \\boxed{{}}, no other text.\n"
                f"Example: \\boxed{{123}}\n\n"
            )
        
        self.current_prompt = {"text": formatted_prompt, "image": None}
        
    
    def update_from_model(self, response: str):
        
        self.current_action = response
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Process the generated reasoning solution and evaluate it against the ground truth.
        """
       
        generated_solution = self.current_action
        env_data.state.reasoning_generated_solution = generated_solution
        env_data.state.reasoning_generated_solution_list.append(generated_solution)

        # 2) Extract answer from the reasoning solution
        extracted_answer_list = parse(self.current_action)
        # parse返回一个列表，取第一个元素作为提取的答案
        extracted_answer = extracted_answer_list[0] if extracted_answer_list else None
        env_data.state.reasoning_extracted_answer = extracted_answer
        if extracted_answer is not None:
            env_data.state.reasoning_extracted_answer_list.append(extracted_answer)
        else:
            env_data.state.reasoning_extracted_answer_list.append("No answer found")
        
        # 3) Evaluate correctness
        ground_truth_answer = env_data.state.ground_truth_answer
        is_correct = False
        
        if extracted_answer is not None and ground_truth_answer is not None:
            try:
                # use the utils in this project to evaluate consistently
                is_correct = verify(extracted_answer, parse(ground_truth_answer))
                env_data.state.reasoning_is_correct = is_correct
                
                if is_correct:
                    self.done = True
                    self.is_pass = True
                    self.agent_reward = 1.0
                    self.value = 1.0
                else:
                    self.agent_reward = 0.0
                    

                    
            except Exception as e:
                print(f"Warning: Failed to evaluate reasoning solution: {e}")
                is_correct = False
                self.agent_reward = 0.0
                if not hasattr(env_data.state, 'reasoning_is_correct'):
                    env_data.state.reasoning_is_correct = False
                else:
                    env_data.state.reasoning_is_correct = False
        else:
            self.agent_reward = 0.0
            if not hasattr(env_data.state, 'reasoning_is_correct'):
                env_data.state.reasoning_is_correct = False
            else:
                env_data.state.reasoning_is_correct = False

        # Ensure agent_reward is not None before converting to float
        if self.agent_reward is None:
            self.agent_reward = 0.0
        self.reward_history.append(float(self.agent_reward))

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
