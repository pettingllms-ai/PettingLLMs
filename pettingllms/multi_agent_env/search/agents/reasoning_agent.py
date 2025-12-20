import copy
import logging
from typing import Any, List
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.search.search_utils import (
    extract_answer_from_response,
    evaluate_search_answer,
    truncatefn,
    parse_sub_questions_with_references,
    format_sub_answers_text
)

logger = logging.getLogger(__name__)

class ReasoningAgent(Agent):
    """
    Agent specialized for question decomposition and final answer synthesis.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Reasoning Agent.
        """
        super().__init__()
        self.rollout_idx = rollout_idx
        # Accept other unrelated keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data: Env):
        """
        Update agent state based on environment data and generate appropriate prompt.
        
        Args:
            turn_idx: Current turn index
            env_data: Environment data containing current state
        """
        # Save environment data
        self.env_data = env_data
        state = getattr(env_data, "state", None)
        
        def as_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, list):
                return "\n".join([str(v) for v in value])
            return str(value)

        question = getattr(state, "question", None)
        search_hint = getattr(state, "search_hint", "")
        
        # Decomposition state
        sub_questions = getattr(state, "sub_questions", [])
        sub_answers = getattr(state, "sub_answers", [])
        decomposition_complete = getattr(state, "decomposition_complete", False)
        
        # Web search results
        web_search_formatted_results = getattr(state, "web_search_formatted_results", "")
        
        # Previous responses
        reasoning_response = getattr(state, "reasoning_generated_response", None)

        if turn_idx == 0:
            # First turn: Question decomposition
            formatted_prompt = (
                f"You are an expert reasoning agent specialized in question decomposition and synthesis.\n\n"
                f"Question: {question}\n\n"
            )
            
            if search_hint:
                formatted_prompt += f"Search hint: {search_hint}\n\n"
            
            formatted_prompt += (
                f"Your task is to break down this complex question into multiple specific sub-questions "
                f"that address individual components of the original question.\n\n"
                f"Instructions:\n"
                f"1. Analyze the main question and identify its key components\n"
                f"2. Create 3-5 sub-questions that each focus on a specific aspect\n"
                f"3. Mark each sub-question with ### at the beginning\n"
                f"4. If you need to refer to answers from earlier sub-questions, use #1, #2, etc.\n"
                f"5. Ensure sub-questions are specific and answerable through search\n\n"
                f"Example format:\n"
                f"### What is the population of Tokyo?\n"
                f"### What is the GDP of Japan in 2023?\n"
                f"### How does Tokyo's population relate to Japan's economy based on #1 and #2?\n\n"
                f"Please decompose the question:"
            )
        elif not decomposition_complete:
            # Decomposition refinement if needed
            formatted_prompt = (
                f"You are refining the question decomposition.\n\n"
                f"Original question: {question}\n\n"
                f"Your previous decomposition:\n{truncatefn(as_text(reasoning_response), 1000)}\n\n"
                f"Please refine or confirm your decomposition. Make sure each sub-question is:\n"
                f"1. Specific and clear\n"
                f"2. Answerable through search\n"
                f"3. Necessary for answering the main question\n\n"
                f"Refined decomposition:"
            )
        else:
            # Final synthesis: Generate answer based on sub-question answers
            sub_answer_text = format_sub_answers_text(sub_questions, sub_answers)
            
            formatted_prompt = (
                f"You are synthesizing the final answer based on sub-question results.\n\n"
                f"Original question: {question}\n\n"
                f"{sub_answer_text}\n"
            )
            
            if web_search_formatted_results:
                formatted_prompt += f"Additional search context:\n{truncatefn(web_search_formatted_results, 1000)}\n\n"
            
            formatted_prompt += (
                f"Based on the sub-questions and their answers, please provide a comprehensive final answer "
                f"to the original question. Make sure your response is grounded in the provided information "
                f"and provides clear reasoning followed by a concise conclusion.\n\n"
                f"Format your response as:\n"
                f"**Reasoning:**\n"
                f"[Your analysis using the sub-question answers]\n\n"
                f"**Final Answer:** [Your concise final answer]\n"
            )

        self.current_prompt = {"text": formatted_prompt, "image": None}

    def update_from_model(self, response: str):
        """
        Parse model response and update agent state.
        
        Args:
            response: Raw response from the language model
            
        Returns:
            Processed response
        """
        self.current_action = response
        return self.current_action

    async def step(self, env_data: Env, env_worker: Any = None):
        """
        Process the generated reasoning response.
        
        Args:
            env_data: Environment data
            env_worker: Optional environment worker for parallel execution
        """
        generated_response = self.current_action
        env_data.state.reasoning_generated_response = generated_response
        env_data.state.reasoning_generated_response_list.append(generated_response)

        # Check if this is decomposition or synthesis phase
        if not env_data.state.decomposition_complete:
            # Process references and prepare sub-questions
            sub_questions = parse_sub_questions_with_references(generated_response)
            env_data.state.sub_questions = sub_questions
            
            # Initialize sub-answers list
            env_data.state.sub_answers = [""] * len(sub_questions)
            env_data.state.current_sub_question_index = 0
            
            if sub_questions:
                env_data.state.decomposition_complete = True
                logger.info(f"Question decomposed into {len(sub_questions)} sub-questions")
            else:
                logger.warning("No valid sub-questions extracted from decomposition")
        else:
            # Final synthesis phase - extract answer
            extracted_answer = extract_answer_from_response(self.current_action)
            env_data.state.reasoning_extracted_answer = extracted_answer
            if extracted_answer is not None:
                env_data.state.reasoning_extracted_answer_list.append(extracted_answer)
            else:
                env_data.state.reasoning_extracted_answer_list.append("No answer found")

            # Evaluate correctness
            ground_truth_answer = env_data.state.ground_truth_answer
            is_correct = False
            
            if extracted_answer is not None and ground_truth_answer is not None:
                try:
                    is_correct = evaluate_search_answer(extracted_answer, ground_truth_answer)
                    env_data.state.reasoning_is_correct = is_correct
                    
                    if is_correct:
                        self.done = True
                        self.is_pass = True
                        self.agent_reward = 1.0
                    else:
                        self.agent_reward = 0.0
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate reasoning solution: {e}")
                    is_correct = False
                    self.agent_reward = 0.0
                    env_data.state.reasoning_is_correct = False
            else:
                self.agent_reward = 0.0
                env_data.state.reasoning_is_correct = False

            # Ensure agent_reward is not None before converting to float
            if self.agent_reward is None:
                self.agent_reward = 0.0

    def reset(self):
        """Reset the agent's internal state for a new episode."""
        super().reset()  # Call parent reset