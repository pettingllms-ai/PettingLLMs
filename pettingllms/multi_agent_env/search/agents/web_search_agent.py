import logging
from typing import Any, List
from pettingllms.multi_agent_env.base.agent import Agent
from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.search.search_utils import (
    extract_search_query_from_response,
    extract_answer_from_response,
    simulate_web_search,
    format_search_results,
    truncatefn,
    clean_search_text
)

logger = logging.getLogger(__name__)

class WebSearchAgent(Agent):
    """
    Agent specialized for searching and answering sub-questions using external search tools.
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the Web Search Agent.
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
        self.env_data = env_data
        state = getattr(env_data, "state", None)

        question = getattr(state, "question", None)
        
        # Decomposition state
        sub_questions = getattr(state, "sub_questions", [])
        sub_answers = getattr(state, "sub_answers", [])
        current_sub_question_index = getattr(state, "current_sub_question_index", 0)
        decomposition_complete = getattr(state, "decomposition_complete", False)
        
        # Search state
        search_results = getattr(state, "web_search_formatted_results", "")
        
        # Check if we have sub-questions to work on
        if not decomposition_complete or not sub_questions:
            formatted_prompt = (
                f"You are a web search agent waiting for question decomposition.\n\n"
                f"Original question: {question}\n\n"
                f"Please wait for the reasoning agent to decompose the question into sub-questions "
                f"before proceeding with search and answering.\n"
            )
        elif current_sub_question_index < len(sub_questions):
            # Work on current sub-question
            current_sub_question = sub_questions[current_sub_question_index]
            
            # Check if we need to search or answer
            if not search_results or "No search results found" in search_results:
                # Need to search for this sub-question
                formatted_prompt = (
                    f"You are a web search agent working on sub-questions.\n\n"
                    f"Original question: {question}\n\n"
                    f"Sub-question {current_sub_question_index + 1}/{len(sub_questions)}: {current_sub_question}\n\n"
                )
                
                # Show previous sub-question answers for context
                if current_sub_question_index > 0:
                    formatted_prompt += "Previous sub-question answers:\n"
                    for i in range(current_sub_question_index):
                        if i < len(sub_answers) and sub_answers[i]:
                            formatted_prompt += f"{i+1}. {sub_questions[i]}\n   Answer: {sub_answers[i]}\n\n"
                
                formatted_prompt += (
                    f"Your task is to search for information to answer this sub-question.\n\n"
                    f"Please formulate a search query using this format:\n"
                    f"**Search Query:** [your search query here]\n\n"
                    f"Make your search query specific and relevant to the sub-question."
                )
            else:
                # Have search results, need to answer the sub-question
                formatted_prompt = (
                    f"You are analyzing search results to answer a sub-question.\n\n"
                    f"Original question: {question}\n\n"
                    f"Sub-question {current_sub_question_index + 1}/{len(sub_questions)}: {current_sub_question}\n\n"
                )
                
                # Show previous sub-question answers for context
                if current_sub_question_index > 0:
                    formatted_prompt += "Previous sub-question answers:\n"
                    for i in range(current_sub_question_index):
                        if i < len(sub_answers) and sub_answers[i]:
                            formatted_prompt += f"{i+1}. {sub_questions[i]}\n   Answer: {sub_answers[i]}\n\n"
                
                formatted_prompt += (
                    f"Search results:\n{truncatefn(search_results, 1500)}\n\n"
                    f"Based on the search results above, please answer the sub-question with a short, "
                    f"concise response. If the search results don't contain enough information, "
                    f"use your own knowledge.\n\n"
                    f"Format your response as:\n"
                    f"**Answer:** [Your concise answer to the sub-question]\n"
                )
        else:
            # All sub-questions completed
            formatted_prompt = (
                f"You have completed searching and answering all sub-questions.\n\n"
                f"Original question: {question}\n\n"
                f"All sub-questions have been answered. The reasoning agent will now synthesize "
                f"the final answer based on your search results and sub-question answers.\n"
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
        Process the generated web search solution and evaluate it against the ground truth.
        
        Args:
            env_data: Environment data
            env_worker: Optional environment worker for parallel execution
        """
        generated_response = self.current_action
        env_data.state.web_search_generated_response = generated_response
        env_data.state.web_search_generated_response_list.append(generated_response)

        # Check if we have sub-questions to work on
        if not env_data.state.decomposition_complete or not env_data.state.sub_questions:
            return

        current_sub_question_index = env_data.state.current_sub_question_index
        sub_questions = env_data.state.sub_questions
        
        if current_sub_question_index >= len(sub_questions):
            return

        # Extract and execute search query if present
        search_query = extract_search_query_from_response(self.current_action)
        if search_query:
            # Clean and optimize the search query
            search_query = clean_search_text(search_query)
            
            # Store the search query
            env_data.state.web_search_queries.append(search_query)
            
            # Perform web search
            search_results = simulate_web_search(search_query, max_results=5)
            env_data.state.web_search_results.extend(search_results)
            
            # Format search results for display
            formatted_results = format_search_results(search_results)
            env_data.state.web_search_formatted_results = formatted_results
            
            logger.info(f"Web search performed for sub-question {current_sub_question_index + 1} with query: '{search_query}'")
        else:
            # Check if this is an answer to the current sub-question
            extracted_answer = extract_answer_from_response(self.current_action)
            if extracted_answer:
                # Store the answer for the current sub-question
                if current_sub_question_index < len(env_data.state.sub_answers):
                    env_data.state.sub_answers[current_sub_question_index] = extracted_answer
                    
                    # Move to next sub-question
                    env_data.state.current_sub_question_index += 1
                    
                    # Clear search results for next sub-question
                    env_data.state.web_search_formatted_results = ""
                    
                    logger.info(f"Answer stored for sub-question {current_sub_question_index + 1}: {extracted_answer}")
                    
                    # Check if all sub-questions are completed
                    if env_data.state.current_sub_question_index >= len(sub_questions):
                        logger.info("All sub-questions have been answered")
                        self.agent_reward = 0.5  # Reward for completing all sub-questions
                    else:
                        self.agent_reward = 0.1  # Small reward for answering a sub-question

        # Store final extracted answer for tracking
        final_extracted_answer = extract_answer_from_response(self.current_action)
        env_data.state.web_search_extracted_answer = final_extracted_answer
        if final_extracted_answer is not None:
            env_data.state.web_search_extracted_answer_list.append(final_extracted_answer)
        else:
            env_data.state.web_search_extracted_answer_list.append("No answer found")

        # Ensure agent_reward is not None before converting to float
        if self.agent_reward is None:
            self.agent_reward = 0.0

    def reset(self):
        """Reset the agent's internal state for a new episode."""
        super().reset()  # Call parent reset