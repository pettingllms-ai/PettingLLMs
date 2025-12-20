import logging
import copy
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

from pettingllms.multi_agent_env.base.env import Env
from pettingllms.multi_agent_env.search.search_utils import (
    load_search_problem_batch,
    simulate_web_search,
    format_search_results
)

logger = logging.getLogger(__name__)

@dataclass
class SearchEnvState:
    """State class for Search environment"""
    question: str = None
    ground_truth_answer: str = None
    search_hint: str = None
    
    # Task decomposition state
    sub_questions: List[str] = field(default_factory=list)
    sub_answers: List[str] = field(default_factory=list)
    current_sub_question_index: int = 0
    decomposition_complete: bool = False
    
    # Reasoning agent state
    reasoning_generated_response: str = None
    reasoning_generated_response_list: List[str] = field(default_factory=list)
    reasoning_extracted_answer: str = None
    reasoning_extracted_answer_list: List[str] = field(default_factory=list)
    reasoning_is_correct: bool = False
    
    # Web search agent state  
    web_search_queries: List[str] = field(default_factory=list)
    web_search_results: List[Dict] = field(default_factory=list)
    web_search_formatted_results: str = ""
    web_search_generated_response: str = None
    web_search_generated_response_list: List[str] = field(default_factory=list)
    web_search_extracted_answer: str = None
    web_search_extracted_answer_list: List[str] = field(default_factory=list)
    web_search_is_correct: bool = False
    
    # Cross-agent alignment
    reasoning_web_search_aligned: bool = False
    
    # Final aggregated answer
    final_answer: str = None
    final_is_correct: bool = False

class SearchEnv(Env):
    """
    Environment for search-based question answering tasks with dual-agent interaction.
    
    This environment coordinates between reasoning and web search agents to answer
    questions that may require external knowledge retrieval.
    """

    def __init__(
        self, 
        env_idx: int,
        rollout_idx: int,
        max_turns: int,
        config: dict | None = None,
    ):
        """
        Initialize the search environment.
        """
        super().__init__(env_idx=env_idx, rollout_idx=rollout_idx, config=config)
        self.state = SearchEnvState()

    def reset(self):
        """Reset the search environment state."""
        # Reset task decomposition state
        self.state.sub_questions = []
        self.state.sub_answers = []
        self.state.current_sub_question_index = 0
        self.state.decomposition_complete = False
        
        self.state.reasoning_generated_response = None
        self.state.reasoning_generated_response_list = []
        self.state.reasoning_extracted_answer = None
        self.state.reasoning_extracted_answer_list = []
        self.state.reasoning_is_correct = False
        
        self.state.web_search_queries = []
        self.state.web_search_results = []
        self.state.web_search_formatted_results = ""
        self.state.web_search_generated_response = None
        self.state.web_search_generated_response_list = []
        self.state.web_search_extracted_answer = None
        self.state.web_search_extracted_answer_list = []
        self.state.web_search_is_correct = False
        
        self.state.reasoning_web_search_aligned = False
        self.state.final_answer = None
        self.state.final_is_correct = False

    async def step(self, role: str, action: str, env_worker: Any = None):
        """
        Execute an action in the search environment.
        
        Args:
            role: Agent role ("reasoning_agent" or "web_search_agent")
            action: Action to take in the environment
            env_worker: Optional environment worker for parallel execution
        """
        if role == "reasoning_agent":
            await self._reasoning_step(action, env_worker)
        elif role == "web_search_agent":
            await self._web_search_step(action, env_worker)
        else:
            raise ValueError(f"Invalid role: {role}")

    async def _reasoning_step(self, action: str, env_worker: Any = None):
        """
        Execute a reasoning agent step.
        
        Args:
            action: Reasoning response from the agent
            env_worker: Optional environment worker
        """
        # Store the reasoning response
        self.state.reasoning_generated_response = action
        self.state.reasoning_generated_response_list.append(action)
        
        # Extract answer from reasoning response
        from pettingllms.multi_agent_env.search.search_utils import extract_answer_from_response
        extracted_answer = extract_answer_from_response(action)
        self.state.reasoning_extracted_answer = extracted_answer
        self.state.reasoning_extracted_answer_list.append(extracted_answer or "No answer found")
        
        # Evaluate correctness
        if extracted_answer and self.state.ground_truth_answer:
            from pettingllms.multi_agent_env.search.search_utils import evaluate_search_answer
            self.state.reasoning_is_correct = evaluate_search_answer(
                extracted_answer, self.state.ground_truth_answer
            )

    async def _web_search_step(self, action: str, env_worker: Any = None):
        """
        Execute a web search agent step.
        
        Args:
            action: Search query or response from the agent
            env_worker: Optional environment worker
        """
        # Extract search query
        from pettingllms.multi_agent_env.search.search_utils import extract_search_query_from_response
        search_query = extract_search_query_from_response(action)
        
        if search_query:
            # Store the search query
            self.state.web_search_queries.append(search_query)
            
            # Perform web search (simulated for now)
            search_results = simulate_web_search(search_query, max_results=5)
            self.state.web_search_results.extend(search_results)
            
            # Format search results for display
            formatted_results = format_search_results(search_results)
            self.state.web_search_formatted_results = formatted_results
        
        # Store the full response
        self.state.web_search_generated_response = action
        self.state.web_search_generated_response_list.append(action)
        
        # Extract answer from web search response
        from pettingllms.multi_agent_env.search.search_utils import extract_answer_from_response
        extracted_answer = extract_answer_from_response(action)
        self.state.web_search_extracted_answer = extracted_answer
        self.state.web_search_extracted_answer_list.append(extracted_answer or "No answer found")
        
        # Evaluate correctness
        if extracted_answer and self.state.ground_truth_answer:
            from pettingllms.multi_agent_env.search.search_utils import evaluate_search_answer
            self.state.web_search_is_correct = evaluate_search_answer(
                extracted_answer, self.state.ground_truth_answer
            )

    def render(self, mode=None):
        """Render the environment state."""
        return f"Question: {self.state.question}\nGround Truth: {self.state.ground_truth_answer}"

    def close(self):
        """Close the environment."""
        pass

class SearchEnvBatch:
    """Batch environment manager for Search environments."""
    
    def __init__(
        self, 
        env_idx_list: List[int], 
        env_indices: List[int],
        rollout_idx_list: List[int], 
        samples: int, 
        max_turns: int, 
        config: dict, 
        mode: str = "train", 
        *, 
        env_workers: List = None
    ):
        """
        Initialize batch of Search environments.
        
        Args:
            env_idx_list: List of environment indices
            env_indices: List of environment indices for problem loading
            rollout_idx_list: List of rollout indices
            samples: Number of samples per environment
            max_turns: Maximum turns per environment
            config: Configuration dictionary
            mode: Environment mode ("train", "val", "test")
            env_workers: Optional list of environment workers
        """
        self.mode = mode
        self.env_list = []
        
        # Convert env_indices to list for safety
        safe_env_indices = list(env_indices) if not isinstance(env_indices, list) else env_indices
        
        # Load search problems
        # Default benchmark for test mode is "bamboogle", but ignored for train mode
        benchmark_name = getattr(config, "benchmark", "bamboogle") if hasattr(config, "benchmark") else "bamboogle"
        self.problem_list = load_search_problem_batch(safe_env_indices, dataset_name=benchmark_name, mode=mode, config=config)
        
        if mode == "validate":
            rollout_idx_list = range(len(self.problem_list) * samples)
            samples = 1
        
        if not self.problem_list:
            raise ValueError(f"Failed to load problems from search dataset. Please check if the dataset is available and accessible.")

        # Create environments
        for i, problem in enumerate(self.problem_list):
            state = SearchEnvState(
                question=problem["question"],
                ground_truth_answer=problem["ground_truth"],
                search_hint=problem.get("search_hint", "")
            )
            
            for s in range(samples):
                env = SearchEnv(
                    env_idx=i, 
                    rollout_idx=rollout_idx_list[i * samples + s], 
                    max_turns=max_turns, 
                    config=None
                )
                env.state = copy.deepcopy(state)
                self.env_list.append(env)
        
        if len(self.env_list) != len(rollout_idx_list):
            raise ValueError(
                f"len(self.env_list)!=len(rollout_idx_list), {len(self.env_list)}!={len(rollout_idx_list)}"
            )