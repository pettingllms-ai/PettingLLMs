"""Debate node that facilitates multi-agent debate and refinement."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Optional
from workflow.core import WorkflowNode, Context, Message, MessageType


class DebateNode(WorkflowNode):
    """Debate node where multiple agents debate to refine answers.
    
    Process:
    1. Each agent provides initial response
    2. Agents see each other's responses and critique/refine
    3. Repeat for N rounds
    4. Judge agent selects best final answer
    """
    
    def __init__(
        self,
        name: str,
        debaters: List[WorkflowNode],
        judge: Optional[WorkflowNode] = None,
        num_rounds: int = 2,
        **kwargs
    ):
        """Initialize the debate node.
        
        Args:
            name: Node name
            debaters: List of agent nodes that will debate
            judge: Optional judge agent to select best answer
            num_rounds: Number of debate rounds
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        
        self.debaters = debaters
        self.judge = judge or debaters[0]  # Use first debater as judge if not specified
        self.num_rounds = num_rounds
    
    def _format_debate_history(self, debate_history: List[List[str]], current_round: int) -> str:
        """Format debate history for agents to review.
        
        Args:
            debate_history: List of responses per round per agent
            current_round: Current debate round
            
        Returns:
            Formatted debate history string
        """
        history_text = "Previous debate rounds:\n\n"
        
        for round_idx in range(current_round):
            history_text += f"=== Round {round_idx + 1} ===\n"
            for agent_idx, response in enumerate(debate_history[round_idx]):
                history_text += f"Agent {agent_idx + 1} ({self.debaters[agent_idx].name}):\n{response}\n\n"
        
        return history_text
    
    def process(self, context: Context) -> Message:
        """Process context through multi-round debate.
        
        Args:
            context: Workflow context
            
        Returns:
            Message containing final debate result
        """
        self.logger.info(f"Starting debate with {len(self.debaters)} debaters for {self.num_rounds} rounds")
        
        # Store debate history: debate_history[round][agent_idx] = response
        debate_history = []
        
        # Get original question
        original_input = context.get_latest_message()
        if not original_input:
            return Message(
                content={"error": "No input message found"},
                message_type=MessageType.ERROR
            )
        
        # Conduct debate rounds
        for round_idx in range(self.num_rounds):
            self.logger.info(f"Debate round {round_idx + 1}/{self.num_rounds}")
            
            round_responses = []
            
            for agent_idx, agent in enumerate(self.debaters):
                # Create context for this agent
                agent_context = Context()
                agent_context.state = context.state.copy()
                
                # Add original question
                if round_idx == 0:
                    # First round: just answer the question
                    prompt = original_input.content
                else:
                    # Later rounds: show debate history and ask to refine
                    history = self._format_debate_history(debate_history, round_idx)
                    prompt = (
                        f"Original question: {original_input.content}\n\n"
                        f"{history}\n"
                        f"Based on the discussion above, please provide your refined answer. "
                        f"You may critique other responses and improve your own."
                    )
                
                agent_message = Message(
                    content=prompt,
                    message_type=MessageType.USER_INPUT
                )
                agent_context.add_message(agent_message)
                
                # Run agent
                result = agent(agent_context)
                
                if result.message_type == MessageType.ERROR:
                    self.logger.warning(f"Agent {agent.name} returned error: {result.content}")
                    round_responses.append(f"[Error: {result.content}]")
                else:
                    round_responses.append(result.content)
                
                self.logger.info(f"Agent {agent_idx + 1} ({agent.name}) completed round {round_idx + 1}")
            
            debate_history.append(round_responses)
        
        # Judge selects best answer
        self.logger.info("Judge selecting final answer")
        
        judge_context = Context()
        judge_context.state = context.state.copy()
        
        # Format all final round responses for judge
        final_responses = debate_history[-1]
        responses_text = "\n\n".join([
            f"Response {i+1} from {self.debaters[i].name}:\n{resp}"
            for i, resp in enumerate(final_responses)
        ])
        
        judge_prompt = (
            f"Original question: {original_input.content}\n\n"
            f"After {self.num_rounds} rounds of debate, here are the final responses:\n\n"
            f"{responses_text}\n\n"
            f"Please select the best response or synthesize a better answer based on the debate. "
            f"Provide your final answer."
        )
        
        judge_message = Message(
            content=judge_prompt,
            message_type=MessageType.USER_INPUT
        )
        judge_context.add_message(judge_message)
        
        # Get judge's decision
        judge_result = self.judge(judge_context)
        
        return Message(
            content=judge_result.content,
            message_type=MessageType.AGENT_RESPONSE,
            metadata={
                "num_rounds": self.num_rounds,
                "num_debaters": len(self.debaters),
                "debate_history": debate_history
            }
        )

