"""Ensemble node that combines multiple agents' outputs through voting or consensus."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any, Callable, Optional
from collections import Counter
import json
from workflow.core import WorkflowNode, Context, Message, MessageType


class EnsembleNode(WorkflowNode):
    """Ensemble node that runs multiple agents and combines their outputs.
    
    Supports multiple aggregation strategies:
    - majority_vote: Select the most common response
    - weighted_vote: Weight responses by confidence scores
    - consensus: Use another agent to synthesize responses
    """
    
    def __init__(
        self,
        name: str,
        agents: List[WorkflowNode],
        strategy: str = "majority_vote",
        consensus_agent: Optional[WorkflowNode] = None,
        weights: Optional[List[float]] = None,
        **kwargs
    ):
        """Initialize the ensemble node.
        
        Args:
            name: Node name
            agents: List of agent nodes to ensemble
            strategy: Aggregation strategy ('majority_vote', 'weighted_vote', 'consensus')
            consensus_agent: Agent to use for consensus strategy
            weights: Weights for each agent (for weighted_vote strategy)
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        
        self.agents = agents
        self.strategy = strategy
        self.consensus_agent = consensus_agent
        self.weights = weights or [1.0] * len(agents)
        
        if len(self.weights) != len(self.agents):
            raise ValueError("Number of weights must match number of agents")
        
        if strategy == "consensus" and consensus_agent is None:
            raise ValueError("consensus strategy requires a consensus_agent")
    
    def _majority_vote(self, responses: List[str]) -> str:
        """Select the most common response.
        
        Args:
            responses: List of agent responses
            
        Returns:
            Most common response
        """
        # Simple string matching for exact duplicates
        counter = Counter(responses)
        most_common = counter.most_common(1)[0][0]
        
        self.logger.info(f"Majority vote result: {most_common} (appeared {counter[most_common]} times)")
        return most_common
    
    def _weighted_vote(self, responses: List[str], weights: List[float]) -> str:
        """Select response based on weighted voting.
        
        Args:
            responses: List of agent responses
            weights: Weight for each response
            
        Returns:
            Response with highest weighted score
        """
        # Group responses and sum weights
        response_weights = {}
        for response, weight in zip(responses, weights):
            if response in response_weights:
                response_weights[response] += weight
            else:
                response_weights[response] = weight
        
        # Select response with highest weight
        best_response = max(response_weights.items(), key=lambda x: x[1])
        
        self.logger.info(f"Weighted vote result: {best_response[0]} (weight: {best_response[1]})")
        return best_response[0]
    
    def _consensus(self, context: Context, responses: List[str]) -> str:
        """Use consensus agent to synthesize responses.
        
        Args:
            context: Workflow context
            responses: List of agent responses
            
        Returns:
            Synthesized consensus response
        """
        # Create a new context for the consensus agent
        consensus_context = Context()
        consensus_context.state = context.state.copy()
        
        # Add all responses as input
        responses_text = "\n\n".join([
            f"Agent {i+1} response:\n{resp}"
            for i, resp in enumerate(responses)
        ])
        
        consensus_input = Message(
            content=f"Please synthesize the following responses into a single, coherent answer:\n\n{responses_text}",
            message_type=MessageType.USER_INPUT
        )
        consensus_context.add_message(consensus_input)
        
        # Run consensus agent
        result = self.consensus_agent(consensus_context)
        
        self.logger.info(f"Consensus result: {result.content}")
        return result.content
    
    def process(self, context: Context) -> Message:
        """Process context by running all agents and combining their outputs.
        
        Args:
            context: Workflow context
            
        Returns:
            Message containing ensemble result
        """
        self.logger.info(f"Running ensemble with {len(self.agents)} agents using {self.strategy} strategy")
        
        # Run all agents
        responses = []
        for i, agent in enumerate(self.agents):
            self.logger.info(f"Running agent {i+1}/{len(self.agents)}: {agent.name}")
            
            # Create a copy of context for each agent to avoid interference
            agent_context = Context()
            agent_context.messages = context.messages.copy()
            agent_context.state = context.state.copy()
            
            # Run agent
            result = agent(agent_context)
            
            if result.message_type == MessageType.ERROR:
                self.logger.warning(f"Agent {agent.name} returned error: {result.content}")
                continue
            
            responses.append(result.content)
        
        if not responses:
            return Message(
                content={"error": "All agents failed"},
                message_type=MessageType.ERROR
            )
        
        # Combine responses based on strategy
        if self.strategy == "majority_vote":
            final_response = self._majority_vote(responses)
        elif self.strategy == "weighted_vote":
            final_response = self._weighted_vote(responses, self.weights)
        elif self.strategy == "consensus":
            final_response = self._consensus(context, responses)
        else:
            raise ValueError(f"Unknown ensemble strategy: {self.strategy}")
        
        return Message(
            content=final_response,
            message_type=MessageType.AGENT_RESPONSE,
            metadata={
                "strategy": self.strategy,
                "num_agents": len(self.agents),
                "all_responses": responses
            }
        )

