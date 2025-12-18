"""
Ensemble Pattern

Multi-agent voting and consensus mechanisms for AG2.
Supports three strategies: majority_vote, weighted_vote, and consensus.
"""

from typing import List, Dict, Optional, Callable
from collections import Counter
from enum import Enum
from autogen import ConversableAgent, GroupChat, GroupChatManager


class VotingStrategy(Enum):
    """Voting strategies for ensemble."""
    MAJORITY = "majority_vote"
    WEIGHTED = "weighted_vote"
    CONSENSUS = "consensus"


def majority_vote(responses: List[str]) -> str:
    """
    Select the most common response (simple majority voting).

    Args:
        responses: List of agent responses

    Returns:
        Most common response

    Example:
        >>> responses = ["Paris", "Paris", "London"]
        >>> majority_vote(responses)
        'Paris'
    """
    if not responses:
        return ""

    counter = Counter(responses)
    most_common = counter.most_common(1)[0][0]

    print(f"[Majority Vote] Winner: {most_common[:50]}... (appeared {counter[most_common]} times)")
    return most_common


def weighted_vote(responses: List[str], weights: List[float]) -> str:
    """
    Select response based on weighted voting.

    Args:
        responses: List of agent responses
        weights: Weight for each response (must match length)

    Returns:
        Response with highest weighted score

    Example:
        >>> responses = ["A", "B", "A"]
        >>> weights = [0.8, 0.5, 0.7]
        >>> weighted_vote(responses, weights)
        'A'  # Score: 0.8 + 0.7 = 1.5
    """
    if not responses or len(responses) != len(weights):
        return responses[0] if responses else ""

    # Accumulate weights for each unique response
    response_weights = {}
    for response, weight in zip(responses, weights):
        response_weights[response] = response_weights.get(response, 0) + weight

    # Select response with highest total weight
    winner = max(response_weights.items(), key=lambda x: x[1])

    print(f"[Weighted Vote] Winner: {winner[0][:50]}... (score: {winner[1]:.2f})")
    return winner[0]


def consensus_vote(
    responses: List[str],
    consensus_agent: ConversableAgent,
    original_question: str
) -> str:
    """
    Use a consensus agent to synthesize all responses.

    Args:
        responses: List of agent responses
        consensus_agent: Agent to synthesize responses
        original_question: Original question for context

    Returns:
        Synthesized consensus response

    Example:
        >>> consensus_vote(responses, judge_agent, "What is the capital?")
    """
    if not responses:
        return ""

    # Format all responses
    formatted_responses = "\n\n".join([
        f"Agent {i+1} Response:\n{resp}"
        for i, resp in enumerate(responses)
    ])

    # Create synthesis prompt
    synthesis_prompt = f"""You are a judge synthesizing multiple expert responses.

Original Question: {original_question}

Expert Responses:
{formatted_responses}

Please synthesize these responses into a single, comprehensive answer.
Consider the commonalities, resolve conflicts, and provide the best answer."""

    # Get consensus
    consensus_response = consensus_agent.generate_reply(
        messages=[{"role": "user", "content": synthesis_prompt}]
    )

    print(f"[Consensus Vote] Synthesized response from {len(responses)} agents")
    return consensus_response


def create_ensemble_groupchat(
    agents: List[ConversableAgent],
    strategy: str = "majority_vote",
    weights: Optional[List[float]] = None,
    consensus_agent: Optional[ConversableAgent] = None,
    max_round: int = 20,
) -> GroupChat:
    """
    Create an ensemble GroupChat with voting mechanism.

    This function creates a GroupChat where multiple agents provide responses,
    and the final result is determined by voting/consensus.

    Args:
        agents: List of agents to participate in ensemble
        strategy: Voting strategy ("majority_vote", "weighted_vote", "consensus")
        weights: Weights for each agent (required for weighted_vote)
        consensus_agent: Agent for synthesis (required for consensus)
        max_round: Maximum rounds (default: agents + 2 for voting)

    Returns:
        Configured GroupChat instance

    Example:
        >>> agents = [agent1, agent2, agent3]
        >>> groupchat = create_ensemble_groupchat(agents, strategy="majority_vote")
        >>> manager = GroupChatManager(groupchat=groupchat)
        >>> result = agents[0].initiate_chat(manager, message="Question")

    Raises:
        ValueError: If invalid strategy or missing required parameters
    """
    # Validate inputs
    if strategy not in ["majority_vote", "weighted_vote", "consensus"]:
        raise ValueError(f"Invalid strategy: {strategy}")

    if strategy == "weighted_vote" and (not weights or len(weights) != len(agents)):
        raise ValueError("weighted_vote requires weights list matching agents length")

    if strategy == "consensus" and consensus_agent is None:
        raise ValueError("consensus strategy requires consensus_agent")

    num_agents = len(agents)

    # Store responses for voting
    agent_responses = []

    def ensemble_speaker_selection(last_speaker: ConversableAgent, groupchat: GroupChat):
        """Custom speaker selection for ensemble voting."""
        nonlocal agent_responses

        # Count how many agents have responded
        assistant_messages = [
            m for m in groupchat.messages
            if m.get("role") == "assistant" and m.get("name") in [a.name for a in agents]
        ]

        num_responses = len(assistant_messages)

        # Phase 1: Collect responses from all agents (round-robin)
        if num_responses < num_agents:
            # Get next agent in round-robin fashion
            current_index = agents.index(last_speaker) if last_speaker in agents else -1
            next_index = (current_index + 1) % num_agents
            next_agent = agents[next_index]

            print(f"[Ensemble] Collecting response {num_responses + 1}/{num_agents} from {next_agent.name}")
            return next_agent

        # Phase 2: All agents have responded, perform voting
        print(f"\n[Ensemble] All {num_agents} agents have responded. Starting voting...")

        # Extract responses
        agent_responses = [m["content"] for m in assistant_messages[-num_agents:]]

        # Perform voting based on strategy
        if strategy == "majority_vote":
            final_answer = majority_vote(agent_responses)

        elif strategy == "weighted_vote":
            final_answer = weighted_vote(agent_responses, weights)

        elif strategy == "consensus":
            # Get original question
            original_q = groupchat.messages[0]["content"] if groupchat.messages else ""
            final_answer = consensus_vote(agent_responses, consensus_agent, original_q)

        # Store final answer in groupchat for retrieval
        groupchat.messages.append({
            "role": "assistant",
            "name": "EnsembleResult",
            "content": final_answer
        })

        print(f"[Ensemble] Voting complete. Final answer ready.\n")

        # Return None to terminate
        return None

    # Create GroupChat
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=max_round,
        speaker_selection_method=ensemble_speaker_selection,
    )

    return groupchat


class EnsembleResult:
    """Container for ensemble voting results."""

    def __init__(
        self,
        final_answer: str,
        individual_responses: List[str],
        strategy: str,
        num_agents: int
    ):
        self.final_answer = final_answer
        self.individual_responses = individual_responses
        self.strategy = strategy
        self.num_agents = num_agents

    def __str__(self):
        return f"EnsembleResult(strategy={self.strategy}, agents={self.num_agents})"

    def summary(self) -> str:
        """Get a summary of the ensemble result."""
        summary = f"""Ensemble Voting Summary
Strategy: {self.strategy}
Number of Agents: {self.num_agents}

Individual Responses:
"""
        for i, resp in enumerate(self.individual_responses, 1):
            summary += f"\nAgent {i}:\n{resp[:100]}...\n"

        summary += f"\nFinal Answer:\n{self.final_answer}\n"
        return summary
