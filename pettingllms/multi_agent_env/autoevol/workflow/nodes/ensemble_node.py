"""Ensemble node that combines multiple agents' outputs through voting or consensus."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import re
from typing import List, Dict, Any, Callable, Optional
from collections import Counter
import json
from workflow.core import WorkflowNode, Context, Message, MessageType


def _extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} in a response string.

    Handles nested braces like \\boxed{\\frac{1}{2}}.

    Args:
        text: Agent response text

    Returns:
        Extracted answer string, or None if not found
    """
    # Find the last \boxed{ occurrence (agents may have multiple attempts)
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i - 1].strip()
    return None


def _normalize_answer(answer: str) -> str:
    """Normalize a math answer string for comparison.

    Converts LaTeX fractions, removes whitespace, and tries to evaluate
    to a canonical numerical form so that e.g. "\\frac{3}{2}" and "1.5"
    are treated as equivalent.

    Args:
        answer: Raw extracted answer string

    Returns:
        Normalized string for comparison
    """
    s = answer.strip()

    # Remove surrounding $ signs and whitespace
    s = s.strip("$ ")

    # Try direct float parse first (handles "42", "1.5", "-3", etc.)
    try:
        val = float(s)
        # Return canonical form: integer if whole, else float
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        pass

    # Handle \\frac{a}{b} -> a/b
    frac_match = re.match(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                val = num / den
                if val == int(val):
                    return str(int(val))
                return str(val)
        except (ValueError, ZeroDivisionError):
            pass

    # Handle plain "a/b" fraction notation
    if "/" in s and "\\" not in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                val = float(parts[0].strip()) / float(parts[1].strip())
                if val == int(val):
                    return str(int(val))
                return str(val)
            except (ValueError, ZeroDivisionError):
                pass

    # Strip common LaTeX formatting for textual comparison
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace(" ", "")

    return s


class EnsembleNode(WorkflowNode):
    """Ensemble node that runs multiple agents and combines their outputs.

    Supports multiple aggregation strategies:
    - majority_vote: Extract \\boxed{} answers and vote.  When a *consensus_agent*
      is provided, unanimous votes are accepted directly while split votes are
      escalated to the judge for a reasoned decision.
    - weighted_vote: Weight responses by confidence scores
    - consensus: Always use a judge agent to synthesize responses (with original question)
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
            consensus_agent: Agent to use for consensus / judge-on-split.
                For 'consensus' strategy this is required.
                For 'majority_vote' this is optional — when provided, split
                votes are escalated to the judge instead of picking the plurality.
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

    def _majority_vote(self, context: Context, responses: List[str]) -> str:
        """Select response by extracting \\boxed{} answers and voting on values.

        Answers are normalized before comparison so that e.g. "\\frac{3}{2}"
        and "1.5" count as the same vote.

        When a *consensus_agent* is available:
        - Unanimous agreement -> accept directly (fast path)
        - Split votes -> escalate to judge for a reasoned decision

        Falls back to first response if no boxed answers are found.

        Args:
            context: Workflow context (for judge escalation)
            responses: List of agent responses

        Returns:
            The full response whose extracted answer is most common,
            or the judge's response when escalated.
        """
        # Extract boxed answers from each response
        extracted = []
        for resp in responses:
            ans = _extract_boxed_answer(resp)
            extracted.append(ans)

        # Filter out None values and normalize for voting
        valid_triples = []
        for resp, ans in zip(responses, extracted):
            if ans is not None:
                valid_triples.append((resp, ans, _normalize_answer(ans)))

        if not valid_triples:
            self.logger.warning("No \\boxed{} answers found, falling back to first response")
            return responses[0]

        # Vote on normalized answer values
        answer_counter = Counter(norm for _, _, norm in valid_triples)
        best_norm = answer_counter.most_common(1)[0][0]
        best_count = answer_counter[best_norm]
        num_distinct = len(answer_counter)

        self.logger.info(
            f"Answer vote: {best_norm} (appeared {best_count}/{len(valid_triples)} times) | "
            f"Raw extracted: {[ans for _, ans, _ in valid_triples]} | "
            f"Normalized: {[norm for _, _, norm in valid_triples]}"
        )

        # --- Judge escalation on split votes ---
        is_unanimous = (num_distinct == 1)
        if not is_unanimous and self.consensus_agent is not None:
            self.logger.info(
                f"Votes split ({num_distinct} distinct answers) — escalating to judge"
            )
            return self._consensus(context, responses)

        # Return the full response that produced the winning answer
        for resp, _, norm in valid_triples:
            if norm == best_norm:
                return resp
        return responses[0]

    def _weighted_vote(self, responses: List[str], weights: List[float]) -> str:
        """Select response based on weighted voting on extracted \\boxed{} answers.

        Answers are normalized before comparison.

        Args:
            responses: List of agent responses
            weights: Weight for each response

        Returns:
            Response with highest weighted score
        """
        # Extract boxed answers
        extracted = []
        for resp in responses:
            ans = _extract_boxed_answer(resp)
            extracted.append(ans)

        valid_quads = [
            (resp, ans, _normalize_answer(ans), w)
            for resp, ans, w in zip(responses, extracted, weights)
            if ans is not None
        ]

        if not valid_quads:
            self.logger.warning("No \\boxed{} answers found, falling back to highest-weight response")
            best_idx = weights.index(max(weights))
            return responses[best_idx]

        # Group by normalized answer value and sum weights
        answer_weights: Dict[str, float] = {}
        for _, _, norm, w in valid_quads:
            answer_weights[norm] = answer_weights.get(norm, 0.0) + w

        best_norm = max(answer_weights, key=answer_weights.get)

        self.logger.info(f"Weighted vote: {best_norm} (weight: {answer_weights[best_norm]})")

        for resp, _, norm, _ in valid_quads:
            if norm == best_norm:
                return resp
        return responses[0]

    def _consensus(self, context: Context, responses: List[str]) -> str:
        """Use consensus agent to synthesize responses.

        The consensus agent receives the original question together with
        all agent responses so it can make an informed judgement.

        Args:
            context: Workflow context (used to retrieve original question)
            responses: List of agent responses

        Returns:
            Synthesized consensus response
        """
        # Create a new context for the consensus agent
        consensus_context = Context()
        consensus_context.state = context.state.copy()

        # Retrieve original question from context
        original_question = ""
        for msg in context.messages:
            if msg.message_type == MessageType.USER_INPUT:
                original_question = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Build prompt with original question + all responses
        responses_text = "\n\n".join([
            f"--- Solution {i+1} ({self.agents[i].name}) ---\n{resp}"
            for i, resp in enumerate(responses)
        ])

        consensus_input = Message(
            content=(
                f"**Original Question:**\n{original_question}\n\n"
                f"**Multiple Solutions:**\n\n{responses_text}\n\n"
                f"Review each solution carefully. Check the reasoning and calculations. "
                f"Select the best answer or synthesize a corrected answer. "
                f"Your final answer MUST be in \\boxed{{}} format."
            ),
            message_type=MessageType.USER_INPUT
        )
        consensus_context.add_message(consensus_input)

        # Run consensus agent
        result = self.consensus_agent(consensus_context)

        self.logger.info(f"Consensus result: {result.content[:200]}...")
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
            final_response = self._majority_vote(context, responses)
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
