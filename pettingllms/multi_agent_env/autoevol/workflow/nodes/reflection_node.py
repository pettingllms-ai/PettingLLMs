"""Reflection node that enables agents to self-critique and refine outputs."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Optional, List, Dict, Any
from workflow.core import WorkflowNode, Context, Message, MessageType


def _text_similarity(a: str, b: str) -> float:
    """Compute a cheap similarity ratio between two strings.

    Uses the ratio of shared character n-grams (trigrams) to detect
    when the solver is repeating essentially the same response.
    """
    if not a or not b:
        return 0.0
    n = 3
    grams_a = set(a[i:i+n] for i in range(len(a) - n + 1))
    grams_b = set(b[i:i+n] for i in range(len(b) - n + 1))
    if not grams_a or not grams_b:
        return 0.0
    intersection = grams_a & grams_b
    union = grams_a | grams_b
    return len(intersection) / len(union)


class ReflectionNode(WorkflowNode):
    """Reflection node that implements self-refinement through reflection.

    Process:
    1. Agent generates initial response
    2. Critic agent (or same agent) reflects on response and identifies issues
    3. Agent refines response based on reflection
    4. Repeat for N iterations

    Supports two modes:
    - Self-reflection: Same agent does generation, reflection, and refinement
    - Critic-based: Separate critic_agent provides external critique
    """

    def __init__(
        self,
        name: str,
        agent: WorkflowNode,
        num_iterations: int = 2,
        reflection_prompt: Optional[str] = None,
        critic_agent: Optional[WorkflowNode] = None,
        refinement_prompt: Optional[str] = None,
        include_history: bool = False,
        **kwargs
    ):
        """Initialize the reflection node.

        Args:
            name: Node name
            agent: Agent node to use for generation and refinement
            num_iterations: Number of reflection iterations
            reflection_prompt: Custom reflection prompt template
            critic_agent: Optional separate agent for reflection/critique.
                         If not provided, uses the main agent (self-reflection mode).
            refinement_prompt: Custom refinement prompt template
            include_history: If True, include all previous iterations in refinement prompt
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.agent = agent
        self.num_iterations = num_iterations
        self.include_history = include_history

        # Critic agent: use provided one or fall back to main agent (backward compatible)
        self.critic_agent = critic_agent if critic_agent is not None else agent

        # Reflection prompt template
        self.reflection_prompt = reflection_prompt or (
            "You are a critical reviewer. Analyze the following response to a question.\n\n"
            "Original question: {question}\n\n"
            "Response to review:\n{response}\n\n"
            "Provide a thorough critique:\n"
            "1. Identify any errors, inaccuracies, or logical flaws\n"
            "2. Check if the response is complete and addresses all aspects of the question\n"
            "3. Evaluate the clarity and structure of the response\n"
            "4. Suggest specific improvements\n\n"
            "Be constructive but rigorous in your critique."
        )

        # Refinement prompt template
        self.refinement_prompt = refinement_prompt or (
            "Original question: {question}\n\n"
            "Your previous response:\n{response}\n\n"
            "Critique from reviewer:\n{reflection}\n\n"
            "Based on this critique, provide an improved response that addresses all the issues raised."
        )

        # Extended refinement prompt with history
        self.refinement_prompt_with_history = (
            "Original question: {question}\n\n"
            "=== Iteration History ===\n{history}\n\n"
            "=== Current Iteration ===\n"
            "Your latest response:\n{response}\n\n"
            "Latest critique:\n{reflection}\n\n"
            "Based on ALL previous feedback, provide a significantly improved response."
        )

    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format iteration history for inclusion in prompt.

        Args:
            history: List of iteration records

        Returns:
            Formatted history string
        """
        if not history:
            return "No previous iterations."

        formatted = []
        for record in history:
            iteration = record.get("iteration", 0)
            response = record.get("response", "")
            reflection = record.get("reflection")

            formatted.append(f"--- Iteration {iteration} ---")
            formatted.append(f"Response: {response}")
            if reflection:
                formatted.append(f"Critique: {reflection}")
            formatted.append("")

        return "\n".join(formatted)

    def _reflect(self, context: Context, question: str, response: str, iteration: int = 0) -> str:
        """Generate reflection/critique on a response.

        Args:
            context: Workflow context
            question: Original question
            response: Response to reflect on
            iteration: Current iteration number (for logging)

        Returns:
            Reflection/critique text
        """
        reflection_context = Context()
        reflection_context.state = context.state.copy()

        # Include question in reflection prompt for better context
        reflection_input = Message(
            content=self.reflection_prompt.format(question=question, response=response),
            message_type=MessageType.USER_INPUT
        )
        reflection_context.add_message(reflection_input)

        # Use critic_agent for reflection
        self.logger.info(f"Iteration {iteration + 1}: Generating critique using {self.critic_agent.name}")
        result = self.critic_agent(reflection_context)

        if result.message_type == MessageType.ERROR:
            self.logger.warning(f"Reflection failed: {result.content}")
            return "Unable to generate reflection due to error."

        return result.content

    def _refine(
        self,
        context: Context,
        question: str,
        response: str,
        reflection: str,
        history: Optional[List[Dict[str, Any]]] = None,
        iteration: int = 0
    ) -> str:
        """Refine response based on reflection.

        Args:
            context: Workflow context
            question: Original question
            response: Previous response
            reflection: Reflection/critique on the response
            history: Optional history of previous iterations
            iteration: Current iteration number (for logging)

        Returns:
            Refined response
        """
        refinement_context = Context()
        refinement_context.state = context.state.copy()

        # Choose prompt based on whether to include history
        if self.include_history and history:
            prompt = self.refinement_prompt_with_history.format(
                question=question,
                response=response,
                reflection=reflection,
                history=self._format_history(history)
            )
        else:
            prompt = self.refinement_prompt.format(
                question=question,
                response=response,
                reflection=reflection
            )

        refinement_input = Message(
            content=prompt,
            message_type=MessageType.USER_INPUT
        )
        refinement_context.add_message(refinement_input)

        # Use main agent for refinement
        self.logger.info(f"Iteration {iteration + 1}: Refining response using {self.agent.name}")
        result = self.agent(refinement_context)

        if result.message_type == MessageType.ERROR:
            self.logger.warning(f"Refinement failed: {result.content}")
            return response  # Return previous response if refinement fails

        return result.content

    def process(self, context: Context) -> Message:
        """Process context through reflection and refinement.

        Args:
            context: Workflow context

        Returns:
            Message containing refined response
        """
        # Determine mode for logging
        mode = "critic-based" if self.critic_agent != self.agent else "self-reflection"
        self.logger.info(f"Starting reflection ({mode} mode) with {self.num_iterations} iterations")

        # Get original question – prefer the first USER_INPUT (which is
        # the original question) rather than the latest message, so the
        # verifier always sees the real question even when ReflectionNode
        # is downstream of other nodes.
        original_input = None
        for msg in context.messages:
            if msg.message_type == MessageType.USER_INPUT:
                original_input = msg
                break
        if original_input is None:
            original_input = context.get_latest_message()
        if not original_input:
            return Message(
                content={"error": "No input message found"},
                message_type=MessageType.ERROR
            )

        question = original_input.content if isinstance(original_input.content, str) else str(original_input.content)

        # Store original question in context state for potential sharing
        context.set_state("original_question", question)

        # Generate initial response
        self.logger.info(f"Generating initial response using {self.agent.name}")
        initial_context = Context()
        initial_context.state = context.state.copy()
        initial_context.add_message(original_input)

        initial_result = self.agent(initial_context)

        if initial_result.message_type == MessageType.ERROR:
            return initial_result

        current_response = initial_result.content

        # Store history for metadata and optional inclusion in prompts
        history = [{
            "iteration": 0,
            "response": current_response,
            "reflection": None
        }]

        # Reflection iterations (with early termination)
        for iteration in range(self.num_iterations):
            self.logger.info(f"=== Reflection iteration {iteration + 1}/{self.num_iterations} ===")

            # Reflect on current response (using critic_agent)
            reflection = self._reflect(context, question, current_response, iteration)

            self.logger.info(f"Generated critique:\n{reflection}")

            # Early termination: if the verifier says the answer is correct,
            # skip refinement to avoid the solver second-guessing a good answer.
            reflection_upper = reflection.upper()
            if ("FINAL VERDICT: CORRECT" in reflection_upper
                    or "VERDICT: CORRECT" in reflection_upper
                    or reflection_upper.strip().endswith("CORRECT")):
                self.logger.info(
                    f"Verifier confirmed CORRECT at iteration {iteration + 1} — "
                    "stopping early to preserve answer."
                )
                history.append({
                    "iteration": iteration + 1,
                    "response": current_response,
                    "reflection": reflection,
                    "early_stop": True
                })
                break

            # Refine based on reflection (using main agent)
            refined_response = self._refine(
                context,
                question,
                current_response,
                reflection,
                history if self.include_history else None,
                iteration
            )

            self.logger.info(f"Generated refined response:\n{refined_response}")

            # Repetition detection: if the solver is repeating itself, stop early
            similarity = _text_similarity(current_response, refined_response)
            if similarity > 0.75:
                self.logger.info(
                    f"Repetition detected (similarity={similarity:.2f}) at iteration "
                    f"{iteration + 1} — stopping early to prevent loop"
                )
                # Keep whichever response has a \boxed{} answer (prefer refined)
                if "\\boxed{" in refined_response:
                    current_response = refined_response
                # else keep current_response as-is
                history.append({
                    "iteration": iteration + 1,
                    "response": current_response,
                    "reflection": reflection,
                    "early_stop": True,
                    "reason": "repetition_detected"
                })
                break

            # Update current response
            current_response = refined_response

            # Store in history
            history.append({
                "iteration": iteration + 1,
                "response": current_response,
                "reflection": reflection
            })

        self.logger.info("Reflection complete")

        return Message(
            content=current_response,
            message_type=MessageType.AGENT_RESPONSE,
            metadata={
                "num_iterations": self.num_iterations,
                "mode": mode,
                "generator_agent": self.agent.name,
                "critic_agent": self.critic_agent.name,
                "history": history
            }
        )
