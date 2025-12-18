"""
Reflection Pattern

Self-reflection and iterative improvement through critique and refinement.
Supports multiple iterations of self-evaluation and response enhancement.
"""

from typing import List, Optional, Callable, Dict, Any
from autogen import ConversableAgent, GroupChat, GroupChatManager


class ReflectionPhase:
    """Phases in a reflection cycle."""
    INITIAL = "initial"
    CRITIQUE = "critique"
    REFINE = "refine"
    COMPLETE = "complete"


def create_reflection_agent(
    name: str,
    system_message: str,
    llm_config: Dict[str, Any],
    critique_prompt: Optional[str] = None,
    refinement_prompt: Optional[str] = None
) -> ConversableAgent:
    """
    Create an agent with built-in reflection capabilities.

    Args:
        name: Agent name
        system_message: Base system message
        llm_config: LLM configuration
        critique_prompt: Custom critique prompt template
        refinement_prompt: Custom refinement prompt template

    Returns:
        ConversableAgent configured for reflection

    Example:
        >>> agent = create_reflection_agent(
        ...     "ReflectiveAgent",
        ...     "You are a helpful assistant.",
        ...     llm_config=get_deepseek_config()
        ... )
    """
    if critique_prompt is None:
        critique_prompt = """Review your previous response and provide constructive criticism:

Previous response:
{response}

Identify:
1. Strengths of the response
2. Weaknesses or gaps
3. Potential improvements
4. Factual errors or inconsistencies

Be specific and actionable in your critique."""

    if refinement_prompt is None:
        refinement_prompt = """Based on the critique, provide an improved response:

Original response:
{response}

Critique:
{critique}

Provide a refined response that addresses the critique points while maintaining the strengths."""

    agent = ConversableAgent(
        name=name,
        system_message=system_message,
        llm_config=llm_config,
    )

    # Store prompts as agent attributes for later use
    agent._critique_prompt = critique_prompt
    agent._refinement_prompt = refinement_prompt

    return agent


def create_reflection_groupchat(
    agent: ConversableAgent,
    num_iterations: int = 2,
    max_round: int = 20,
    critique_prompt: Optional[str] = None,
    refinement_prompt: Optional[str] = None
) -> GroupChat:
    """
    Create a reflection GroupChat for iterative self-improvement.

    This creates a workflow where the agent:
    1. Generates initial response
    2. Critiques its own response
    3. Refines based on critique
    4. Repeats for num_iterations

    Args:
        agent: The agent to reflect
        num_iterations: Number of reflection cycles
        max_round: Maximum total rounds
        critique_prompt: Custom critique prompt
        refinement_prompt: Custom refinement prompt

    Returns:
        Configured GroupChat for reflection

    Example:
        >>> agent = create_reflection_agent("Agent", "You are helpful.", llm_config)
        >>> groupchat = create_reflection_groupchat(agent, num_iterations=2)
        >>> manager = GroupChatManager(groupchat=groupchat)
        >>> result = agent.initiate_chat(manager, message="Explain quantum computing")

    Workflow:
        1. Initial response
        2. Self-critique
        3. Refinement
        4. Repeat steps 2-3 for remaining iterations
    """
    # Use provided prompts or fall back to agent's stored prompts
    if critique_prompt is None:
        critique_prompt = getattr(agent, '_critique_prompt', None)
    if refinement_prompt is None:
        refinement_prompt = getattr(agent, '_refinement_prompt', None)

    # Default prompts if none provided
    if critique_prompt is None:
        critique_prompt = """Review the previous response critically.

Identify strengths, weaknesses, and areas for improvement.
Be specific and constructive."""

    if refinement_prompt is None:
        refinement_prompt = """Provide an improved version of your response.

Address the critique points while preserving strengths.
Make the response clearer and more comprehensive."""

    current_iteration = [0]
    current_phase = [ReflectionPhase.INITIAL]
    original_query = [None]
    responses_history = []

    def reflection_speaker_selection(last_speaker: ConversableAgent, groupchat: GroupChat):
        """Manage reflection cycle phases."""
        nonlocal current_iteration, current_phase, original_query, responses_history

        if not groupchat.messages:
            return agent

        # Store original query from first user message
        if original_query[0] is None:
            user_msgs = [m for m in groupchat.messages if m.get("role") == "user"]
            if user_msgs:
                original_query[0] = user_msgs[0].get("content", "")

        assistant_messages = [
            m for m in groupchat.messages
            if m.get("role") == "assistant" and m.get("name") == agent.name
        ]

        num_responses = len(assistant_messages)

        # Determine current phase based on message count
        # Pattern: initial(1) → critique(2) → refine(3) → critique(4) → refine(5) → ...
        if num_responses == 0:
            current_phase[0] = ReflectionPhase.INITIAL
            print(f"[Reflection] Phase: Initial response")
            return agent

        elif num_responses == 1:
            # Initial response complete, start first critique
            current_iteration[0] = 1
            current_phase[0] = ReflectionPhase.CRITIQUE
            print(f"\n[Reflection] Iteration {current_iteration[0]}/{num_iterations} - Phase: Critique")

            # Store initial response
            responses_history.append(assistant_messages[-1]["content"])

            # Inject critique prompt
            critique_msg = critique_prompt.format(response=responses_history[-1])
            groupchat.messages.append({
                "role": "user",
                "content": critique_msg,
                "name": "ReflectionSystem"
            })
            return agent

        elif num_responses > 1:
            # Calculate which phase we're in
            # After initial (1): critique, refine, critique, refine, ...
            phase_in_cycle = (num_responses - 1) % 2

            if phase_in_cycle == 1:
                # Just finished critique, now refine
                current_phase[0] = ReflectionPhase.REFINE
                print(f"[Reflection] Iteration {current_iteration[0]}/{num_iterations} - Phase: Refine")

                # Get last critique
                critique = assistant_messages[-1]["content"]
                last_response = responses_history[-1] if responses_history else ""

                # Inject refinement prompt
                refine_msg = refinement_prompt.format(
                    response=last_response,
                    critique=critique
                )
                groupchat.messages.append({
                    "role": "user",
                    "content": refine_msg,
                    "name": "ReflectionSystem"
                })
                return agent

            else:
                # Just finished refinement
                refined_response = assistant_messages[-1]["content"]
                responses_history.append(refined_response)

                # Check if we should continue iterations
                current_iteration[0] += 1

                if current_iteration[0] <= num_iterations:
                    # Start next critique cycle
                    current_phase[0] = ReflectionPhase.CRITIQUE
                    print(f"\n[Reflection] Iteration {current_iteration[0]}/{num_iterations} - Phase: Critique")

                    critique_msg = critique_prompt.format(response=refined_response)
                    groupchat.messages.append({
                        "role": "user",
                        "content": critique_msg,
                        "name": "ReflectionSystem"
                    })
                    return agent
                else:
                    # All iterations complete
                    current_phase[0] = ReflectionPhase.COMPLETE
                    print(f"\n[Reflection] Complete - {num_iterations} iterations finished")

                    # Add final summary
                    groupchat.messages.append({
                        "role": "assistant",
                        "name": "ReflectionResult",
                        "content": f"**Final Refined Response:**\n\n{refined_response}"
                    })
                    return None  # Terminate

        return None

    groupchat = GroupChat(
        agents=[agent],
        messages=[],
        max_round=max_round,
        speaker_selection_method=reflection_speaker_selection,
    )

    return groupchat


def create_multi_agent_reflection(
    initial_agent: ConversableAgent,
    critique_agent: ConversableAgent,
    num_iterations: int = 2,
    max_round: int = 20
) -> GroupChat:
    """
    Create a reflection GroupChat with separate agents for initial response and critique.

    This variant uses two different agents:
    - initial_agent: Generates and refines responses
    - critique_agent: Provides critical feedback

    Args:
        initial_agent: Agent that generates responses
        critique_agent: Agent that critiques responses
        num_iterations: Number of reflection cycles
        max_round: Maximum total rounds

    Returns:
        Configured GroupChat for multi-agent reflection

    Example:
        >>> responder = ConversableAgent("Responder", system_message="Answer questions...", llm_config=config)
        >>> critic = ConversableAgent("Critic", system_message="Provide constructive critique...", llm_config=config)
        >>> groupchat = create_multi_agent_reflection(responder, critic, num_iterations=2)
        >>> manager = GroupChatManager(groupchat=groupchat)
        >>> result = responder.initiate_chat(manager, message="Explain machine learning")
    """
    current_iteration = [0]
    original_query = [None]
    last_response = [None]

    # Allow self-transitions for tool execution!
    # When an agent makes a tool call, it needs to process the tool response itself
    # before transitioning to the next agent
    allowed_transitions = {
        initial_agent: [initial_agent, critique_agent],  # Allow self-transition for tool calls
        critique_agent: [initial_agent]
    }

    def multi_agent_reflection_selection(last_speaker: ConversableAgent, groupchat: GroupChat):
        """Alternate between responder and critic."""
        nonlocal current_iteration, original_query, last_response

        if not groupchat.messages:
            return initial_agent

        # Store original query
        if original_query[0] is None:
            user_msgs = [m for m in groupchat.messages if m.get("role") == "user"]
            if user_msgs:
                original_query[0] = user_msgs[0].get("content", "")

        # Helper to check if a message is a real response (not just a tool call)
        def is_real_response(msg):
            """Check if message has actual content (not just tool_calls)."""
            if msg.get("role") != "assistant":
                return False
            # If message has tool_calls but no content, it's not a real response
            if msg.get("tool_calls") and not msg.get("content"):
                return False
            # Must have actual content
            return bool(msg.get("content"))

        # Get the last message to check for pending tool calls
        last_msg = groupchat.messages[-1] if groupchat.messages else None
        
        # If last message was a tool call from initial_agent, keep initial_agent to process result
        if last_msg and last_msg.get("name") == initial_agent.name:
            if last_msg.get("tool_calls") and not last_msg.get("content"):
                print(f"[Multi-Agent Reflection] {initial_agent.name} processing tool call...")
                return initial_agent
        
        # If last message was a tool response, keep initial_agent to generate actual response
        if last_msg and last_msg.get("role") == "tool":
            print(f"[Multi-Agent Reflection] {initial_agent.name} generating response from tool result...")
            return initial_agent

        # Count only real responses (with actual content, not just tool calls)
        responder_messages = [
            m for m in groupchat.messages
            if m.get("name") == initial_agent.name and is_real_response(m)
        ]
        critic_messages = [
            m for m in groupchat.messages
            if m.get("name") == critique_agent.name and is_real_response(m)
        ]

        num_responses = len(responder_messages)
        num_critiques = len(critic_messages)

        if num_responses == 0:
            # Initial response
            print(f"[Multi-Agent Reflection] {initial_agent.name} generating initial response")
            return initial_agent

        elif num_critiques < num_responses:
            # Need critique
            current_iteration[0] = num_critiques + 1
            print(f"[Multi-Agent Reflection] Iteration {current_iteration[0]}/{num_iterations} - {critique_agent.name} critiquing")
            last_response[0] = responder_messages[-1].get("content", "")
            return critique_agent

        elif num_responses <= num_iterations:
            # Need refinement
            print(f"[Multi-Agent Reflection] Iteration {current_iteration[0]}/{num_iterations} - {initial_agent.name} refining")
            return initial_agent

        else:
            # Complete
            print(f"[Multi-Agent Reflection] Complete - {num_iterations} iterations finished")
            final_response = responder_messages[-1].get("content", "")
            groupchat.messages.append({
                "role": "assistant",
                "name": "ReflectionResult",
                "content": f"**Final Refined Response:**\n\n{final_response}"
            })
            return None

    groupchat = GroupChat(
        agents=[initial_agent, critique_agent],
        messages=[],
        max_round=max_round,
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
        speaker_selection_method=multi_agent_reflection_selection,
    )

    return groupchat


def format_reflection_history(groupchat: GroupChat) -> str:
    """
    Format reflection history for display.

    Args:
        groupchat: The reflection GroupChat

    Returns:
        Formatted reflection history
    """
    history = []
    history.append("=== Reflection History ===\n")

    iteration = 0
    phase = "Initial"

    for msg in groupchat.messages:
        if msg.get("role") == "assistant":
            speaker = msg.get("name", "Unknown")
            content = msg.get("content", "")

            # Detect phase transitions
            if "critique" in content.lower()[:100] and speaker != "ReflectionResult":
                iteration += 1
                phase = "Critique"
                history.append(f"\n--- Iteration {iteration}: Critique ---\n")
            elif iteration > 0 and phase == "Critique":
                phase = "Refinement"
                history.append(f"\n--- Iteration {iteration}: Refinement ---\n")
            elif speaker == "ReflectionResult":
                history.append(f"\n--- Final Result ---\n")

            history.append(f"\n[{speaker}]:\n{content}\n")

    history.append("\n=== End of Reflection ===")
    return "".join(history)


class ReflectionResult:
    """Container for reflection results."""

    def __init__(
        self,
        initial_response: str,
        critiques: List[str],
        refined_responses: List[str],
        final_response: str,
        num_iterations: int
    ):
        self.initial_response = initial_response
        self.critiques = critiques
        self.refined_responses = refined_responses
        self.final_response = final_response
        self.num_iterations = num_iterations

    def __str__(self):
        return f"ReflectionResult(iterations={self.num_iterations}, improvements={len(self.refined_responses)})"

    def summary(self) -> str:
        """Get a summary of the reflection process."""
        summary = f"""Reflection Summary
Iterations: {self.num_iterations}
Refinements: {len(self.refined_responses)}

Initial Response Length: {len(self.initial_response)} chars
Final Response Length: {len(self.final_response)} chars

Improvement: {len(self.final_response) - len(self.initial_response):+d} chars
"""
        return summary

    def get_improvements(self) -> List[str]:
        """Extract key improvements from critiques."""
        improvements = []
        for critique in self.critiques:
            # Simple extraction - look for numbered points or bullet points
            lines = critique.split('\n')
            for line in lines:
                if line.strip() and (line[0].isdigit() or line.strip().startswith('-') or line.strip().startswith('*')):
                    improvements.append(line.strip())
        return improvements


def extract_reflection_result(groupchat: GroupChat) -> ReflectionResult:
    """
    Extract reflection results from GroupChat.

    Args:
        groupchat: The reflection GroupChat

    Returns:
        ReflectionResult object with all iterations
    """
    assistant_messages = [
        m for m in groupchat.messages
        if m.get("role") == "assistant" and m.get("name") != "ReflectionResult"
    ]

    if not assistant_messages:
        raise ValueError("No reflection messages found")

    initial_response = assistant_messages[0]["content"]

    critiques = []
    refined_responses = []

    # Parse alternating critique/refinement pattern (after initial)
    for i in range(1, len(assistant_messages)):
        content = assistant_messages[i]["content"]
        if i % 2 == 1:  # Odd index = critique
            critiques.append(content)
        else:  # Even index = refinement
            refined_responses.append(content)

    final_response = refined_responses[-1] if refined_responses else initial_response
    num_iterations = len(critiques)

    return ReflectionResult(
        initial_response=initial_response,
        critiques=critiques,
        refined_responses=refined_responses,
        final_response=final_response,
        num_iterations=num_iterations
    )
