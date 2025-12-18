"""
Debate Pattern

Multi-round debate between agents with a judge making the final decision.
Supports structured argumentation and iterative refinement.
"""

from typing import List, Optional
from enum import Enum
from autogen import ConversableAgent, GroupChat, GroupChatManager


class DebatePhase(Enum):
    """Phases in a debate."""
    INITIAL = "initial"
    REBUTTAL = "rebuttal"
    FINAL = "final"
    JUDGING = "judging"


def create_debate_groupchat(
    debaters: List[ConversableAgent],
    judge: ConversableAgent,
    num_rounds: int = 3,
    max_round: int = 30,
) -> GroupChat:
    """
    Create a debate GroupChat with multiple debaters and a judge.

    This function creates a structured debate where agents argue different
    positions across multiple rounds, followed by a judge's final decision.

    Args:
        debaters: List of debating agents (typically 2-4)
        judge: Judge agent to make final decision
        num_rounds: Number of debate rounds (default: 3)
        max_round: Maximum total rounds including judging (default: 30)

    Returns:
        Configured GroupChat instance

    Example:
        >>> pro = ConversableAgent("Pro", system_message="Argue FOR the topic...")
        >>> con = ConversableAgent("Con", system_message="Argue AGAINST the topic...")
        >>> judge = ConversableAgent("Judge", system_message="Evaluate arguments...")
        >>>
        >>> groupchat = create_debate_groupchat([pro, con], judge, num_rounds=3)
        >>> manager = GroupChatManager(groupchat=groupchat)
        >>> result = pro.initiate_chat(manager, message="Topic: AI regulation")

    Workflow:
        Round 1: All debaters present initial arguments
        Round 2-N: Debaters respond to each other (rebuttals)
        Final: Judge evaluates and decides

    Note:
        - Debaters should have opposing system messages
        - Judge should have neutral evaluation instructions
        - Each round allows all debaters to speak once
    """
    num_debaters = len(debaters)
    current_round = [0]  # Use list to allow modification in closure

    # Define allowed transitions (graph structure)
    allowed_transitions = {}

    # Debaters can transition to each other (debate flow)
    for i, debater in enumerate(debaters):
        # Each debater can go to the next debater (circular)
        next_debaters = debaters[(i + 1):] + debaters[:(i + 1)]
        allowed_transitions[debater] = next_debaters + [judge]  # Can also go to judge

    # Judge is terminal
    allowed_transitions[judge] = []

    def debate_speaker_selection(last_speaker: ConversableAgent, groupchat: GroupChat):
        """Custom speaker selection for debate flow."""
        nonlocal current_round

        # Count messages per phase
        debater_names = [d.name for d in debaters]
        assistant_messages = [
            m for m in groupchat.messages
            if m.get("role") == "assistant" and m.get("name") in debater_names
        ]

        num_debater_messages = len(assistant_messages)

        # Calculate current round (each round = all debaters speak once)
        calculated_round = num_debater_messages // num_debaters + 1
        current_round[0] = calculated_round

        # Check if debate rounds are complete
        if calculated_round > num_rounds:
            # Time for judge
            if last_speaker != judge:
                print(f"\n[Debate] All {num_rounds} rounds complete. Judge will now decide.\n")
                return judge
            else:
                # Judge has spoken, terminate
                print(f"[Debate] Judge has made the final decision. Debate concluded.\n")
                return None

        # Within debate rounds - rotate through debaters
        print(f"[Debate] Round {calculated_round}/{num_rounds}")

        # Determine next debater in rotation
        if last_speaker not in debaters:
            # Start with first debater
            next_speaker = debaters[0]
        else:
            # Get next debater in sequence
            current_idx = debaters.index(last_speaker)
            next_idx = (current_idx + 1) % num_debaters
            next_speaker = debaters[next_idx]

        print(f"[Debate] Next speaker: {next_speaker.name}")
        return next_speaker

    # Create GroupChat
    groupchat = GroupChat(
        agents=debaters + [judge],
        messages=[],
        max_round=max_round,
        allowed_or_disallowed_speaker_transitions=allowed_transitions,
        speaker_transitions_type="allowed",
        speaker_selection_method=debate_speaker_selection,
    )

    return groupchat


def format_debate_history(groupchat: GroupChat) -> str:
    """
    Format debate history for display or analysis.

    Args:
        groupchat: The debate GroupChat

    Returns:
        Formatted debate history as string
    """
    history = []
    history.append("=== Debate History ===\n")

    current_round = 0
    messages_in_round = 0

    for msg in groupchat.messages:
        if msg.get("role") == "assistant":
            speaker = msg.get("name", "Unknown")
            content = msg.get("content", "")

            # Detect round transitions
            if messages_in_round % len([a for a in groupchat.agents if a.name != "Judge"]) == 0:
                current_round += 1
                history.append(f"\n--- Round {current_round} ---\n")
                messages_in_round = 0

            history.append(f"\n[{speaker}]:\n{content}\n")
            messages_in_round += 1

    history.append("\n=== End of Debate ===")
    return "".join(history)


class DebateResult:
    """Container for debate results."""

    def __init__(
        self,
        judge_decision: str,
        debate_history: List[dict],
        num_rounds: int,
        num_debaters: int
    ):
        self.judge_decision = judge_decision
        self.debate_history = debate_history
        self.num_rounds = num_rounds
        self.num_debaters = num_debaters

    def __str__(self):
        return f"DebateResult(rounds={self.num_rounds}, debaters={self.num_debaters})"

    def summary(self) -> str:
        """Get a summary of the debate."""
        summary = f"""Debate Summary
Rounds: {self.num_rounds}
Debaters: {self.num_debaters}

Judge's Decision:
{self.judge_decision}

Total messages: {len(self.debate_history)}
"""
        return summary


def create_pro_con_debate(
    topic: str,
    pro_agent: Optional[ConversableAgent] = None,
    con_agent: Optional[ConversableAgent] = None,
    judge_agent: Optional[ConversableAgent] = None,
    num_rounds: int = 3,
) -> GroupChat:
    """
    Create a classic pro/con debate GroupChat.

    This is a convenience function that creates a standard two-sided debate
    with pre-configured system messages.

    Args:
        topic: The debate topic
        pro_agent: Agent arguing FOR (creates if None)
        con_agent: Agent arguing AGAINST (creates if None)
        judge_agent: Judge agent (creates if None)
        num_rounds: Number of debate rounds

    Returns:
        Configured debate GroupChat

    Example:
        >>> groupchat = create_pro_con_debate(
        ...     topic="AI should be regulated by governments",
        ...     num_rounds=3
        ... )
        >>> manager = GroupChatManager(groupchat=groupchat)
        >>> result = groupchat.agents[0].initiate_chat(manager, message="Begin debate")
    """
    from ..ag2_core import create_custom_agent

    # Create pro agent if not provided
    if pro_agent is None:
        pro_agent = create_custom_agent(
            name="Pro_Debater",
            system_message=f"""You are debating FOR the following topic:
"{topic}"

Your role:
- Present strong arguments SUPPORTING the topic
- Counter opposing arguments effectively
- Use evidence and logic
- Be respectful but persuasive
- Build on your previous arguments in later rounds

Remember: You are arguing FOR the topic.""",
            temperature=0.7
        )

    # Create con agent if not provided
    if con_agent is None:
        con_agent = create_custom_agent(
            name="Con_Debater",
            system_message=f"""You are debating AGAINST the following topic:
"{topic}"

Your role:
- Present strong arguments OPPOSING the topic
- Counter supporting arguments effectively
- Use evidence and logic
- Be respectful but persuasive
- Build on your previous arguments in later rounds

Remember: You are arguing AGAINST the topic.""",
            temperature=0.7
        )

    # Create judge if not provided
    if judge_agent is None:
        judge_agent = create_custom_agent(
            name="Judge",
            system_message=f"""You are an impartial judge evaluating a debate on:
"{topic}"

Your role:
- Review all arguments from both sides
- Evaluate based on:
  * Strength of evidence
  * Logical consistency
  * Persuasiveness
  * Rebuttal effectiveness
- Provide a fair and balanced decision
- Explain your reasoning

Conclude with: "Winner: [Pro/Con/Tie]" and explain why.""",
            temperature=0.3  # Lower temp for more consistent judging
        )

    # Create debate groupchat
    return create_debate_groupchat(
        debaters=[pro_agent, con_agent],
        judge=judge_agent,
        num_rounds=num_rounds
    )
