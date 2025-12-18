"""
AG2 Patterns Module

Workflow pattern implementations for multi-agent coordination.
Includes: Ensemble, Debate, Reflection, Router, and Graph patterns.
"""

from .ensemble import (
    create_ensemble_groupchat,
    VotingStrategy,
    majority_vote,
    weighted_vote,
    consensus_vote,
)

from .debate import (
    create_debate_groupchat,
    DebatePhase,
)

from .router import (
    create_router_groupchat,
    keyword_router,
    llm_router,
)

from .reflection import (
    create_reflection_agent,
    create_reflection_groupchat,
    create_multi_agent_reflection,
    ReflectionPhase,
    format_reflection_history,
    extract_reflection_result,
)

__all__ = [
    # Ensemble
    "create_ensemble_groupchat",
    "VotingStrategy",
    "majority_vote",
    "weighted_vote",
    "consensus_vote",
    # Debate
    "create_debate_groupchat",
    "DebatePhase",
    # Router
    "create_router_groupchat",
    "keyword_router",
    "llm_router",
    # Reflection
    "create_reflection_agent",
    "create_reflection_groupchat",
    "create_multi_agent_reflection",
    "ReflectionPhase",
    "format_reflection_history",
    "extract_reflection_result",
]
