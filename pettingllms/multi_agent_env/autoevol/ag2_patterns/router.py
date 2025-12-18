"""
Router Pattern

Conditional routing of tasks to specialized agents based on content or conditions.
Supports keyword-based, LLM-based, and custom routing functions.
"""

from typing import Dict, List, Callable, Optional
from autogen import ConversableAgent, GroupChat, GroupChatManager


def keyword_router(
    keywords_map: Dict[str, ConversableAgent],
    default_agent: ConversableAgent,
    case_sensitive: bool = False
) -> Callable:
    """
    Create a keyword-based routing function.

    Args:
        keywords_map: Map of keywords to agents
        default_agent: Default agent if no keyword matches
        case_sensitive: Whether keyword matching is case-sensitive

    Returns:
        Routing function for speaker_selection_method

    Example:
        >>> router_func = keyword_router(
        ...     {"math": math_agent, "search": search_agent},
        ...     default_agent=general_agent
        ... )
    """
    def router(last_speaker: ConversableAgent, groupchat: GroupChat):
        """Route based on keywords in latest message."""
        if not groupchat.messages:
            return default_agent

        latest_msg = groupchat.messages[-1].get("content", "")

        # Convert to lowercase if not case sensitive
        search_content = latest_msg if case_sensitive else latest_msg.lower()

        # Check each keyword
        for keyword, agent in keywords_map.items():
            check_keyword = keyword if case_sensitive else keyword.lower()
            if check_keyword in search_content:
                print(f"[Router] Keyword '{keyword}' found, routing to {agent.name}")
                return agent

        # No match, use default
        print(f"[Router] No keyword match, routing to default {default_agent.name}")
        return default_agent

    return router


def llm_router(
    classifier_agent: ConversableAgent,
    agents_map: Dict[str, ConversableAgent],
    default_agent: ConversableAgent
) -> Callable:
    """
    Create an LLM-based routing function using a classifier agent.

    Args:
        classifier_agent: Agent that classifies the query
        agents_map: Map of classifications to agents
        default_agent: Default agent if classification not found

    Returns:
        Routing function for speaker_selection_method

    Example:
        >>> classifier = ConversableAgent(
        ...     "Classifier",
        ...     system_message="Classify queries as 'math', 'search', or 'general'"
        ... )
        >>> router_func = llm_router(
        ...     classifier,
        ...     {"math": math_agent, "search": search_agent},
        ...     general_agent
        ... )
    """
    def router(last_speaker: ConversableAgent, groupchat: GroupChat):
        """Route using LLM classification."""
        if not groupchat.messages:
            return default_agent

        # Only classify on first message
        if len(groupchat.messages) > 1:
            # Already routed, check if need to terminate
            if last_speaker != classifier_agent:
                return None  # Terminate after specialist responds
            return default_agent

        latest_msg = groupchat.messages[-1].get("content", "")

        # Ask classifier
        classification_prompt = f"""Classify the following query into one of these categories: {', '.join(agents_map.keys())}

Query: {latest_msg}

Respond with ONLY the category name, nothing else."""

        classification = classifier_agent.generate_reply(
            messages=[{"role": "user", "content": classification_prompt}]
        )

        # Parse classification
        classification = classification.strip().lower()

        # Find matching agent
        for category, agent in agents_map.items():
            if category.lower() in classification:
                print(f"[Router] Classified as '{category}', routing to {agent.name}")
                return agent

        # No match, use default
        print(f"[Router] Classification unclear, routing to default {default_agent.name}")
        return default_agent

    return router


def create_router_groupchat(
    routing_func: Callable,
    agents: List[ConversableAgent],
    max_round: int = 10,
) -> GroupChat:
    """
    Create a router GroupChat with custom routing logic.

    Args:
        routing_func: Function that returns next agent based on context
        agents: List of all agents (including router/classifier)
        max_round: Maximum rounds

    Returns:
        Configured GroupChat instance

    Example:
        >>> router_func = keyword_router(
        ...     {"math": math_agent, "search": search_agent},
        ...     default_agent=general_agent
        ... )
        >>> groupchat = create_router_groupchat(router_func, [math_agent, search_agent, general_agent])
        >>> manager = GroupChatManager(groupchat=groupchat)
        >>> result = agents[0].initiate_chat(manager, message="Calculate 5 * 3")
    """
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=max_round,
        speaker_selection_method=routing_func,
    )

    return groupchat


def condition_router(
    conditions: List[tuple[Callable, ConversableAgent]],
    default_agent: ConversableAgent
) -> Callable:
    """
    Create a condition-based routing function.

    Args:
        conditions: List of (condition_func, agent) tuples
        default_agent: Default agent if no condition matches

    Returns:
        Routing function for speaker_selection_method

    Example:
        >>> def is_math(msg): return any(op in msg for op in ['+', '-', '*', '/'])
        >>> def is_search(msg): return 'search' in msg.lower() or '?' in msg
        >>>
        >>> router_func = condition_router(
        ...     [(is_math, math_agent), (is_search, search_agent)],
        ...     default_agent=general_agent
        ... )
    """
    def router(last_speaker: ConversableAgent, groupchat: GroupChat):
        """Route based on conditions."""
        if not groupchat.messages:
            return default_agent

        latest_msg = groupchat.messages[-1].get("content", "")

        # Check each condition in order
        for condition_func, agent in conditions:
            try:
                if condition_func(latest_msg):
                    print(f"[Router] Condition matched, routing to {agent.name}")
                    return agent
            except Exception as e:
                print(f"[Router] Error in condition function: {e}")
                continue

        # No match, use default
        print(f"[Router] No condition matched, routing to default {default_agent.name}")
        return default_agent

    return router


class RouterConfig:
    """Configuration for creating a router."""

    def __init__(
        self,
        router_type: str,
        agents_map: Dict[str, ConversableAgent],
        default_agent: ConversableAgent,
        **kwargs
    ):
        """
        Initialize router configuration.

        Args:
            router_type: "keyword", "llm", or "condition"
            agents_map: Mapping for routing
            default_agent: Default agent
            **kwargs: Additional parameters for specific router types
        """
        self.router_type = router_type
        self.agents_map = agents_map
        self.default_agent = default_agent
        self.kwargs = kwargs

    def create_routing_func(self) -> Callable:
        """Create the routing function based on configuration."""
        if self.router_type == "keyword":
            return keyword_router(
                self.agents_map,
                self.default_agent,
                self.kwargs.get("case_sensitive", False)
            )

        elif self.router_type == "llm":
            classifier = self.kwargs.get("classifier_agent")
            if not classifier:
                raise ValueError("llm router requires classifier_agent in kwargs")
            return llm_router(classifier, self.agents_map, self.default_agent)

        elif self.router_type == "condition":
            conditions = self.kwargs.get("conditions", [])
            return condition_router(conditions, self.default_agent)

        else:
            raise ValueError(f"Unknown router_type: {self.router_type}")


def create_simple_router(
    query_patterns: Dict[str, ConversableAgent],
    default_agent: ConversableAgent,
    method: str = "keyword"
) -> GroupChat:
    """
    Create a simple router GroupChat with minimal configuration.

    This is a convenience function for common routing scenarios.

    Args:
        query_patterns: Map of patterns/keywords to agents
        default_agent: Default agent
        method: "keyword" or "regex" (default: "keyword")

    Returns:
        Configured GroupChat ready to use

    Example:
        >>> groupchat = create_simple_router(
        ...     {"math": math_agent, "search": search_agent},
        ...     default_agent=general_agent
        ... )
        >>> manager = GroupChatManager(groupchat=groupchat)
        >>> result = general_agent.initiate_chat(manager, message="Calculate 2+2")
    """
    if method == "keyword":
        routing_func = keyword_router(query_patterns, default_agent)
    else:
        raise ValueError(f"Unsupported method: {method}")

    all_agents = list(query_patterns.values()) + [default_agent]
    # Remove duplicates while preserving order
    seen = set()
    unique_agents = []
    for agent in all_agents:
        if agent.name not in seen:
            seen.add(agent.name)
            unique_agents.append(agent)

    return create_router_groupchat(routing_func, unique_agents)
