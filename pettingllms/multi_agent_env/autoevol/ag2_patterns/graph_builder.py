"""
Graph Builder Pattern

Build complex directed graph workflows with conditional transitions.
Supports cycles, conditional edges, and state management for advanced agent coordination.
"""

from typing import Dict, List, Callable, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from autogen import ConversableAgent, GroupChat, GroupChatManager


class EdgeType(Enum):
    """Types of edges in the graph."""
    UNCONDITIONAL = "unconditional"
    CONDITIONAL = "conditional"
    PROBABILISTIC = "probabilistic"


@dataclass
class GraphNode:
    """A node in the agent graph."""
    agent: ConversableAgent
    name: str
    node_type: str = "agent"  # agent, start, end
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, GraphNode):
            return self.name == other.name
        return False


@dataclass
class GraphEdge:
    """An edge connecting two nodes."""
    from_node: str
    to_node: str
    edge_type: EdgeType = EdgeType.UNCONDITIONAL
    condition: Optional[Callable] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphWorkflow:
    """
    Builder for creating complex agent graph workflows.

    Supports:
    - Directed graphs with cycles
    - Conditional transitions
    - State management
    - Multiple entry/exit points
    """

    def __init__(self, name: str = "GraphWorkflow"):
        """Initialize the graph workflow."""
        self.name = name
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.start_node: Optional[str] = None
        self.end_nodes: Set[str] = set()
        self.state: Dict[str, Any] = {}

    def add_node(
        self,
        agent: ConversableAgent,
        node_name: Optional[str] = None,
        node_type: str = "agent",
        **metadata
    ) -> 'GraphWorkflow':
        """
        Add a node to the graph.

        Args:
            agent: The ConversableAgent for this node
            node_name: Optional custom name (defaults to agent.name)
            node_type: Type of node (agent, start, end)
            **metadata: Additional metadata for the node

        Returns:
            Self for chaining

        Example:
            >>> workflow = GraphWorkflow()
            >>> workflow.add_node(search_agent, "search")
            >>> workflow.add_node(verify_agent, "verify")
        """
        name = node_name or agent.name
        node = GraphNode(
            agent=agent,
            name=name,
            node_type=node_type,
            metadata=metadata
        )
        self.nodes[name] = node
        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable] = None,
        priority: int = 0,
        **metadata
    ) -> 'GraphWorkflow':
        """
        Add an edge between two nodes.

        Args:
            from_node: Source node name
            to_node: Target node name
            condition: Optional condition function(message, state) -> bool
            priority: Priority for edge selection (higher = higher priority)
            **metadata: Additional metadata

        Returns:
            Self for chaining

        Example:
            >>> workflow.add_edge("search", "verify")
            >>> workflow.add_edge("verify", "search", condition=lambda m, s: s.get("needs_more_info"))
        """
        edge_type = EdgeType.CONDITIONAL if condition else EdgeType.UNCONDITIONAL

        edge = GraphEdge(
            from_node=from_node,
            to_node=to_node,
            edge_type=edge_type,
            condition=condition,
            priority=priority,
            metadata=metadata
        )
        self.edges.append(edge)
        return self

    def set_start_node(self, node_name: str) -> 'GraphWorkflow':
        """
        Set the starting node for the workflow.

        Args:
            node_name: Name of the start node

        Returns:
            Self for chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        self.start_node = node_name
        return self

    def add_end_node(self, node_name: str) -> 'GraphWorkflow':
        """
        Mark a node as an end node (terminal).

        Args:
            node_name: Name of the end node

        Returns:
            Self for chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node '{node_name}' not found in graph")
        self.end_nodes.add(node_name)
        return self

    def get_outgoing_edges(self, node_name: str) -> List[GraphEdge]:
        """Get all outgoing edges from a node."""
        return [e for e in self.edges if e.from_node == node_name]

    def get_next_node(
        self,
        current_node: str,
        message: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Determine the next node based on edges and conditions.

        Args:
            current_node: Current node name
            message: Latest message for condition evaluation

        Returns:
            Next node name or None if terminal
        """
        # Check if current node is terminal
        if current_node in self.end_nodes:
            return None

        # Get outgoing edges
        edges = self.get_outgoing_edges(current_node)
        if not edges:
            return None

        # Sort by priority (descending)
        edges.sort(key=lambda e: e.priority, reverse=True)

        # Evaluate conditions
        for edge in edges:
            if edge.edge_type == EdgeType.UNCONDITIONAL:
                return edge.to_node
            elif edge.edge_type == EdgeType.CONDITIONAL and edge.condition:
                try:
                    if edge.condition(message, self.state):
                        return edge.to_node
                except Exception as e:
                    print(f"[Graph] Error evaluating condition: {e}")
                    continue

        # No matching edge found
        return None

    def build_groupchat(
        self,
        max_round: int = 30,
        initial_message: Optional[str] = None
    ) -> GroupChat:
        """
        Build an AG2 GroupChat from the graph workflow.

        Args:
            max_round: Maximum number of rounds
            initial_message: Optional initial message

        Returns:
            Configured GroupChat

        Example:
            >>> workflow = GraphWorkflow()
            >>> workflow.add_node(agent1, "a1").add_node(agent2, "a2")
            >>> workflow.add_edge("a1", "a2").set_start_node("a1")
            >>> groupchat = workflow.build_groupchat()
            >>> manager = GroupChatManager(groupchat=groupchat)
        """
        if not self.start_node:
            raise ValueError("Start node not set. Call set_start_node() first.")

        # Build allowed transitions dict
        allowed_transitions = {}
        for node_name, node in self.nodes.items():
            outgoing = self.get_outgoing_edges(node_name)
            targets = [self.nodes[e.to_node].agent for e in outgoing]
            allowed_transitions[node.agent] = targets

        # Get all agents
        agents = [node.agent for node in self.nodes.values()]

        # Create speaker selection function
        def graph_speaker_selection(last_speaker: ConversableAgent, groupchat: GroupChat):
            """Select next speaker based on graph structure."""
            # Get current node name
            current_node_name = None
            for name, node in self.nodes.items():
                if node.agent == last_speaker:
                    current_node_name = name
                    break

            if current_node_name is None:
                # Start from start_node
                if self.start_node:
                    print(f"[Graph] Starting workflow at: {self.start_node}")
                    return self.nodes[self.start_node].agent
                return None

            # Get latest message
            latest_message = None
            if groupchat.messages:
                latest_message = groupchat.messages[-1]

            # Determine next node
            next_node_name = self.get_next_node(current_node_name, latest_message)

            if next_node_name is None:
                print(f"[Graph] Workflow complete at: {current_node_name}")
                return None

            print(f"[Graph] Transition: {current_node_name} → {next_node_name}")
            return self.nodes[next_node_name].agent

        # Create GroupChat
        groupchat = GroupChat(
            agents=agents,
            messages=[],
            max_round=max_round,
            allowed_or_disallowed_speaker_transitions=allowed_transitions,
            speaker_transitions_type="allowed",
            speaker_selection_method=graph_speaker_selection,
        )

        return groupchat

    def visualize(self) -> str:
        """
        Generate a text-based visualization of the graph.

        Returns:
            Graph visualization as string
        """
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"Graph Workflow: {self.name}")
        lines.append(f"{'='*60}\n")

        lines.append(f"Nodes ({len(self.nodes)}):")
        for name, node in self.nodes.items():
            marker = ""
            if name == self.start_node:
                marker = " [START]"
            elif name in self.end_nodes:
                marker = " [END]"
            lines.append(f"  - {name} ({node.node_type}){marker}")

        lines.append(f"\nEdges ({len(self.edges)}):")
        for edge in self.edges:
            condition_str = ""
            if edge.edge_type == EdgeType.CONDITIONAL:
                condition_str = " [conditional]"
            priority_str = f" (priority={edge.priority})" if edge.priority != 0 else ""
            lines.append(f"  - {edge.from_node} → {edge.to_node}{condition_str}{priority_str}")

        lines.append(f"\n{'='*60}\n")
        return "\n".join(lines)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the graph structure.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Check start node
        if not self.start_node:
            errors.append("No start node set")
        elif self.start_node not in self.nodes:
            errors.append(f"Start node '{self.start_node}' not found")

        # Check end nodes
        if not self.end_nodes:
            errors.append("No end nodes defined (workflow may not terminate)")

        # Check edge references
        for edge in self.edges:
            if edge.from_node not in self.nodes:
                errors.append(f"Edge references unknown from_node: {edge.from_node}")
            if edge.to_node not in self.nodes:
                errors.append(f"Edge references unknown to_node: {edge.to_node}")

        # Check connectivity from start
        if self.start_node and self.start_node in self.nodes:
            reachable = self._get_reachable_nodes(self.start_node)
            unreachable = set(self.nodes.keys()) - reachable - {self.start_node}
            if unreachable:
                errors.append(f"Unreachable nodes from start: {unreachable}")

        return len(errors) == 0, errors

    def _get_reachable_nodes(self, start: str) -> Set[str]:
        """Get all nodes reachable from start node (BFS)."""
        reachable = set()
        queue = [start]
        visited = {start}

        while queue:
            current = queue.pop(0)
            for edge in self.get_outgoing_edges(current):
                if edge.to_node not in visited:
                    visited.add(edge.to_node)
                    reachable.add(edge.to_node)
                    queue.append(edge.to_node)

        return reachable


def create_linear_workflow(
    agents: List[ConversableAgent],
    agent_names: Optional[List[str]] = None
) -> GraphWorkflow:
    """
    Create a simple linear workflow (A → B → C → ...).

    Args:
        agents: List of agents in order
        agent_names: Optional custom names for agents

    Returns:
        GraphWorkflow configured as linear chain

    Example:
        >>> workflow = create_linear_workflow([search_agent, verify_agent, summarize_agent])
        >>> groupchat = workflow.build_groupchat()
    """
    if not agents:
        raise ValueError("At least one agent required")

    workflow = GraphWorkflow(name="LinearWorkflow")

    # Add nodes
    names = []
    for i, agent in enumerate(agents):
        name = agent_names[i] if agent_names and i < len(agent_names) else agent.name
        workflow.add_node(agent, name)
        names.append(name)

    # Add edges in sequence
    for i in range(len(names) - 1):
        workflow.add_edge(names[i], names[i + 1])

    # Set start and end
    workflow.set_start_node(names[0])
    workflow.add_end_node(names[-1])

    return workflow


def create_conditional_workflow(
    agents: Dict[str, ConversableAgent],
    transitions: List[Tuple[str, str, Optional[Callable]]],
    start_node: str
) -> GraphWorkflow:
    """
    Create a workflow with conditional transitions.

    Args:
        agents: Dict mapping names to agents
        transitions: List of (from, to, condition) tuples
        start_node: Starting node name

    Returns:
        GraphWorkflow with conditional edges

    Example:
        >>> agents = {"search": search_agent, "verify": verify_agent, "summarize": sum_agent}
        >>> transitions = [
        ...     ("search", "verify", None),
        ...     ("verify", "summarize", lambda m, s: s.get("verified")),
        ...     ("verify", "search", lambda m, s: not s.get("verified"))
        ... ]
        >>> workflow = create_conditional_workflow(agents, transitions, "search")
    """
    workflow = GraphWorkflow(name="ConditionalWorkflow")

    # Add all nodes
    for name, agent in agents.items():
        workflow.add_node(agent, name)

    # Add edges with conditions
    for from_node, to_node, condition in transitions:
        workflow.add_edge(from_node, to_node, condition=condition)

    # Set start node
    workflow.set_start_node(start_node)

    return workflow


def create_parallel_workflow(
    input_agent: ConversableAgent,
    parallel_agents: List[ConversableAgent],
    output_agent: ConversableAgent
) -> GraphWorkflow:
    """
    Create a workflow with parallel processing (fan-out, fan-in).

    Structure: Input → [Agent1, Agent2, ...] → Output

    Args:
        input_agent: Initial agent
        parallel_agents: Agents to run in parallel
        output_agent: Final aggregation agent

    Returns:
        GraphWorkflow with parallel structure

    Example:
        >>> workflow = create_parallel_workflow(
        ...     input_agent=coordinator,
        ...     parallel_agents=[worker1, worker2, worker3],
        ...     output_agent=aggregator
        ... )
    """
    workflow = GraphWorkflow(name="ParallelWorkflow")

    # Add input node
    workflow.add_node(input_agent, "input")

    # Add parallel nodes
    for i, agent in enumerate(parallel_agents):
        name = f"parallel_{i}"
        workflow.add_node(agent, name)
        # Input fans out to all parallel agents
        workflow.add_edge("input", name)

    # Add output node
    workflow.add_node(output_agent, "output")

    # All parallel agents converge to output
    for i in range(len(parallel_agents)):
        workflow.add_edge(f"parallel_{i}", "output")

    workflow.set_start_node("input")
    workflow.add_end_node("output")

    return workflow


def create_loop_workflow(
    agents: List[ConversableAgent],
    loop_condition: Callable,
    max_iterations: int = 5
) -> GraphWorkflow:
    """
    Create a workflow with a loop structure.

    Args:
        agents: Agents in the loop
        loop_condition: Condition to continue looping
        max_iterations: Maximum loop iterations

    Returns:
        GraphWorkflow with loop

    Example:
        >>> def should_continue(msg, state):
        ...     return state.get("iteration", 0) < 3
        >>> workflow = create_loop_workflow([agent1, agent2], should_continue)
    """
    workflow = GraphWorkflow(name="LoopWorkflow")

    # Add nodes
    names = [f"node_{i}" for i in range(len(agents))]
    for name, agent in zip(names, agents):
        workflow.add_node(agent, name)

    # Add forward edges
    for i in range(len(names) - 1):
        workflow.add_edge(names[i], names[i + 1])

    # Add loop-back edge with condition
    def iteration_condition(msg, state):
        state["iteration"] = state.get("iteration", 0) + 1
        should_loop = loop_condition(msg, state) and state["iteration"] < max_iterations
        if should_loop:
            print(f"[Loop] Iteration {state['iteration']}/{max_iterations}")
        return should_loop

    workflow.add_edge(names[-1], names[0], condition=iteration_condition, priority=10)

    # Add exit edge (lower priority)
    workflow.add_node(agents[0], "exit", node_type="end")  # Reuse first agent as exit
    workflow.add_edge(names[-1], "exit", priority=0)

    workflow.set_start_node(names[0])
    workflow.add_end_node("exit")

    return workflow
