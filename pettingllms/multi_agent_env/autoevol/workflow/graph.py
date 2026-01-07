"""
Graph-based workflow system for flexible agent interaction.

This provides maximum flexibility for customizing agent interactions,
unlike the linear Workflow class. You can:
- Define custom edges between agents
- Conditional routing
- Loops and cycles
- Complete control over execution flow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Optional, Callable, Union, Any
from dataclasses import dataclass
from workflow.core import WorkflowNode, Context, Message, MessageType
import logging


@dataclass
class Edge:
    """An edge connecting two nodes in the graph."""
    source: str
    target: str
    condition: Optional[Callable[[Context], bool]] = None
    
    def should_follow(self, context: Context) -> bool:
        """Check if this edge should be followed given the context."""
        if self.condition is None:
            return True
        return self.condition(context)


class AgentGraph:
    """Graph-based workflow for flexible agent interactions.
    
    Unlike the linear Workflow, this allows you to:
    - Define any agent interaction pattern (not just sequential)
    - Add conditional edges
    - Create loops and cycles
    - Full control over execution flow
    
    Example:
        graph = AgentGraph()
        graph.add_node("search", search_agent)
        graph.add_node("summarize", summarize_agent)
        graph.add_node("verify", verify_agent)
        
        # Define edges with conditions
        graph.add_edge("search", "summarize")
        graph.add_edge(
            "summarize", 
            "verify",
            condition=lambda ctx: ctx.get_state("needs_verification")
        )
        graph.add_edge("verify", "search", 
            condition=lambda ctx: not ctx.get_state("verified")
        )
        
        graph.set_entry_point("search")
        graph.set_finish_point("summarize")
        
        result = graph.run("Your query")
    """
    
    def __init__(self, name: str = "agent_graph", max_steps: int = 50):
        """Initialize the agent graph.
        
        Args:
            name: Graph name
            max_steps: Maximum execution steps to prevent infinite loops
        """
        self.name = name
        self.max_steps = max_steps
        self.nodes: Dict[str, WorkflowNode] = {}
        self.edges: List[Edge] = []
        self.entry_point: Optional[str] = None
        self.finish_points: List[str] = []
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger(f"AgentGraph.{self.name}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                f'[%(asctime)s] Graph.{self.name} - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def add_node(self, name: str, node: WorkflowNode) -> 'AgentGraph':
        """Add a node to the graph.
        
        Args:
            name: Node identifier
            node: WorkflowNode instance
            
        Returns:
            Self for chaining
        """
        self.nodes[name] = node
        self.logger.info(f"Added node: {name} ({node.__class__.__name__})")
        return self
    
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable[[Context], bool]] = None
    ) -> 'AgentGraph':
        """Add an edge between nodes.
        
        Args:
            source: Source node name
            target: Target node name
            condition: Optional condition function. Edge is followed only if returns True
            
        Returns:
            Self for chaining
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' not found")
        
        edge = Edge(source=source, target=target, condition=condition)
        self.edges.append(edge)
        
        cond_str = " (conditional)" if condition else ""
        self.logger.info(f"Added edge: {source} -> {target}{cond_str}")
        return self
    
    def add_conditional_edges(
        self,
        source: str,
        routing_func: Callable[[Context], str]
    ) -> 'AgentGraph':
        """Add conditional edges from a source node using a routing function.
        
        This is a convenience method for adding multiple conditional edges.
        
        Args:
            source: Source node name
            routing_func: Function that takes context and returns target node name
            
        Returns:
            Self for chaining
            
        Example:
            def router(ctx):
                if "urgent" in ctx.get_latest_message().content:
                    return "urgent_handler"
                return "normal_handler"
            
            graph.add_conditional_edges("entry", router)
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' not found")
        
        # Store the routing function in the graph
        if not hasattr(self, '_routing_funcs'):
            self._routing_funcs = {}
        self._routing_funcs[source] = routing_func
        
        self.logger.info(f"Added conditional routing from: {source}")
        return self
    
    def set_entry_point(self, node_name: str) -> 'AgentGraph':
        """Set the entry point of the graph.
        
        Args:
            node_name: Name of the entry node
            
        Returns:
            Self for chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Entry point node '{node_name}' not found")
        
        self.entry_point = node_name
        self.logger.info(f"Set entry point: {node_name}")
        return self
    
    def set_finish_point(self, node_name: str) -> 'AgentGraph':
        """Set a finish point of the graph.
        
        Args:
            node_name: Name of a finish node
            
        Returns:
            Self for chaining
        """
        if node_name not in self.nodes:
            raise ValueError(f"Finish point node '{node_name}' not found")
        
        if node_name not in self.finish_points:
            self.finish_points.append(node_name)
            self.logger.info(f"Set finish point: {node_name}")
        return self
    
    def _get_next_nodes(self, current_node: str, context: Context) -> List[str]:
        """Get the next nodes to execute based on edges and conditions.
        
        Args:
            current_node: Current node name
            context: Execution context
            
        Returns:
            List of next node names
        """
        # Check if there's a routing function for this node
        if hasattr(self, '_routing_funcs') and current_node in self._routing_funcs:
            routing_func = self._routing_funcs[current_node]
            try:
                next_node = routing_func(context)
                if next_node in self.nodes:
                    return [next_node]
                else:
                    self.logger.warning(f"Routing function returned unknown node: {next_node}")
                    return []
            except Exception as e:
                self.logger.error(f"Error in routing function: {e}")
                return []
        
        # Otherwise, follow edges
        next_nodes = []
        for edge in self.edges:
            if edge.source == current_node and edge.should_follow(context):
                next_nodes.append(edge.target)
        
        return next_nodes
    
    def run(
        self,
        input_message: str,
        initial_state: Optional[Dict[str, Any]] = None
    ) -> Message:
        """Run the graph.
        
        Args:
            input_message: Initial input message
            initial_state: Initial state dictionary
            
        Returns:
            Final message from execution
        """
        if self.entry_point is None:
            raise ValueError("Entry point not set. Call set_entry_point() first.")
        
        if not self.finish_points:
            raise ValueError("No finish points set. Call set_finish_point() at least once.")
        
        # Initialize context
        context = Context()
        if initial_state:
            context.state = initial_state.copy()
        
        initial_msg = Message(
            content=input_message,
            message_type=MessageType.USER_INPUT
        )
        context.add_message(initial_msg)
        
        self.logger.info(f"Starting graph execution")
        self.logger.info(f"Entry: {self.entry_point}, Finish: {self.finish_points}")
        self.logger.info(f"Input: {input_message[:100]}...")
        
        # Execute graph
        current_node = self.entry_point
        step = 0
        
        while step < self.max_steps:
            step += 1
            self.logger.info(f"Step {step}: Executing node '{current_node}'")
            
            # Execute current node
            node = self.nodes[current_node]
            result = node(context)
            
            # Check for errors
            if result.message_type == MessageType.ERROR:
                self.logger.error(f"Node '{current_node}' returned error: {result.content}")
                return result
            
            # Check if we've reached a finish point
            if current_node in self.finish_points:
                self.logger.info(f"Reached finish point: {current_node}")
                return result
            
            # Get next nodes
            next_nodes = self._get_next_nodes(current_node, context)
            
            if not next_nodes:
                self.logger.info(f"No outgoing edges from '{current_node}', treating as finish point")
                return result
            
            if len(next_nodes) > 1:
                self.logger.warning(f"Multiple next nodes from '{current_node}': {next_nodes}. Taking first one.")
            
            current_node = next_nodes[0]
        
        # Max steps reached
        self.logger.warning(f"Max steps ({self.max_steps}) reached")
        final_result = context.get_latest_message()
        return final_result


def create_simple_chain(*nodes: tuple[str, WorkflowNode]) -> AgentGraph:
    """Create a simple sequential chain of nodes.
    
    This is a convenience function for creating linear workflows quickly.
    
    Args:
        *nodes: Tuples of (name, node)
        
    Returns:
        Configured AgentGraph
        
    Example:
        graph = create_simple_chain(
            ("search", search_agent),
            ("analyze", analyze_agent),
            ("summarize", summary_agent)
        )
        result = graph.run("Your query")
    """
    graph = AgentGraph()
    
    for i, (name, node) in enumerate(nodes):
        graph.add_node(name, node)
        
        if i > 0:
            prev_name = nodes[i-1][0]
            graph.add_edge(prev_name, name)
    
    if nodes:
        graph.set_entry_point(nodes[0][0])
        graph.set_finish_point(nodes[-1][0])
    
    return graph

