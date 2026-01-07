"""Router node for conditional branching in workflows."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, Callable, Any
from workflow.core import WorkflowNode, Context, Message, MessageType


class RouterNode(WorkflowNode):
    """Router node that conditionally selects next node based on context.
    
    Enables dynamic workflow branching based on:
    - Message content
    - Context state
    - Custom routing logic
    """
    
    def __init__(
        self,
        name: str,
        routes: Dict[str, WorkflowNode],
        routing_func: Callable[[Context], str],
        default_route: str = "default",
        **kwargs
    ):
        """Initialize the router node.
        
        Args:
            name: Node name
            routes: Dictionary mapping route names to nodes
            routing_func: Function that takes context and returns route name
            default_route: Default route if routing_func returns unknown route
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        
        self.routes = routes
        self.routing_func = routing_func
        self.default_route = default_route
        
        if default_route not in routes:
            raise ValueError(f"Default route '{default_route}' not found in routes")
    
    def process(self, context: Context) -> Message:
        """Route to appropriate node based on context.
        
        Args:
            context: Workflow context
            
        Returns:
            Message from the selected route
        """
        # Determine route
        try:
            route_name = self.routing_func(context)
            self.logger.info(f"Routing to: {route_name}")
        except Exception as e:
            self.logger.error(f"Error in routing function: {e}")
            route_name = self.default_route
            self.logger.info(f"Using default route: {route_name}")
        
        # Get target node
        if route_name not in self.routes:
            self.logger.warning(f"Route '{route_name}' not found, using default")
            route_name = self.default_route
        
        target_node = self.routes[route_name]
        
        # Execute target node
        result = target_node(context)
        
        # Add routing metadata
        result.metadata["routed_from"] = self.name
        result.metadata["route_taken"] = route_name
        
        return result


def create_keyword_router(
    name: str,
    keyword_routes: Dict[str, WorkflowNode],
    default_node: WorkflowNode,
    case_sensitive: bool = False
) -> RouterNode:
    """Create a router that routes based on keywords in the latest message.
    
    Args:
        name: Router name
        keyword_routes: Dictionary mapping keywords to nodes
        default_node: Node to use if no keyword matches
        case_sensitive: Whether keyword matching is case sensitive
        
    Returns:
        Configured RouterNode
    """
    def routing_func(context: Context) -> str:
        latest_msg = context.get_latest_message()
        if not latest_msg:
            return "default"
        
        content = str(latest_msg.content)
        if not case_sensitive:
            content = content.lower()
        
        for keyword, _ in keyword_routes.items():
            check_keyword = keyword if case_sensitive else keyword.lower()
            if check_keyword in content:
                return keyword
        
        return "default"
    
    routes = dict(keyword_routes)
    routes["default"] = default_node
    
    return RouterNode(
        name=name,
        routes=routes,
        routing_func=routing_func,
        default_route="default"
    )

