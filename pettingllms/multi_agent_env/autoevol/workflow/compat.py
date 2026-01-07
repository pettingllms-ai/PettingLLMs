"""
Compatibility layer for migrating from old BaseAgent to new workflow system.

This module provides wrapper classes that maintain the old API while using
the new workflow system internally.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import List, Dict, Any, Optional
from workflow.core import ToolRegistry, Context, Message, MessageType
from workflow.nodes import AgentNode
from utils.environments.search_env import SearchEnvironment


class CompatBaseAgent:
    """Compatibility wrapper for BaseAgent using new workflow system.
    
    This class maintains the old BaseAgent API while using the new workflow
    system internally, making migration easier.
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: List[str],
        max_turns: int = 10
    ):
        """Initialize compatible base agent.
        
        Args:
            name: Agent name
            system_prompt: System prompt
            tools: List of tool names (e.g., ["google-search", "fetch_data"])
            max_turns: Maximum turns for tool calling
        """
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_turns = max_turns
        
        # Setup tool registry
        self.tool_registry = self._setup_tools()
        
        # Create internal agent node
        self.agent_node = AgentNode(
            name=name,
            system_prompt=system_prompt,
            tool_registry=self.tool_registry,
            max_turns=max_turns
        )
        
        # Memory for maintaining conversation
        self.memory = None
    
    def _setup_tools(self) -> ToolRegistry:
        """Setup tools based on tool names."""
        registry = ToolRegistry()
        
        # Initialize search environment if needed
        search_env = None
        if "google-search" in self.tools or "fetch_data" in self.tools:
            search_env = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
        
        # Register tools
        if "google-search" in self.tools and search_env:
            registry.register(
                name="google-search",
                func=search_env.search,
                description="Search the web using Google",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            )
        
        if "fetch_data" in self.tools and search_env:
            registry.register(
                name="fetch_data",
                func=search_env.fetch,
                description="Fetch content from a URL",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to fetch"
                        }
                    },
                    "required": ["url"]
                }
            )
        
        return registry
    
    def set_memory(self, memory: List[Dict[str, Any]]) -> None:
        """Set conversation memory.
        
        Args:
            memory: List of message dicts
        """
        self.memory = memory
    
    def run(self, input: str) -> str:
        """Run the agent (compatible with old API).
        
        Args:
            input: Input text
            
        Returns:
            Agent response as string
        """
        # Create context
        context = Context()
        
        # Add input message
        input_msg = Message(
            content=input,
            message_type=MessageType.USER_INPUT
        )
        context.add_message(input_msg)
        
        # Run agent
        result = self.agent_node(context)
        
        # Return content as string (for compatibility)
        if isinstance(result.content, str):
            return result.content
        else:
            return str(result.content)


class CompatBaseWorkflow:
    """Compatibility wrapper for BaseWorkFlow using new workflow system.
    
    This provides a compatible interface for the old workflow system while
    using the new implementation.
    """
    
    def __init__(self):
        """Initialize compatible workflow."""
        from workflow.workflow import Workflow
        
        # Setup agents (matching old BaseWorkFlow)
        self.agents = {
            "CaptionAgent": CompatBaseAgent(
                name="CaptionAgent",
                system_prompt=(
                    "You are a Caption Agent that can coordinate with other agents. "
                    "You can search for information and summarize content. "
                    "Provide clear, well-researched answers."
                ),
                tools=["google-search", "fetch_data"]
            ),
            "SearchAgent": CompatBaseAgent(
                name="SearchAgent",
                system_prompt=(
                    "You are a Search Agent. Use google search to find information "
                    "and provide valuable information and helpful links."
                ),
                tools=["google-search"]
            ),
            "SummarizeAgent": CompatBaseAgent(
                name="SummarizeAgent",
                system_prompt=(
                    "You are a Web Content Summarize Agent. Fetch web content from "
                    "given URLs and provide clear summaries."
                ),
                tools=["fetch_data"]
            )
        }
        
        self.workflow = Workflow(name="compat_workflow")
        self.max_turns = 10
    
    def run_workflow(self, input: str) -> str:
        """Run workflow (compatible with old API).
        
        Args:
            input: Input question
            
        Returns:
            Final result as string
        """
        # For simple compatibility, just use CaptionAgent
        # In the new system, you'd compose agents differently
        result = self.agents["CaptionAgent"].run(input)
        return result


def migrate_to_new_system():
    """
    Guide for migrating from old system to new system.
    
    OLD SYSTEM:
    -----------
    from BaseAgent import BaseAgent
    from BaseWorkFlow import BaseWorkFlow
    
    agent = BaseAgent(
        name="MyAgent",
        system_prompt="...",
        tools=["google-search"]
    )
    
    workflow = BaseWorkFlow()
    result = workflow.run_workflow("question")
    
    
    NEW SYSTEM (Recommended):
    -------------------------
    from workflow.core import ToolRegistry
    from workflow.nodes import AgentNode
    from workflow.workflow import Workflow
    from utils.environments.search_env import SearchEnvironment
    
    # Setup tools
    search_env = SearchEnvironment()
    tool_registry = ToolRegistry()
    tool_registry.register("google-search", search_env.search, ...)
    
    # Create agent
    agent = AgentNode(
        name="MyAgent",
        system_prompt="...",
        tool_registry=tool_registry
    )
    
    # Create workflow
    workflow = Workflow()
    workflow.add_node(agent)
    
    # Run
    result = workflow.run("question")
    print(result.content)
    
    
    COMPATIBILITY LAYER (For gradual migration):
    --------------------------------------------
    from workflow.compat import CompatBaseAgent, CompatBaseWorkflow
    
    # Drop-in replacement for old BaseAgent
    agent = CompatBaseAgent(
        name="MyAgent",
        system_prompt="...",
        tools=["google-search"]
    )
    
    # Drop-in replacement for old BaseWorkFlow
    workflow = CompatBaseWorkflow()
    result = workflow.run_workflow("question")
    """
    pass

