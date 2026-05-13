"""
Core abstractions for the workflow system.

This module provides the foundational classes for building robust, composable workflows:
- Message: Structured message passing between nodes
- Context: Workflow execution context
- WorkflowNode: Base class for all workflow components
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import json


class MessageType(Enum):
    """Types of messages that can flow through the workflow."""
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    INTERMEDIATE = "intermediate"
    FINAL_RESULT = "final_result"
    ERROR = "error"


@dataclass
class Message:
    """Structured message for communication between workflow nodes.
    
    Eliminates the need for string parsing by providing a structured format.
    """
    content: Any
    message_type: MessageType
    metadata: Dict[str, Any] = field(default_factory=dict)
    sender: Optional[str] = None
    recipient: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "content": self.content,
            "type": self.message_type.value,
            "metadata": self.metadata,
            "sender": self.sender,
            "recipient": self.recipient
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary format."""
        return cls(
            content=data["content"],
            message_type=MessageType(data["type"]),
            metadata=data.get("metadata", {}),
            sender=data.get("sender"),
            recipient=data.get("recipient")
        )


@dataclass
class Context:
    """Execution context for workflow.
    
    Tracks the state and history of workflow execution.
    """
    messages: List[Message] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the context history."""
        self.messages.append(message)
    
    def get_messages_by_type(self, message_type: MessageType) -> List[Message]:
        """Get all messages of a specific type."""
        return [msg for msg in self.messages if msg.message_type == message_type]
    
    def get_latest_message(self, message_type: Optional[MessageType] = None) -> Optional[Message]:
        """Get the latest message, optionally filtered by type."""
        if message_type is None:
            return self.messages[-1] if self.messages else None
        
        for msg in reversed(self.messages):
            if msg.message_type == message_type:
                return msg
        return None
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state variable."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state variable."""
        return self.state.get(key, default)


class WorkflowNode(ABC):
    """Base class for all workflow nodes.
    
    A node is a unit of processing in the workflow. It can be:
    - An agent that performs reasoning and actions
    - A combiner (ensemble, debate, etc.)
    - A transformer (reflection, refinement, etc.)
    - A controller (conditional branching, loops, etc.)
    """
    
    def __init__(self, name: str, **kwargs):
        """Initialize the workflow node.
        
        Args:
            name: Unique name for this node
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for this node."""
        logger = logging.getLogger(f"{self.__class__.__name__}.{self.name}")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                f'[%(asctime)s] {self.name} - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    @abstractmethod
    def process(self, context: Context) -> Message:
        """Process the current context and return a message.
        
        Args:
            context: The workflow execution context
            
        Returns:
            Message: The result of processing
        """
        pass
    
    def __call__(self, context: Context) -> Message:
        """Make the node callable. Adds logging and error handling.
        
        Args:
            context: The workflow execution context
            
        Returns:
            Message: The result of processing
        """
        try:
            self.logger.info(f"Starting processing")
            result = self.process(context)
            self.logger.info(f"Finished processing")
            
            # Add result to context
            result.sender = self.name
            context.add_message(result)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}", exc_info=True)
            error_msg = Message(
                content={"error": str(e), "node": self.name},
                message_type=MessageType.ERROR,
                sender=self.name,
                metadata={"exception_type": type(e).__name__}
            )
            context.add_message(error_msg)
            return error_msg


class ToolRegistry:
    """Registry for tools that can be used by agents.
    
    Provides a clean interface for tool management without string parsing.
    """
    
    def __init__(self):
        self._tools = {}
    
    def register(self, name: str, func: callable, description: str, parameters: Dict[str, Any]) -> None:
        """Register a tool.
        
        Args:
            name: Tool name
            func: The callable function
            description: Tool description
            parameters: JSON schema for parameters
        """
        self._tools[name] = {
            "function": func,
            "description": description,
            "parameters": parameters
        }
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool schemas for all registered tools."""
        schemas = []
        for name, tool in self._tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return schemas
    
    def call_tool(self, name: str, parameters: Dict[str, Any]) -> Any:
        """Call a tool with given parameters.
        
        Args:
            name: Tool name
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        
        return tool["function"](**parameters)

