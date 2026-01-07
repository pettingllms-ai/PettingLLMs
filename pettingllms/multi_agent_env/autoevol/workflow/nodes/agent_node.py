"""Agent node that wraps an AI agent with tool calling capabilities."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any, Optional
import json
from workflow.core import WorkflowNode, Context, Message, MessageType, ToolRegistry
from utils.BaseOpenAI import AIClient
from utils.conversation_logger import get_global_tracker


class AgentNode(WorkflowNode):
    """A node that wraps an AI agent with tool calling capabilities.
    
    This node:
    - Uses structured message passing instead of string parsing
    - Automatically handles tool calls through ToolRegistry
    - Manages conversation history
    - Provides clean interface for custom agents
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_registry: Optional[ToolRegistry] = None,
        ai_client: Optional[AIClient] = None,
        max_turns: int = 10,
        enable_conversation_logging: bool = True,
        **kwargs
    ):
        """Initialize the agent node.
        
        Args:
            name: Node name
            system_prompt: System prompt for the agent
            tool_registry: Registry of available tools
            ai_client: AI client for making API calls
            max_turns: Maximum number of turns for tool calling
            enable_conversation_logging: Whether to log conversations to ShareGPT format
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)
        
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_turns = max_turns
        self.enable_conversation_logging = enable_conversation_logging

        # Store ai_client if provided, otherwise will use workflow's client
        self.ai_client = ai_client
        self._client_from_workflow = (ai_client is None)
    
    def _build_initial_messages(self, context: Context) -> List[Dict[str, str]]:
        """Build initial message list from context.

        Args:
            context: Workflow context

        Returns:
            List of messages in OpenAI format
        """
        # Check if we need to add tool descriptions to system prompt (for vLLM)
        system_prompt = self.system_prompt

        # In evaluate mode, use prompt-based tool calling instead of native API tools
        if os.getenv("EVALUATE_MODE") == "True" and self.tool_registry.list_tools():
            tools = self.tool_registry.get_tool_schemas()
            system_prompt = (
                self.system_prompt
                + f"\n\nYou have access to the following tools: {json.dumps(tools, indent=2)}.\n"
                "When you need to call a tool, output it in the following format:\n"
                "<tool_call>{\n"
                '  "name": "tool_name",\n'
                '  "parameters": {\n'
                '    "parameter_name": "parameter_value"\n'
                "  }\n"
                "}</tool_call>\n"
            )

        messages = [{"role": "system", "content": system_prompt}]

        # Get the latest user input or intermediate result
        latest_msg = context.get_latest_message()
        if latest_msg:
            if isinstance(latest_msg.content, str):
                content = latest_msg.content
            elif isinstance(latest_msg.content, dict):
                content = json.dumps(latest_msg.content, ensure_ascii=False)
            else:
                content = str(latest_msg.content)

            messages.append({"role": "user", "content": content})

        return messages
    
    def _handle_tool_calls(self, messages: List[Dict[str, str]]) -> tuple[str, int, List[Dict[str, str]]]:
        """Handle agent tool calls in a loop.

        Args:
            messages: Current message history

        Returns:
            Tuple of (final_response, total_tokens, full_messages)
        """
        total_tokens = 0

        # Get tool schemas if available
        tools = None
        evaluate_mode = os.getenv("EVALUATE_MODE") == "True"

        # Only use native tool calling if NOT in evaluate mode (vLLM doesn't support it)
        if not evaluate_mode and self.tool_registry.list_tools():
            tools = self.tool_registry.get_tool_schemas()

        for turn in range(self.max_turns):
            # Make API call
            # In evaluate mode, never pass tools (use prompt-based approach instead)
            if tools and not evaluate_mode:
                response, prompt_tokens, completion_tokens = self.ai_client.chat(
                    messages,
                    tools=tools
                )
            else:
                response, prompt_tokens, completion_tokens = self.ai_client.chat(messages)
            
            total_tokens += prompt_tokens + completion_tokens
            
            # Check if response contains tool calls (native OpenAI format)
            # For now, we support both native and custom format
            
            # Try to parse custom format first (for backward compatibility)
            if "<tool_call>" in response and "</tool_call>" in response:
                # Add assistant's tool call to messages
                messages.append({"role": "assistant", "content": response})
                
                try:
                    tool_call_str = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                    tool_call = json.loads(tool_call_str)
                    tool_name = tool_call["name"]
                    tool_params = tool_call.get("parameters", {})
                    
                    self.logger.info(f"Calling tool: {tool_name} with params: {tool_params}")
                    
                    # Execute tool
                    tool_result = self.tool_registry.call_tool(tool_name, tool_params)
                    
                    # Add tool result to messages
                    tool_response = f"Tool '{tool_name}' returned: {tool_result}"
                    messages.append({
                        "role": "user",
                        "content": tool_response
                    })
                    
                    self.logger.debug(f"Tool response: {tool_response[:200]}...")
                    
                except Exception as e:
                    self.logger.error(f"Error executing tool: {e}")
                    error_msg = f"Error executing tool: {str(e)}"
                    messages.append({
                        "role": "user",
                        "content": error_msg
                    })
                
                # Continue to next turn
                continue
            
            # No tool call in response - this is the final answer
            # Add final response to messages
            messages.append({"role": "assistant", "content": response})
            
            # Return final response and full message history
            return response, total_tokens, messages
        
        # Max turns reached, force final answer
        messages.append({
            "role": "user",
            "content": "Please provide your final answer now without using any tools."
        })
        response, pt, ct = self.ai_client.chat(messages)
        total_tokens += pt + ct
        messages.append({"role": "assistant", "content": response})
        
        return response, total_tokens, messages
    
    def process(self, context: Context) -> Message:
        """Process the context and generate agent response.

        Args:
            context: Workflow context

        Returns:
            Message containing agent response
        """
        workflow = context.state.get('workflow')
        self.ai_client = workflow.ai_client
            
        # Build message history
        initial_messages = self._build_initial_messages(context)
        
        # Handle tool calls and get final response with full message history
        response, total_tokens, full_messages = self._handle_tool_calls(initial_messages)
        
        # Log conversation to ShareGPT format if enabled
        if self.enable_conversation_logging:
            try:
                tracker = get_global_tracker()
                logger = tracker.get_logger(self.name)
                
                # Log all messages in the conversation (including tool calls and responses)
                for msg in full_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    # Map OpenAI roles to ShareGPT format
                    if role == "system":
                        sharegpt_role = "system"
                    elif role == "user":
                        sharegpt_role = "human"
                    elif role == "assistant":
                        sharegpt_role = "gpt"
                    else:
                        sharegpt_role = "human"
                    
                    logger.add_message(sharegpt_role, content)
                
            except Exception as e:
                self.logger.warning(f"Failed to log conversation: {e}")
        
        # Store token usage in metadata
        self.logger.info(f"Total tokens used: {total_tokens}")
        
        # Create result message
        result = Message(
            content=response,
            message_type=MessageType.AGENT_RESPONSE,
            metadata={
                "tokens": total_tokens,
                "agent_name": self.name
            }
        )
        
        return result

