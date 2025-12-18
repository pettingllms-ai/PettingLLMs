"""
Message Tracker

Tracks all messages in AG2 conversations by hooking into agent communication.
"""

from autogen import ConversableAgent
from typing import Dict, List, Any, Optional
from datetime import datetime
import json


class MessageTracker:
    """Track all messages in AG2 conversations."""

    def __init__(self):
        """Initialize the message tracker."""
        self.agent_conversations: Dict[str, List[Dict[str, Any]]] = {}
        self._registered_agents = set()
        self._agent_system_messages: Dict[str, str] = {}  # Store system prompts
        self._token_ids_store: Dict[str, List[List[int]]] = {}  # Store token IDs for each agent's responses

    def _extract_tool_info(self, agent: ConversableAgent) -> str:
        """
        Extract tool information from agent's llm_config and format it as a string.

        Args:
            agent: The ConversableAgent

        Returns:
            Formatted string describing the tools available to the agent
        """
        tool_info = ""

        # Get llm_config from agent
        llm_config = getattr(agent, 'llm_config', None)
        if not llm_config or llm_config is False:
            return tool_info

        # Check for tools in llm_config
        tools = llm_config.get('tools', [])
        if not tools:
            return tool_info

        # Format tool information
        tool_descriptions = []
        tool_descriptions.append("\n\n## Available Tools\n")
        tool_descriptions.append("You have access to the following tools:\n")

        for tool in tools:
            if tool.get('type') == 'function':
                func = tool.get('function', {})
                name = func.get('name', 'unknown')
                description = func.get('description', 'No description')
                parameters = func.get('parameters', {})

                tool_descriptions.append(f"\n### {name}")
                tool_descriptions.append(f"Description: {description}")

                # Format parameters
                if parameters and parameters.get('properties'):
                    tool_descriptions.append("Parameters:")
                    for param_name, param_info in parameters.get('properties', {}).items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        required = param_name in parameters.get('required', [])
                        req_str = " (required)" if required else " (optional)"
                        tool_descriptions.append(f"  - {param_name}: {param_type}{req_str} - {param_desc}")

        # IMPORTANT:
        # Do NOT instruct models to emit custom <tool_call> tags in plain text.
        # Autogen's OpenAI-compatible clients will send tool schemas in the request and expect
        # the model to return structured tool calls (tool_calls). Mixing text-based tool calls
        # with role="tool" messages can trigger API errors like:
        # "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'".
        tool_descriptions.append("\n\n## Tool Calling")
        tool_descriptions.append(
            "Call tools using the model's native tool-calling mechanism when needed. "
            "You do NOT need to print any special <tool_call> tags or JSON manually."
        )

        tool_info = "\n".join(tool_descriptions)
        return tool_info

    def register_agent(self, agent: ConversableAgent):
        """
        Hook into agent's send/receive methods to track messages.

        Args:
            agent: The ConversableAgent to track
        """
        if agent.name in self._registered_agents:
            return  # Already registered

        agent_name = agent.name
        self.agent_conversations[agent_name] = []
        self._token_ids_store[agent_name] = []  # Initialize token IDs list for this agent

        # Store system message if available, with tool information appended
        base_system_message = ""
        if hasattr(agent, 'system_message') and agent.system_message:
            base_system_message = agent.system_message

        # Extract and append tool information
        tool_info = self._extract_tool_info(agent)
        if tool_info:
            full_system_message = base_system_message + tool_info
            self._agent_system_messages[agent_name] = full_system_message
        elif base_system_message:
            self._agent_system_messages[agent_name] = base_system_message

        # Store original methods
        original_send = agent.send
        original_receive = agent.receive

        # Create tracked version of send
        def tracked_send(message, recipient, request_reply=True, silent=False):
            self._log_message(
                agent_name=agent_name,
                direction="send",
                message=message,
                other_party=recipient.name
            )
            return original_send(message, recipient, request_reply, silent)

        # Create tracked version of receive
        def tracked_receive(message, sender, request_reply=None, silent=False):
            self._log_message(
                agent_name=agent_name,
                direction="receive",
                message=message,
                other_party=sender.name
            )
            return original_receive(message, sender, request_reply, silent)

        # Replace methods
        agent.send = tracked_send
        agent.receive = tracked_receive

        self._registered_agents.add(agent_name)
        print(f"✓ Message tracking enabled for {agent_name}")

    def add_token_ids(self, agent_name: str, token_ids: List[int]):
        """
        Add token IDs for an agent's response.

        Args:
            agent_name: Name of the agent
            token_ids: List of token IDs for the response
        """
        if agent_name not in self._token_ids_store:
            self._token_ids_store[agent_name] = []
        self._token_ids_store[agent_name].append(token_ids)

    def _log_message(
        self,
        agent_name: str,
        direction: str,
        message: Any,
        other_party: str
    ):
        """
        Log a message.

        Args:
            agent_name: Name of the agent
            direction: "send" or "receive"
            message: The message content
            other_party: The other party in communication
        """
        # Extract content from message
        if isinstance(message, dict):
            content = message.get("content", str(message))
            role = message.get("role", "user")
        elif isinstance(message, str):
            content = message
            role = "user" if direction == "receive" else "assistant"
        else:
            content = str(message)
            role = "user"

        self.agent_conversations[agent_name].append({
            "direction": direction,
            "content": content,
            "role": role,
            "other_party": other_party,
            "timestamp": datetime.now().isoformat(),
            "raw_message": message if isinstance(message, dict) else None
        })

    def _format_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """
        Format a single tool call, removing unnecessary fields.

        Args:
            tool_call: The raw tool call dict from AG2

        Returns:
            Formatted tool call JSON string
        """
        # Extract only the necessary fields: function name and arguments
        formatted = {
            "function": {
                "name": tool_call.get("function", {}).get("name", ""),
                "arguments": tool_call.get("function", {}).get("arguments", "{}")
            }
        }

        # Parse arguments if it's a string (to ensure proper encoding)
        if isinstance(formatted["function"]["arguments"], str):
            try:
                # Parse and re-encode to ensure proper unicode handling
                args_dict = json.loads(formatted["function"]["arguments"])
                formatted["function"]["arguments"] = args_dict
            except json.JSONDecodeError:
                # Keep as string if parsing fails
                pass

        # Use ensure_ascii=False to properly encode Chinese characters
        return json.dumps(formatted, ensure_ascii=False)

    def get_sharegpt_format(self, agent_name: str, include_token_ids: bool = True) -> Dict[str, Any]:
        """
        Convert agent's conversation to ShareGPT format.

        Args:
            agent_name: Name of the agent
            include_token_ids: If True, include token_ids in the output

        Returns:
            ShareGPT formatted conversation
        """
        conversations = []

        # Add system message first if available
        if agent_name in self._agent_system_messages:
            conversations.append({
                "from": "system",
                "value": self._agent_system_messages[agent_name]
            })

        response_idx = 0  # Track index of assistant responses for token_ids mapping
        for msg in self.agent_conversations.get(agent_name, []):
            # Determine role based on direction and message content
            if msg["direction"] == "send":
                # Agent sending = assistant/gpt
                role = "gpt"
            else:
                # Agent receiving - check if from another agent or user
                # If the other_party is a registered agent, and the message has assistant role, it's from an agent
                if msg.get("raw_message") and isinstance(msg["raw_message"], dict):
                    msg_role = msg["raw_message"].get("role", "user")
                    # If message role is assistant, it's from another agent
                    if msg_role == "assistant":
                        role = "gpt"  # Message from another agent
                    elif msg_role == "tool":
                        # Tool response should be treated as human input
                        role = "human"
                    else:
                        role = "human"
                else:
                    # Check if other_party is a registered agent
                    if msg["other_party"] in self._registered_agents:
                        # Message from another agent - but we're receiving it, so it's input to us
                        role = "human"
                    else:
                        role = "human"

            # Handle tool calls - format each tool call separately
            content = msg["content"]
            if msg.get("raw_message") and isinstance(msg["raw_message"], dict):
                if "tool_calls" in msg["raw_message"]:
                    tool_calls = msg["raw_message"]["tool_calls"]
                    # Format each tool call as a separate <tool_call> block
                    for tool_call in tool_calls:
                        formatted_call = self._format_tool_call(tool_call)
                        content += f"\n<tool_call>{formatted_call}</tool_call>"

            conv_entry = {
                "from": role,
                "value": content
            }

            # Add token_ids if this is a GPT response and we have token_ids stored
            if role == "gpt" and include_token_ids:
                token_ids_list = self._token_ids_store.get(agent_name, [])
                if response_idx < len(token_ids_list):
                    conv_entry["token_ids"] = token_ids_list[response_idx]
                response_idx += 1

            conversations.append(conv_entry)

        return {
            "id": f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "conversations": conversations
        }

    def save_all(self, filepath: str, append: bool = True, include_token_ids: bool = True):
        """
        Save all tracked conversations to JSONL.

        Args:
            filepath: Output file path
            append: If True, append to existing file; if False, overwrite (default: True)
            include_token_ids: If True, include token_ids in the output (default: True)
        """
        mode = 'a' if append else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            for agent_name in self.agent_conversations:
                if self.agent_conversations[agent_name]:  # Only save if has messages
                    data = self.get_sharegpt_format(agent_name, include_token_ids=include_token_ids)
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

        action = "Appended" if append else "Saved"
        print(f"✓ {action} {len(self.agent_conversations)} agent conversations to {filepath}")
        if include_token_ids:
            total_responses = sum(len(ids) for ids in self._token_ids_store.values())
            print(f"✓ Included token_ids for {total_responses} responses")

    def get_all_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all tracked conversations."""
        return self.agent_conversations.copy()


# Global tracker instance
_global_tracker = None


def get_global_tracker(reset: bool = False) -> MessageTracker:
    """
    Get or create the global message tracker.

    Args:
        reset: If True, create a new tracker

    Returns:
        The global MessageTracker instance
    """
    global _global_tracker

    if _global_tracker is None or reset:
        _global_tracker = MessageTracker()

    return _global_tracker


def reset_global_tracker():
    """
    Reset the global message tracker to None.

    This is useful when processing multiple questions in batch to ensure
    each question gets a fresh tracker.
    """
    global _global_tracker
    _global_tracker = None
