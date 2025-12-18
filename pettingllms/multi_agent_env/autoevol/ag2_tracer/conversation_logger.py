"""
Conversation Logger

ShareGPT format conversation logging for training data collection.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


class ConversationLogger:
    """Logger for conversations in ShareGPT format."""

    def __init__(self, save_dir: str = "data"):
        """
        Initialize conversation logger.

        Args:
            save_dir: Directory to save conversation logs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.conversations = []

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation.

        Args:
            role: Role of the speaker (system/human/gpt)
            content: Content of the message
        """
        self.conversations.append({
            "from": role,
            "value": content
        })

    def clear(self):
        """Clear the conversation history."""
        self.conversations = []

    def to_sharegpt_format(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert to ShareGPT format.

        Args:
            conversation_id: Unique ID for this conversation

        Returns:
            Dictionary in ShareGPT format
        """
        if conversation_id is None:
            conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        return {
            "id": conversation_id,
            "conversations": self.conversations.copy()
        }

    def save_to_jsonl(
        self,
        filepath: str,
        conversation_id: Optional[str] = None,
        append: bool = True
    ):
        """
        Save conversation to JSONL file.

        Args:
            filepath: Path to the JSONL file
            conversation_id: Unique ID for this conversation
            append: Whether to append to existing file
        """
        data = self.to_sharegpt_format(conversation_id)

        mode = 'a' if append else 'w'
        with open(filepath, mode, encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def get_conversations(self) -> List[Dict[str, str]]:
        """Get all conversations."""
        return self.conversations.copy()


def _format_tool_call(tool_call: Dict[str, Any]) -> str:
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


def trajectory_to_sharegpt(
    trajectory: List[Dict[str, Any]],
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert AG2 message trajectory to ShareGPT format.

    Args:
        trajectory: List of message dictionaries from AG2
        conversation_id: Unique conversation ID

    Returns:
        ShareGPT formatted dictionary
    """
    if conversation_id is None:
        conversation_id = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    conversations = []

    # Add system message if exists
    system_msgs = [m for m in trajectory if m.get("role") == "system"]
    if system_msgs:
        conversations.append({
            "from": "system",
            "value": system_msgs[0].get("content", "")
        })

    # Process user/assistant messages
    for msg in trajectory:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            conversations.append({"from": "user", "value": content})
        elif role == "assistant":
            # Preserve tool calls in content - format each separately
            if msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    formatted_call = _format_tool_call(tool_call)
                    content += f"\n<tool_call>{formatted_call}</tool_call>"
            conversations.append({"from": "assistant", "value": content})
        elif role == "tool":
            # Tool results as human messages (not gpt)
            tool_name = msg.get("name", "tool")
            conversations.append({
                "from": "user",
                "value": f"Tool '{tool_name}' returned: {content}"
            })

    return {
        "id": conversation_id,
        "conversations": conversations
    }


def save_conversations(
    conversations: List[Dict[str, Any]],
    filepath: str,
    append: bool = False
):
    """
    Save multiple conversations to JSONL.

    Args:
        conversations: List of ShareGPT formatted conversations
        filepath: Output file path
        append: Whether to append to existing file
    """
    Path(filepath).parent.mkdir(exist_ok=True, parents=True)

    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
