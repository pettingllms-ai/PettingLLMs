"""
Conversation Logger for ShareGPT format

This module handles logging conversations in ShareGPT format for training data collection.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

# Global lock for thread-safe file writing
_file_write_lock = threading.Lock()


class ConversationRole(Enum):
    """Roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ConversationLogger:
    """Logger for conversations in ShareGPT format."""
    
    def __init__(self, save_dir: str = "data"):
        """
        Initialize conversation logger.
        
        Args:
            save_dir: Directory to save conversation logs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.conversations = []
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation.
        
        Args:
            role: Role of the speaker (system/user/assistant/tool)
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
    
    def save_to_jsonl(self, filepath: str, conversation_id: Optional[str] = None, append: bool = True):
        """
        Save conversation to JSONL file (thread-safe).
        
        Args:
            filepath: Path to the JSONL file
            conversation_id: Unique ID for this conversation
            append: Whether to append to existing file
        """
        data = self.to_sharegpt_format(conversation_id)
        
        mode = 'a' if append else 'w'
        # Use lock for thread-safe file writing
        with _file_write_lock:
            with open(filepath, mode, encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def get_conversations(self) -> List[Dict[str, str]]:
        """Get all conversations."""
        return self.conversations.copy()


class AgentConversationTracker:
    """Track conversations for multiple agents."""
    
    def __init__(self, save_dir: str = "data"):
        """
        Initialize agent conversation tracker.
        
        Args:
            save_dir: Directory to save conversation logs
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.agent_loggers: Dict[str, ConversationLogger] = {}
    
    def get_logger(self, agent_name: str) -> ConversationLogger:
        """
        Get or create a logger for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            ConversationLogger for the agent
        """
        if agent_name not in self.agent_loggers:
            self.agent_loggers[agent_name] = ConversationLogger(self.save_dir)
        return self.agent_loggers[agent_name]
    
    def save_all_agents(self, filename_prefix: str = "executors", filepath: Optional[str] = None):
        """
        Save all agent conversations to a single JSONL file.

        Args:
            filename_prefix: Prefix for the filename (used if filepath not provided)
            filepath: Optional full path to the output file. If provided, overrides filename_prefix.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if filepath:
            # Use provided filepath directly
            filepath = Path(filepath)
        else:
            # Generate filepath with timestamp
            filepath = self.save_dir / f"{filename_prefix}_{timestamp}.jsonl"

        for agent_name, logger in self.agent_loggers.items():
            if logger.conversations:  # Only save if there are conversations
                conversation_id = f"{agent_name}_{timestamp}"
                logger.save_to_jsonl(str(filepath), conversation_id, append=True)
    
    def save_agent(self, agent_name: str, filename: Optional[str] = None):
        """
        Save a specific agent's conversation.
        
        Args:
            agent_name: Name of the agent
            filename: Optional filename (default: agent_name_timestamp.jsonl)
        """
        if agent_name not in self.agent_loggers:
            return
        
        logger = self.agent_loggers[agent_name]
        if not logger.conversations:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{agent_name}_{timestamp}.jsonl"
        
        filepath = self.save_dir / filename
        conversation_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        logger.save_to_jsonl(filepath, conversation_id, append=True)
    
    def clear_all(self):
        """Clear all agent loggers."""
        for logger in self.agent_loggers.values():
            logger.clear()
    
    def get_all_conversations(self) -> Dict[str, List[Dict[str, str]]]:
        """Get all conversations from all agents."""
        return {
            agent_name: logger.get_conversations()
            for agent_name, logger in self.agent_loggers.items()
        }


def save_designer_conversation(
    messages: List[Dict[str, str]],
    save_dir: str = "data",
    filename_prefix: str = "designer",
    replace_user_prompt: Optional[str] = None,
    append: bool = False
) -> str:
    """
    Save designer (code generator) conversation to JSONL.

    Args:
        messages: List of messages in OpenAI format
        save_dir: Directory to save the file
        filename_prefix: Prefix for the filename
        replace_user_prompt: If provided, replace the user message content with this
                           (useful for SFT data where you want simplified prompts)
        append: If True, append to existing file; if False, create timestamped file

    Returns:
        Path to the saved file
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine filepath
    if append:
        # Append mode: use fixed filename (e.g., designer.jsonl)
        filepath = save_path / f"{filename_prefix}.jsonl"
    else:
        # New file mode: use timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = save_path / f"{filename_prefix}_{timestamp}.jsonl"

    # Convert OpenAI format to ShareGPT format
    logger = ConversationLogger(save_dir)

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Replace user prompt if specified (for SFT data collection)
        if replace_user_prompt is not None and role == "user":
            content = replace_user_prompt

        # Map OpenAI roles to ShareGPT roles
        if role == "system":
            sharegpt_role = "system"
        elif role == "user":
            sharegpt_role = "user"
        elif role == "assistant":
            sharegpt_role = "assistant"
        else:
            sharegpt_role = "user"  # Default to user

        logger.add_message(sharegpt_role, content)

    # Generate conversation ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    conversation_id = f"designer_{timestamp}"

    # Save to file
    logger.save_to_jsonl(filepath, conversation_id, append=append)

    return str(filepath)


def load_conversations_from_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load conversations from JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations


# Global tracker instance
_global_tracker: Optional[AgentConversationTracker] = None


def get_global_tracker(save_dir: str = "data") -> AgentConversationTracker:
    """Get or create the global conversation tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = AgentConversationTracker(save_dir)
    return _global_tracker


def reset_global_tracker():
    """Reset the global conversation tracker."""
    global _global_tracker
    _global_tracker = None
