"""
Trajectory Collector

Extract and save complete trajectories from AG2 GroupChat conversations.
"""

from autogen import GroupChat
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path
from .conversation_logger import trajectory_to_sharegpt


def extract_groupchat_trajectory(groupchat: GroupChat) -> List[Dict[str, Any]]:
    """
    Extract full message trajectory from GroupChat.

    Args:
        groupchat: The GroupChat instance

    Returns:
        List of message dictionaries
    """
    trajectory = []

    for msg in groupchat.messages:
        trajectory.append({
            "role": msg.get("role", "user"),
            "name": msg.get("name", "unknown"),
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls", None),
            "function_call": msg.get("function_call", None)
        })

    return trajectory


def save_trajectory(
    groupchat: GroupChat,
    filepath: str,
    conversation_id: Optional[str] = None,
    append: bool = True
):
    """
    Save GroupChat trajectory to JSONL in ShareGPT format.

    Args:
        groupchat: The GroupChat instance
        filepath: Output file path
        conversation_id: Optional conversation ID
        append: Whether to append to existing file
    """
    # Extract trajectory
    trajectory = extract_groupchat_trajectory(groupchat)

    # Convert to ShareGPT format
    sharegpt_data = trajectory_to_sharegpt(trajectory, conversation_id)

    # Ensure directory exists
    Path(filepath).parent.mkdir(exist_ok=True, parents=True)

    # Save to file
    mode = 'a' if append else 'w'
    with open(filepath, mode, encoding='utf-8') as f:
        f.write(json.dumps(sharegpt_data, ensure_ascii=False) + '\n')

    print(f"✓ Saved trajectory ({len(trajectory)} messages) to {filepath}")


def save_multiple_trajectories(
    trajectories: List[List[Dict[str, Any]]],
    filepath: str,
    prefix: str = "traj"
):
    """
    Save multiple trajectories to JSONL.

    Args:
        trajectories: List of trajectory lists
        filepath: Output file path
        prefix: Prefix for conversation IDs
    """
    Path(filepath).parent.mkdir(exist_ok=True, parents=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for i, trajectory in enumerate(trajectories):
            conversation_id = f"{prefix}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            sharegpt_data = trajectory_to_sharegpt(trajectory, conversation_id)
            f.write(json.dumps(sharegpt_data, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(trajectories)} trajectories to {filepath}")
