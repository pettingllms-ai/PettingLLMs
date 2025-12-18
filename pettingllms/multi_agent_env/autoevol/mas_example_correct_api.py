"""
Correct AG2/PyAutogen API example using GroupChat instead of AutoPattern
This example shows the correct way to use AG2 0.7.0+ API
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autogen import ConversableAgent, GroupChat, GroupChatManager
from ag2_tools import register_code_interpreter
from ag2_tracer import get_global_tracker

# Get question from environment or use fallback
question = os.getenv("WORKFLOW_QUESTION", "Solve a math problem")

# LLM configuration
llm_config = {
    "config_list": [{
        "model": os.getenv("CHAT_MODEL", "gpt-4"),
        "api_key": os.getenv("API_KEY", ""),
        "base_url": os.getenv("API_BASE", ""),
    }],
    "temperature": 0.2,
}

# Create agents
solver = ConversableAgent(
    name="Solver",
    llm_config=llm_config,
    system_message="You are a math expert. Solve problems step by step using Python when needed.",
    description="Primary math solver using Python execution."
)

critic = ConversableAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="Review solutions for accuracy and completeness. Request corrections if needed.",
    description="Reviews solutions for errors."
)

# Register code interpreter
register_code_interpreter(solver)

# Register agents with tracer
tracker = get_global_tracker()
for agent in [solver, critic]:
    tracker.register_agent(agent)

# Create user proxy for termination
user_proxy = ConversableAgent(
    name="user",
    human_input_mode="NEVER",
    llm_config=False,
    is_termination_msg=lambda x: x.get("content", "").strip() == ""
)

# Create GroupChat
groupchat = GroupChat(
    agents=[user_proxy, solver, critic],
    messages=[],
    max_round=10,
    speaker_selection_method="auto"
)

# Create GroupChatManager
manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

# Initiate the chat
result = user_proxy.initiate_chat(
    manager,
    message=question
)

# Output the result
print("WORKFLOW_SUMMARY_START")
print(result.summary if hasattr(result, 'summary') else result.chat_history[-1].get("content", "No response"))
print("WORKFLOW_SUMMARY_END")
