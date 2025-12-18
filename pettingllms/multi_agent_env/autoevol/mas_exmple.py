"""
Generated Workflow Code
Question: What was the motto of the Olympics that had Fuwa as the mascots?

Your response should be in the following format:
Explanation: {your explanation for your final answer}
Exact Answer: {your succinct, final answer}
Confidence: {your confidence score between 0% and 100% for your answer}
Generated at: 2025-12-13 15:37:29
"""

import sys
import os

# Add parent directory to path so we can import workflow modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from autogen import ConversableAgent
from autogen.agentchat.group.patterns import AutoPattern
from autogen.agentchat import initiate_group_chat
from ag2_tools import register_search_tools
from ag2_tracer import get_global_tracker

# Get question from environment or use fallback
question = os.getenv("WORKFLOW_QUESTION", "What was the motto of the Olympics that had Fuwa as the mascots?")

# LLM configuration
llm_config = {
    "config_list": [{
        "model": os.getenv("CHAT_MODEL", "gpt-4"),
        "api_key": os.getenv("API_KEY"),
        "base_url": os.getenv("API_BASE"),
    }],
    "temperature": 0.3,
}

# Create agents
executor = ConversableAgent(
    name="Executor",
    llm_config=llm_config,
    system_message=(
        "You answer factual questions accurately. "
        "Use web search tools if needed to verify information. "
        "Provide clear, concise answers with citations when using web search. "
        "For the Olympics question: Identify which Olympics used Fuwa mascots, "
        "then state the official motto of those Olympics."
    ),
    description="Executes tasks and provides answers using web search when needed."
)

critic = ConversableAgent(
    name="Critic",
    llm_config=llm_config,
    system_message=(
        "Critique answers for accuracy, completeness, and evidence. "
        "Check if the answer correctly identifies: "
        "1. Which Olympics used Fuwa mascots (year, location) "
        "2. The official motto of those Olympics "
        "3. Proper citation of sources if applicable "
        "Request clarification or correction if any information is missing or questionable."
    ),
    description="Reviews and critiques answers for quality improvement."
)

# Register search tools for both agents
register_search_tools(executor)
register_search_tools(critic)

# Register agents with tracer
tracker = get_global_tracker()
for agent in [executor, critic]:
    tracker.register_agent(agent)

# Custom summary method to extract final answer
def get_last_non_empty_summary(sender, recipient, summary_args):
    """Get the last non-empty assistant message as summary."""
    chat_history = recipient.chat_messages.get(sender, [])
    for msg in reversed(chat_history):
        if isinstance(msg, dict) and msg.get("content") and msg.get("role") == "assistant":
            return msg["content"]
    return "No response generated"

# Create user agent for termination
user = ConversableAgent(name="user", human_input_mode="NEVER")

# Create AutoPattern for executor-reflection workflow
pattern = AutoPattern(
    initial_agent=executor,
    agents=[executor, critic],
    group_manager_args={
        "llm_config": llm_config,
        "is_termination_msg": lambda x: x.get("name") == "user" and not x.get("content"),
    },
    user_agent=user,
    summary_method=get_last_non_empty_summary
)

# Execute the group chat
result, context_variables, last_agent = initiate_group_chat(
    pattern=pattern,
    messages=question,
    max_rounds=10
)

# Output the result
print("WORKFLOW_SUMMARY_START")
print(result.summary)
print("WORKFLOW_SUMMARY_END")
