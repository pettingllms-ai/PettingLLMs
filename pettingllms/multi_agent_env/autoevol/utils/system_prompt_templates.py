"""
System prompt templates for tool-calling agents.

These templates provide clear instructions for models to use tools properly.
"""


def get_tool_calling_system_prompt(
    role_description: str,
    task_description: str,
    tools_info: list[dict],
    specific_instructions: str = ""
) -> str:
    """
    Generate a system prompt that clearly instructs the model to use tools.
    
    Args:
        role_description: What role the agent plays (e.g., "You are a research assistant")
        task_description: What the agent needs to do
        tools_info: List of dicts with 'name', 'params', 'description' for each tool
        specific_instructions: Additional task-specific instructions
    
    Returns:
        Complete system prompt with clear tool usage instructions
    """
    
    # Build tools list
    tools_list = ""
    for i, tool in enumerate(tools_info, 1):
        params_str = ", ".join(tool.get("params", []))
        tools_list += f"{i}. {tool['name']}({params_str}): {tool['description']}\n"
    
    # Complete prompt
    prompt = f"""{role_description}

YOUR TASK: {task_description}

IMPORTANT - YOU MUST USE TOOLS:
You have access to tools that you MUST use to complete this task. Do NOT answer from memory.

TOOL USAGE FORMAT:
To call a tool, respond EXACTLY like this:
<tool_call>{{"name": "tool_name", "parameters": {{"param": "value"}}}}</tool_call>

AVAILABLE TOOLS:
{tools_list}

EXAMPLE CONVERSATION:
Human: What is the capital of France?
Assistant: <tool_call>{{"name": "google-search", "parameters": {{"query": "capital of France"}}}}</tool_call>
[System returns: Paris is the capital and largest city of France...]
Assistant: Based on the search results, the capital of France is Paris.

WORKFLOW:
1. READ the question carefully
2. DECIDE which tool to use
3. CALL the tool with appropriate parameters
4. WAIT for the tool result
5. ANALYZE the result
6. If you need more information, call another tool (steps 2-5)
7. When you have enough information, provide your final answer

{specific_instructions}

Remember: You MUST use tools to gather information. Start by calling a tool!"""
    
    return prompt


def get_basic_search_prompt(question_type: str = "general") -> str:
    """
    Get a system prompt for basic search tasks.
    
    Args:
        question_type: Type of question (e.g., "factual", "current_events", "research")
    """
    
    role = "You are a research assistant that helps find accurate, up-to-date information from the web."
    
    task = "Find accurate information to answer the user's question using web search tools."
    
    tools = [
        {
            "name": "google-search",
            "params": ["query", "filter_year (optional)"],
            "description": "Search the web for current information. Use clear, specific search queries."
        },
        {
            "name": "fetch_data",
            "params": ["url"],
            "description": "Fetch and read detailed content from a specific URL. Use after finding relevant URLs."
        }
    ]
    
    specific = """
SEARCH STRATEGY:
- Start with a broad search to find sources
- If you find a relevant URL, use fetch_data to read it
- Cross-reference multiple sources when possible
- Cite your sources in the final answer

IMPORTANT: You must make at least ONE tool call before answering. Do not answer from your training data."""
    
    return get_tool_calling_system_prompt(role, task, tools, specific)


# Example usage for different scenarios

BASIC_SEARCH_PROMPT = get_basic_search_prompt()

FACTUAL_SEARCH_PROMPT = get_tool_calling_system_prompt(
    role_description="You are a fact-checking research assistant.",
    task_description="Find accurate, verifiable facts using web search tools.",
    tools_info=[
        {
            "name": "google-search",
            "params": ["query"],
            "description": "Search for factual information from reliable sources"
        }
    ],
    specific_instructions="Focus on authoritative sources like government sites, academic institutions, and established news outlets."
)

CURRENT_EVENTS_PROMPT = get_tool_calling_system_prompt(
    role_description="You are a current events research assistant.",
    task_description="Find the latest information about current events using web search.",
    tools_info=[
        {
            "name": "google-search",
            "params": ["query", "filter_year"],
            "description": "Search for recent news and updates. Use filter_year for year-specific results."
        }
    ],
    specific_instructions="Always use filter_year to ensure you get the most recent information. Prefer recent sources (2023-2024)."
)


def inject_tool_instructions_into_prompt(original_prompt: str) -> str:
    """
    Inject tool calling instructions into an existing prompt.
    
    Args:
        original_prompt: The original system prompt
        
    Returns:
        Enhanced prompt with tool calling instructions
    """
    
    tool_instructions = """

CRITICAL INSTRUCTION - TOOL USAGE:
You MUST use tools to gather information. To call a tool, use this exact format:
<tool_call>{"name": "tool_name", "parameters": {"param": "value"}}</tool_call>

Example:
<tool_call>{"name": "google-search", "parameters": {"query": "your search query"}}</tool_call>

You have these tools available:
- google-search: Search the web (required parameters: query)
- fetch_data: Read URL content (required parameters: url)

IMPORTANT: Make at least ONE tool call before providing your final answer!"""
    
    # Insert after the first sentence or paragraph
    if '\n' in original_prompt:
        parts = original_prompt.split('\n', 1)
        return parts[0] + tool_instructions + '\n' + parts[1]
    else:
        return original_prompt + tool_instructions

