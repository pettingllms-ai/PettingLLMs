"""
Few-shot examples for workflow code generation.

This module contains example workflow code patterns that can be used
as few-shot examples for LLM-based code generation.
"""

# Few-shot examples for different problem categories
WORKFLOW_EXAMPLES = {
    "basic_search": {
        "category": "Basic Search",
        "description": "Single agent performs web search and provides answer",
        "code": '''# Example 1: Basic Single-Agent Search
from workflow import AgentNode, Workflow, ToolRegistry
from utils.environments.search_env import SearchEnvironment

# Setup search tools
search_env = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
tool_registry = ToolRegistry()

tool_registry.register(
    name="google-search",
    func=search_env.search,
    description="Search the web using Google. Use this to find current information.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "filter_year": {"type": "integer", "description": "Optional: Filter results to a specific year (YYYY)"}
        },
        "required": ["query"]
    }
)

tool_registry.register(
    name="fetch_data",
    func=search_env.fetch,
    description="Fetch and read content from a specific URL.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch content from"}
        },
        "required": ["url"]
    }
)

# Create a search agent with clear tool usage instructions
search_agent = AgentNode(
    name="SearchAgent",
    system_prompt=(
        "You are a research assistant that helps find accurate information from the web.\\n\\n"
        
        "IMPORTANT - YOU MUST USE TOOLS:\\n"
        "You have access to tools that you MUST use. Do NOT answer from memory.\\n\\n"
        
        "RESPONSE FORMAT:\\n"
        "When you need to use a tool, respond in this format:\\n"
        "<think>Your reasoning about what information you need and which tool to use</think>\\n"
        '<tool_call>{"name": "tool_name", "parameters": {"param": "value"}}</tool_call>\\n\\n'
        
        "AVAILABLE TOOLS:\\n"
        "1. google-search(query, filter_year): Search the web for information\\n"
        "2. fetch_data(url): Fetch and read content from a specific URL\\n\\n"
        
        "EXAMPLE:\\n"
        "Human: What is the capital of France?\\n"
        "Assistant: <think>I need to search for the capital of France to provide accurate information.</think>\\n"
        '<tool_call>{"name": "google-search", "parameters": {"query": "capital of France"}}</tool_call>\\n'
        "[Tool returns: Paris is the capital of France...]\\n"
        "Assistant: <think>The search confirms Paris is the capital. I now have enough information to answer.</think>\\n"
        "Based on the search results, the capital of France is Paris.\\n\\n"
        
        "WORKFLOW:\\n"
        "1. READ the question\\n"
        "2. THINK about what information you need\\n"
        "3. CALL the appropriate tool (with <think> before <tool_call>)\\n"
        "4. ANALYZE the tool results\\n"
        "5. If you need more info, repeat steps 2-4\\n"
        "6. PROVIDE your final answer\\n\\n"
        
        "Remember: Always show your thinking process with <think> tags!"
    ),
    tool_registry=tool_registry,
    max_turns=5,
    enable_conversation_logging=True
)

# Create workflow
workflow = Workflow(name="basic_search")
workflow.add_node(search_agent)

# Run workflow
result = workflow.run(question)
print(result.content)
'''
    },
    
    "ensemble_search": {
        "category": "Ensemble Search",
        "description": "Multiple agents with different approaches reach consensus",
        "code": '''# Example 2: Ensemble Search with Multiple Agents
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode
from utils.environments.search_env import SearchEnvironment

# Setup search tools
search_env = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
tool_registry = ToolRegistry()

tool_registry.register(
    name="google-search",
    func=search_env.search,
    description="Search the web using Google.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)

# Create multiple search agents with different approaches
agent1 = AgentNode(
    name="ThoroughSearchAgent",
    system_prompt=(
        "You are a thorough researcher. Search multiple sources and "
        "cross-reference information before providing an answer."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

agent2 = AgentNode(
    name="QuickSearchAgent",
    system_prompt=(
        "You are an efficient researcher. Quickly find the most relevant "
        "information and provide a concise answer."
    ),
    tool_registry=tool_registry,
    max_turns=3
)

agent3 = AgentNode(
    name="CriticalSearchAgent",
    system_prompt=(
        "You are a critical researcher. Evaluate source credibility and "
        "provide well-verified information."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

# Create consensus synthesizer
consensus_agent = AgentNode(
    name="ConsensusAgent",
    system_prompt=(
        "You are a synthesis expert. Review multiple research results and "
        "create a comprehensive, accurate answer that captures the best insights."
    ),
    tool_registry=tool_registry,
    max_turns=2
)

# Create ensemble node
ensemble = EnsembleNode(
    name="SearchEnsemble",
    agents=[agent1, agent2, agent3],
    strategy="consensus",
    consensus_agent=consensus_agent
)

# Create workflow
workflow = Workflow(name="ensemble_search")
workflow.add_node(ensemble)

# Run workflow
result = workflow.run(question)
print(result.content)
'''
    },
    
    "debate_search": {
        "category": "Debate Search",
        "description": "Multiple agents debate different perspectives, then a judge synthesizes",
        "code": '''# Example 3: Multi-Agent Debate Search
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import DebateNode
from utils.environments.search_env import SearchEnvironment

# Setup search tools
search_env = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
tool_registry = ToolRegistry()

tool_registry.register(
    name="google-search",
    func=search_env.search,
    description="Search the web using Google.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)

# Create debaters with different perspectives
debater1 = AgentNode(
    name="ProDebater",
    system_prompt=(
        "You are a debater focusing on positive aspects and benefits. "
        "Use search to find supporting evidence."
    ),
    tool_registry=tool_registry,
    max_turns=4
)

debater2 = AgentNode(
    name="ConDebater",
    system_prompt=(
        "You are a debater focusing on challenges and concerns. "
        "Use search to find counterpoints and issues."
    ),
    tool_registry=tool_registry,
    max_turns=4
)

judge = AgentNode(
    name="Judge",
    system_prompt=(
        "You are an impartial judge. Review the debate and synthesize "
        "a balanced, comprehensive answer."
    ),
    tool_registry=tool_registry,
    max_turns=2
)

# Create debate node
debate = DebateNode(
    name="SearchDebate",
    debaters=[debater1, debater2],
    judge=judge,
    num_rounds=2
)

# Create workflow
workflow = Workflow(name="debate_search")
workflow.add_node(debate)

# Run workflow
result = workflow.run(question)
print(result.content)
'''
    },
    
    "reflection_search": {
        "category": "Reflection Search",
        "description": "Agent performs search and iteratively refines answer through self-reflection",
        "code": '''# Example 4: Reflection-based Search
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import ReflectionNode
from utils.environments.search_env import SearchEnvironment

# Setup search tools
search_env = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
tool_registry = ToolRegistry()

tool_registry.register(
    name="google-search",
    func=search_env.search,
    description="Search the web using Google.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)

# Create a search agent
search_agent = AgentNode(
    name="ReflectiveSearchAgent",
    system_prompt=(
        "You are a careful researcher. Search for information and provide "
        "well-thought-out answers. You are capable of self-reflection and improvement."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

# Create reflection node
reflection = ReflectionNode(
    name="SearchReflection",
    agent=search_agent,
    num_iterations=2
)

# Create workflow
workflow = Workflow(name="reflection_search")
workflow.add_node(reflection)

# Run workflow
result = workflow.run(question)
print(result.content)
'''
    },
    
    "complex_workflow": {
        "category": "Complex Multi-Stage Workflow",
        "description": "Multi-stage pipeline: research -> fact-check -> write",
        "code": '''# Example 5: Complex Multi-Stage Workflow
from workflow import AgentNode, Workflow, ToolRegistry
from utils.environments.search_env import SearchEnvironment

# Setup search tools
search_env = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
tool_registry = ToolRegistry()

tool_registry.register(
    name="google-search",
    func=search_env.search,
    description="Search the web using Google.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)

tool_registry.register(
    name="fetch_data",
    func=search_env.fetch,
    description="Fetch and read content from a specific URL.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch content from"}
        },
        "required": ["url"]
    }
)

# Stage 1: Initial research
researcher = AgentNode(
    name="Researcher",
    system_prompt=(
        "You are a research assistant. Search for comprehensive information "
        "about the topic and gather key facts."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

# Stage 2: Fact checker
fact_checker = AgentNode(
    name="FactChecker",
    system_prompt=(
        "You are a fact checker. Review the research and verify key claims "
        "by searching for additional sources. Identify any inconsistencies."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

# Stage 3: Writer
writer = AgentNode(
    name="Writer",
    system_prompt=(
        "You are a professional writer. Take the researched and verified information "
        "and create a clear, well-structured final answer."
    ),
    tool_registry=tool_registry,
    max_turns=2
)

# Create workflow
workflow = Workflow(name="complex_search")
workflow.add_nodes([researcher, fact_checker, writer])

# Run workflow
result = workflow.run(question)
print(result.content)
'''
    },
    
    "graph_conditional": {
        "category": "Graph with Conditional Routing",
        "description": "Agent graph with conditional routing based on content",
        "code": '''# Example 6: Graph with Conditional Routing
from workflow import AgentNode, AgentGraph, ToolRegistry
from utils.environments.search_env import SearchEnvironment

# Setup search tools
search_env = SearchEnvironment(serper_key=os.getenv("SERPER_KEY"))
tool_registry = ToolRegistry()

tool_registry.register(
    name="google-search",
    func=search_env.search,
    description="Search the web using Google.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"}
        },
        "required": ["query"]
    }
)

# Create router agent
router = AgentNode(
    name="Router",
    system_prompt=(
        "Analyze the question and decide the type. "
        "Reply ONLY with one word: 'FACTUAL' for factual questions, "
        "'ANALYTICAL' for analysis questions, or 'OPINION' for opinion questions."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Create specialized agents
factual_agent = AgentNode(
    name="FactualAgent",
    system_prompt="You are a factual researcher. Provide accurate, well-sourced facts.",
    tool_registry=tool_registry,
    max_turns=5
)

analytical_agent = AgentNode(
    name="AnalyticalAgent",
    system_prompt="You are an analytical researcher. Provide in-depth analysis with evidence.",
    tool_registry=tool_registry,
    max_turns=5
)

opinion_agent = AgentNode(
    name="OpinionAgent",
    system_prompt="You are a balanced researcher. Present multiple perspectives fairly.",
    tool_registry=tool_registry,
    max_turns=5
)

# Create graph
graph = AgentGraph(max_steps=10)
graph.add_node("router", router)
graph.add_node("factual", factual_agent)
graph.add_node("analytical", analytical_agent)
graph.add_node("opinion", opinion_agent)

# Define routing logic
def route_by_type(context):
    decision = context.get_latest_message().content.upper()
    if "FACTUAL" in decision:
        return "factual"
    elif "ANALYTICAL" in decision:
        return "analytical"
    elif "OPINION" in decision:
        return "opinion"
    return "factual"  # default

graph.add_conditional_edges("router", route_by_type)

# Set entry and finish points
graph.set_entry_point("router")
graph.set_finish_point("factual")
graph.set_finish_point("analytical")
graph.set_finish_point("opinion")

# Run graph
result = graph.run(question)
print(result.content)
'''
    }
}


# Prompt template for code generation
CODE_GENERATION_PROMPT_TEMPLATE = """You are an expert Python developer specializing in multi-agent workflow systems. Your task is to generate workflow code based on the user's question category and specific question.

## Available Workflow Patterns

You have access to the following workflow patterns. Choose the most appropriate one based on the question:

1. **basic_search**: Single agent performs web search (for straightforward questions)
2. **ensemble_search**: Multiple agents reach consensus (for questions requiring multiple perspectives)
3. **debate_search**: Agents debate different sides (for controversial or multi-sided questions)
4. **reflection_search**: Agent refines answer through self-reflection (for complex questions requiring careful thought)
5. **complex_workflow**: Multi-stage pipeline (for questions requiring research, verification, and synthesis)
6. **graph_conditional**: Conditional routing based on question type (for questions that need different handling based on type)

## Few-Shot Examples

{examples}

## Task

Given:
- **Question Category**: {category}
- **Specific Question**: {question}

Generate complete, runnable Python code that:
1. Imports all necessary modules at the top
2. Sets up the appropriate workflow pattern
3. Configures agents with suitable system prompts for the question
4. Runs the workflow and prints the result
5. Is self-contained and can be executed directly

## Requirements

1. Start with all imports:
```python
import os
import sys
# Add more imports as needed
```

2. Use the workflow pattern that best fits the question category
3. Design general agent that can be used for different questions.
4. Include proper error handling
5. Make sure the code is complete and runnable

## Output Format

IMPORTANT: You must first reason about the workflow design in a <think> block, then provide the code in a <code> block.

In your <think> block, you MUST answer these questions:
1. **Problem Analysis**: What is this question asking? What kind of information is needed?
2. **Workflow Pattern Selection**: Which workflow pattern is most suitable and why?
   - Is this a straightforward factual question? → basic_search
   - Does it need multiple perspectives or verification? → ensemble_search or complex_workflow
   - Is it controversial with multiple viewpoints? → debate_search
   - Does it require deep thinking and iteration? → reflection_search
   - Does it need different handling based on question type? → graph_conditional
3. **Agent Design**: What agents are needed? What should their roles and system prompts be?
4. **Tool Requirements**: What tools do the agents need? (search, fetch, etc.)
5. **Expected Workflow**: How should information flow through the agents?

Format:
<think>
1. Problem Analysis: [Your analysis of what the question needs]
2. Workflow Pattern: [Selected pattern and justification]
3. Agent Design: [Description of agents needed and their roles]
4. Tool Requirements: [What tools are needed]
5. Workflow Flow: [How agents will interact]
</think>
<code>
```python
...
```
</code>
"""


def get_code_generation_prompt(question: str, include_examples: list = None,
                               use_simple_format: bool = False, random_sample_examples: bool = True,
                               force_nested: bool = False) -> tuple:
    """
    Generate a prompt for code generation.

    Args:
        question: The specific question to answer
        include_examples: List of example categories to include (default: randomly sampled)
        use_simple_format: If True, return simple format without detailed guide (for SFT data)
        random_sample_examples: If True, randomly sample examples for diversity (default: True)
        force_nested: If True, instruct LLM to create nested/combined workflows

    Returns:
        Tuple of (prompt_string, selected_examples)
    """
    import random

    # Simple format for SFT data collection (just question)
    if use_simple_format:
        return f"Question: {question}", []

    # Determine which examples to include
    if include_examples is None:
        all_examples = list(WORKFLOW_EXAMPLES.keys())

        # Define sampling weights: basic_search gets 10% probability, others share 90%
        weights = []
        for ex in all_examples:
            if ex == "basic_search":
                weights.append(0.1)  # 10% weight for basic_search
            else:
                weights.append(0.9 / (len(all_examples) - 1))  # Share remaining 90%

        # Sample with weighted probabilities
        num_to_sample = max(2, len(all_examples) // 2)  # At least 2 examples
        include_examples = random.choices(all_examples, weights=weights, k=num_to_sample)

    # Build examples section
    examples_text = ""
    for ex_name in include_examples:
        if ex_name in WORKFLOW_EXAMPLES:
            ex = WORKFLOW_EXAMPLES[ex_name]
            examples_text += f"\n### {ex['category']}\n"
            examples_text += f"**Description**: {ex['description']}\n\n"
            examples_text += f"```python\n{ex['code']}\n```\n"

    # Add instruction for nested/combined workflows if requested
    nested_instruction = ""
    if force_nested:
        nested_instruction = """

IMPORTANT: For this question, create a NESTED/COMBINED workflow by combining multiple patterns.
For example:
- Use ensemble search where each agent uses reflection
- Use complex workflow where each stage uses ensemble
- Use debate where each debater uses complex multi-stage process
- Combine graph conditional with any of the above patterns

Be creative and create a sophisticated multi-layered workflow that combines 2-3 patterns."""

    # Format the prompt (no category specified - let LLM decide from examples)
    prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
        examples=examples_text,
        category="auto-detect from question",
        question=question
    ) + nested_instruction

    return prompt, include_examples


# Category selection prompt
CATEGORY_SELECTION_PROMPT = """Given a question, determine the most appropriate workflow pattern.

Available patterns:
- **basic_search**: ONLY for extremely simple, single-fact questions (e.g., "What is the capital of France?"). Use sparingly - less than 10% of questions.
- **ensemble_search**: For questions that benefit from multiple perspectives and consensus. Use for most factual questions requiring verification.
- **debate_search**: For controversial questions or questions with multiple sides that need balanced analysis.
- **reflection_search**: For complex questions requiring careful thought and iterative refinement.
- **complex_workflow**: For questions requiring multi-stage processing (research, verify, write). PREFER this for most research questions.
- **graph_conditional**: For questions that need to be routed to different handlers based on question type.

IMPORTANT GUIDELINES:
- AVOID basic_search unless the question is trivially simple
- PREFER complex_workflow or ensemble_search for most questions
- Use reflection_search for questions requiring deep analysis
- Use debate_search for questions with multiple viewpoints

Question: {question}

Analyze the question complexity and reply with ONLY the pattern name (one of: basic_search, ensemble_search, debate_search, reflection_search, complex_workflow, graph_conditional).
"""


def get_category_selection_prompt(question: str) -> str:
    """Get prompt for selecting the appropriate workflow category."""
    return CATEGORY_SELECTION_PROMPT.format(question=question)

