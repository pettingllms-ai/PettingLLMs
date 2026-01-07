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
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
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
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
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
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
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
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
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
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },
    
    "math_solver": {
        "category": "Math Problem Solver",
        "description": "Single agent uses Python code execution to solve mathematical problems",
        "code": '''# Example 6: Math Problem Solver with Code Execution
from workflow import AgentNode, Workflow, ToolRegistry
from utils.environments.math_env import MathEnvironment
print(result.content)
'''
    },
    
    "math_solver": {
        "category": "Math Problem Solver",
        "description": "Single agent uses Python code execution to solve mathematical problems",
        "code": '''# Example 6: Math Problem Solver with Code Execution
from workflow import AgentNode, Workflow, ToolRegistry
from utils.environments.math_env import MathEnvironment

# Setup math environment with code executor
math_env = MathEnvironment(timeout=120)
tool_registry = ToolRegistry()

tool_registry.register(
    name="python_execute",
    func=math_env.execute,
    description="Execute Python code to solve mathematical problems. The code has access to numpy, scipy, sympy, and other math libraries. IMPORTANT: Each execution runs in an ISOLATED environment - variables, functions, and imports from previous executions are NOT preserved. You must include ALL necessary imports and function definitions in EACH code block.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The Python code to execute. Must be SELF-CONTAINED with all imports and function definitions. Should print the final answer."}
        },
        "required": ["code"]
    }
)

# Create a math solver agent
math_agent = AgentNode(
    name="MathSolver",
    system_prompt=(
        "You are an expert mathematician who solves problems using Python code.\\n\\n"
        
        "IMPORTANT - YOU MUST USE THE PYTHON CODE EXECUTOR:\\n"
        "For any calculation, you MUST write and execute Python code. Do NOT compute in your head.\\n\\n"
        
        "RESPONSE FORMAT:\\n"
        "When you need to solve a problem, respond in this format:\\n"
        "<think>Your reasoning about the mathematical approach</think>\\n"
        '<tool_call>{"name": "python_execute", "parameters": {"code": "your_python_code"}}</tool_call>\\n\\n'
        
        "AVAILABLE TOOLS:\\n"
        "1. python_execute(code): Execute Python code. Has access to numpy, scipy, sympy, math, etc.\\n\\n"
        
        "EXAMPLE:\\n"
        "Human: What is the integral of x^2 from 0 to 1?\\n"
        "Assistant: <think>I need to compute the definite integral of x^2 from 0 to 1. I'll use sympy for symbolic integration.</think>\\n"
        '<tool_call>{"name": "python_execute", "parameters": {"code": "from sympy import symbols, integrate\\nx = symbols(\\'x\\')\\nresult = integrate(x**2, (x, 0, 1))\\nprint(f\\'The integral is: {result}\\')"}}</tool_call>\\n'
        "[Tool returns: Code executed successfully.\\nOutput: The integral is: 1/3]\\n"
        "Assistant: <think>The code executed successfully and gave the answer 1/3.</think>\\n"
        "The integral of x² from 0 to 1 is **1/3**.\\n\\n"
        
        "WORKFLOW:\\n"
        "1. READ the problem carefully\\n"
        "2. THINK about the mathematical approach\\n"
        "3. WRITE Python code to solve the problem\\n"
        "4. EXECUTE the code using python_execute\\n"
        "5. INTERPRET the results\\n"
        "6. PROVIDE the final answer\\n\\n"
        
        "Remember: Always show your thinking process and always use code for calculations!"
    ),
    tool_registry=tool_registry,
    max_turns=8,
    enable_conversation_logging=True
)

# Create workflow
workflow = Workflow(name="math_solver")
workflow.add_node(math_agent)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },
    
    "math_ensemble": {
        "category": "Math Ensemble with Verification",
        "description": "Multiple math agents solve problem independently, then verify each other's solutions",
        "code": '''# Example 7: Math Ensemble with Cross-Verification
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode
from utils.environments.math_env import MathEnvironment

# Setup math environment
math_env = MathEnvironment(timeout=120)
tool_registry = ToolRegistry()

tool_registry.register(
    name="python_execute",
    func=math_env.execute,
    description="Execute Python code to solve math problems. Has access to numpy, scipy, sympy. IMPORTANT: Each execution is ISOLATED - include ALL imports and definitions in each code block.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Self-contained Python code with all imports and definitions"}
        },
        "required": ["code"]
    }
)

# Create multiple math solvers with different approaches
algebraic_solver = AgentNode(
    name="AlgebraicSolver",
    system_prompt=(
        "You are a mathematician who prefers algebraic and symbolic approaches. "
        "Use sympy for symbolic computation. Always write Python code to solve."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

numerical_solver = AgentNode(
    name="NumericalSolver",
    system_prompt=(
        "You are a mathematician who prefers numerical methods. "
        "Use numpy and scipy for numerical computation. Always write Python code to solve."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

verification_solver = AgentNode(
    name="VerificationSolver",
    system_prompt=(
        "You are a careful mathematician who double-checks every step. "
        "Solve the problem step by step, verify intermediate results with code."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

# Create consensus agent to verify and synthesize
consensus_agent = AgentNode(
    name="MathConsensus",
    system_prompt=(
        "You are a senior mathematician. Review the solutions from multiple solvers. "
        "Verify the answers match. If they differ, run your own calculation to determine the correct answer."
    ),
    tool_registry=tool_registry,
    max_turns=4
)

# Create ensemble node
ensemble = EnsembleNode(
    name="MathEnsemble",
    agents=[algebraic_solver, numerical_solver, verification_solver],
    strategy="consensus",
    consensus_agent=consensus_agent
)

# Create workflow
workflow = Workflow(name="math_ensemble")
workflow.add_node(ensemble)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },
    
    "math_reflection": {
        "category": "Math Reflection Solver",
        "description": "Agent solves math problem and iteratively refines solution through self-reflection",
        "code": '''# Example 8: Reflective Math Solver
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import ReflectionNode
from utils.environments.math_env import MathEnvironment

# Setup math environment
math_env = MathEnvironment(timeout=120)
tool_registry = ToolRegistry()

tool_registry.register(
    name="python_execute",
    func=math_env.execute,
    description="Execute Python code for mathematical computation. IMPORTANT: Each execution is ISOLATED - include ALL imports and function definitions in each code block.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Self-contained Python code with all imports and definitions"}
        },
        "required": ["code"]
    }
)

# Create a reflective math solver
math_agent = AgentNode(
    name="ReflectiveMathSolver",
    system_prompt=(
        "You are a meticulous mathematician. Solve problems using Python code. "
        "After getting a result, reflect on whether the approach was correct and "
        "verify the answer using a different method if possible."
    ),
    tool_registry=tool_registry,
    max_turns=8
)

# Create reflection node for iterative improvement
reflection = ReflectionNode(
    name="MathReflection",
    agent=math_agent,
    num_iterations=2
)

# Create workflow
workflow = Workflow(name="math_reflection")
workflow.add_node(reflection)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },
    
    "reflection_ensemble_search": {
        "category": "Reflection Ensemble Search (Combined)",
        "description": "Each agent in ensemble uses self-reflection before consensus - combines Reflection + Ensemble patterns",
        "code": '''# Example 9: Reflection Ensemble Search (Combined Pattern)
# Each ensemble agent uses self-reflection to refine their answer before consensus
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode, ReflectionNode
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

# Create base agents with different research approaches
agent1 = AgentNode(
    name="ThoroughResearcher",
    system_prompt=(
        "You are a thorough researcher. Search multiple sources and "
        "cross-reference information before providing an answer."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

agent2 = AgentNode(
    name="CriticalAnalyst",
    system_prompt=(
        "You are a critical analyst. Evaluate source credibility and "
        "provide well-verified information with citations."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

agent3 = AgentNode(
    name="CreativeExplorer",
    system_prompt=(
        "You are a creative explorer. Look for unconventional sources "
        "and unique perspectives on the topic."
    ),
    tool_registry=tool_registry,
    max_turns=5
)

# Wrap each agent in ReflectionNode for self-improvement
# This is the key combination: each agent reflects on their answer
reflective_agent1 = ReflectionNode(
    name="ReflectiveThorough",
    agent=agent1,
    num_iterations=1  # One reflection iteration per agent
)

reflective_agent2 = ReflectionNode(
    name="ReflectiveCritical",
    agent=agent2,
    num_iterations=1
)

reflective_agent3 = ReflectionNode(
    name="ReflectiveCreative",
    agent=agent3,
    num_iterations=1
)

# Create consensus agent
consensus_agent = AgentNode(
    name="ConsensusAgent",
    system_prompt=(
        "You are a synthesis expert. Review multiple refined research results "
        "and create a comprehensive, accurate final answer."
    ),
    tool_registry=tool_registry,
    max_turns=2
)

# Create ensemble with reflective agents (Combined Pattern)
ensemble = EnsembleNode(
    name="ReflectionEnsemble",
    agents=[reflective_agent1, reflective_agent2, reflective_agent3],
    strategy="consensus",
    consensus_agent=consensus_agent
)

# Create workflow
workflow = Workflow(name="reflection_ensemble_search")
workflow.add_node(ensemble)

# Run workflow
result = workflow.run(question)
print("================================================")
print("FINAL ANSWER:")
print(result.content)
print("================================================")   
'''
    },
    
    "math_ensemble_then_reflection": {
        "category": "Math Ensemble + Reflection (Combined)",
        "description": "Multiple math solvers reach consensus, then result is verified through reflection - combines Ensemble + Reflection patterns",
        "code": '''# Example 10: Math Ensemble then Reflection (Combined Pattern)
# Multiple solvers find consensus, then reflection verifies and refines the answer
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode, ReflectionNode
from utils.environments.math_env import MathEnvironment

# Setup math environment
math_env = MathEnvironment(timeout=120)
tool_registry = ToolRegistry()

tool_registry.register(
    name="python_execute",
    func=math_env.execute,
    description="Execute Python code for mathematical computation. IMPORTANT: Each execution is ISOLATED - include ALL imports and function definitions in each code block.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Self-contained Python code with all imports and definitions"}
        },
        "required": ["code"]
    }
)

# Create math solvers with different approaches
symbolic_solver = AgentNode(
    name="SymbolicSolver",
    system_prompt=(
        "You are a mathematician who uses symbolic computation. "
        "Use sympy for exact algebraic solutions. Always include all imports in your code. "
        "Each code execution is isolated - define all functions and imports in each code block."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

numerical_solver = AgentNode(
    name="NumericalSolver",
    system_prompt=(
        "You are a mathematician who uses numerical methods. "
        "Use numpy and scipy for numerical solutions. Always include all imports in your code. "
        "Each code execution is isolated - define all functions and imports in each code block."
    ),
    tool_registry=tool_registry,
    max_turns=6
)

# Create consensus agent
math_consensus = AgentNode(
    name="MathConsensus",
    system_prompt=(
        "You are a senior mathematician. Compare solutions from symbolic and numerical approaches. "
        "If they agree, synthesize the answer. If they differ, run your own verification."
    ),
    tool_registry=tool_registry,
    max_turns=4
)

# Create ensemble for multiple solution approaches
ensemble = EnsembleNode(
    name="MathEnsemble",
    agents=[symbolic_solver, numerical_solver],
    strategy="consensus",
    consensus_agent=math_consensus
)

# Create verification agent for final reflection
verifier = AgentNode(
    name="MathVerifier",
    system_prompt=(
        "You are a meticulous math verifier. Check the solution step by step. "
        "Verify by substituting back into original equations or using alternative methods. "
        "Each code execution is isolated - include all imports and definitions."
    ),
    tool_registry=tool_registry,
    max_turns=4
)

# Wrap verifier in reflection for thorough checking
reflection = ReflectionNode(
    name="MathVerification",
    agent=verifier,
    num_iterations=1
)

# Create workflow: Ensemble -> Reflection (Combined Pattern)
workflow = Workflow(name="math_ensemble_reflection")
workflow.add_nodes([ensemble, reflection])

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    }
}


# Prompt template for code generation
CODE_GENERATION_PROMPT_TEMPLATE = """You are an expert Python developer specializing in multi-agent workflow systems. Your task is to generate workflow code based on the user's question category and specific question.

## Step 1: Identify the Problem Type

First, analyze the question to determine if it is:
- **MATH_PROBLEM**: Requires mathematical computation, algebraic manipulation, calculus, numerical analysis, etc.
- **SEARCH_PROBLEM**: Requires web search, fact-finding, information retrieval, etc.

This is critical because:
- For MATH problems: Use MathEnvironment with python_execute tool (NO search tools)
- For SEARCH problems: Use SearchEnvironment with google-search and fetch_data tools (NO code execution)

## Available Workflow Patterns

You have access to the following workflow patterns. Choose the most appropriate one based on the question:

### For SEARCH problems:
1. **basic_search**: Single agent performs web search (for straightforward factual questions)
2. **ensemble_search**: Multiple agents reach consensus (for questions requiring multiple perspectives)
3. **debate_search**: Agents debate different sides (for controversial or multi-sided questions)
4. **reflection_search**: Agent refines answer through self-reflection (for complex questions requiring careful thought)
5. **complex_workflow**: Multi-stage pipeline (for questions requiring research, verification, and synthesis)
6. **reflection_ensemble_search**: COMBINED - Each ensemble agent uses reflection before consensus (for complex questions needing both diverse perspectives AND refined answers)

### For MATH problems:
1. **math_solver**: Single agent uses Python code execution to solve mathematical problems
2. **math_ensemble**: Multiple math agents solve independently, then verify each other's solutions
3. **math_reflection**: Agent solves and a critic agent to verify the solution
4. **complex_workflow**: Multi-stage pipeline (for questions requiring research, verification, and synthesis)
5. **math_ensemble_then_reflection**: COMBINED - Ensemble reaches consensus, then reflection verifies (for challenging problems needing both multiple approaches AND verification)


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
1. **Problem Type Identification**: Is this a MATH problem or a SEARCH problem?
   - MATH: Contains equations, asks for calculations, proofs, numerical answers, algebraic manipulation
   - SEARCH: Asks about facts, events, people, places, current information, opinions
2. **Problem Analysis**: What is this question asking? What kind of information/computation is needed?
3. **Workflow Pattern Selection**: Which workflow pattern is most suitable and why?
   
   For SEARCH problems:
   - Is this a straightforward factual question? → basic_search
   - Does it need multiple perspectives or verification? → ensemble_search or complex_workflow
   - Is it controversial with multiple viewpoints? → debate_search
   - Does it require deep thinking and iteration? → reflection_search
   
   For MATH problems:
   - Is this a straightforward calculation? → math_solver
   - Does it need verification from multiple approaches (symbolic vs numerical)? → math_ensemble
   - Is it complex and requires iterative refinement? → math_reflection
   - Does it require a multi-stage pipeline? → complex_workflow
   
4. **Agent Design**: What agents are needed? What should their roles and system prompts be?
5. **Tool Requirements**:
   - For MATH: python_execute (uses MathEnvironment with numpy, scipy, sympy)
   - For SEARCH: google-search, fetch_data (uses SearchEnvironment)
6. **Expected Workflow**: How should information flow through the agents?

Format:
<think>
Problem Type: [MATH or SEARCH]
Problem Analysis: [Your analysis of what the question needs]
Workflow Pattern: [Selected pattern and justification]
Agent Design: [Description of agents needed and their roles]
Tool Requirements: [What tools are needed based on problem type]
Workflow Flow: [How agents will interact]
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

## Step 1: Identify Problem Type
First, determine if this is a MATH problem or a SEARCH problem:
- MATH: Contains equations, asks for calculations, proofs, numerical answers, algebraic manipulation, integrals, derivatives, etc.
- SEARCH: Asks about facts, events, people, places, current information, opinions, web content

## Step 2: Select Pattern

### For SEARCH problems:
- **basic_search**: ONLY for extremely simple, single-fact questions (e.g., "What is the capital of France?"). Use sparingly - less than 10%.
- **ensemble_search**: For questions that benefit from multiple perspectives and consensus.
- **debate_search**: For controversial questions or questions with multiple sides.
- **reflection_search**: For complex questions requiring careful thought and iterative refinement.
- **complex_workflow**: For questions requiring multi-stage processing (research, verify, write).
- **reflection_ensemble_search**: COMBINED pattern - each ensemble agent uses reflection. For complex questions needing both diverse perspectives AND refined answers.

### For MATH problems:
- **math_solver**: For straightforward mathematical problems solvable with code.
- **math_ensemble**: For problems that benefit from multiple solution approaches (symbolic + numerical).
- **math_reflection**: For complex math problems requiring iterative refinement and verification.
- **complex_workflow**: For questions requiring multi-stage processing.
- **math_ensemble_then_reflection**: COMBINED pattern - ensemble reaches consensus, then reflection verifies. For challenging problems needing both multiple approaches AND thorough verification.

IMPORTANT GUIDELINES:
- For MATH problems: Use math_solver, math_ensemble, math_reflection, or math_ensemble_then_reflection
- For SEARCH problems: AVOID basic_search unless trivially simple
- PREFER ensemble, reflection, or COMBINED patterns for complex questions

Question: {question}

Analyze the question and reply with ONLY the pattern name (one of: basic_search, ensemble_search, debate_search, reflection_search, complex_workflow, reflection_ensemble_search, math_solver, math_ensemble, math_reflection, math_ensemble_then_reflection).
"""


def get_category_selection_prompt(question: str) -> str:
    """Get prompt for selecting the appropriate workflow category."""
    return CATEGORY_SELECTION_PROMPT.format(question=question)
