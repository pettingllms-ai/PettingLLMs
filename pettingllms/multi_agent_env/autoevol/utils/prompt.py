"""
Few-shot examples for workflow code generation.

This module contains example workflow code patterns that can be used
as few-shot examples for LLM-based code generation.

Agent roles are differentiated by **solving strategy** (not mathematical
subfield) to encourage genuinely diverse reasoning paths:
  - ForwardSolver:  derives answer from given conditions step by step
  - BackwardSolver: reasons backwards from what the answer must satisfy
  - CaseAnalyzer:   enumerates and checks key cases / boundary conditions
  - Verifier:       checks whether a proposed answer satisfies ALL constraints

Math problems use pure reasoning (no code execution).
"""

# Few-shot examples for different problem categories
WORKFLOW_EXAMPLES = {
    "math_single_agent": {
        "category": "Math Single Agent Solver",
        "description": "Single agent solves math problem through pure reasoning, outputs answer in \\boxed{} format",
        "code": '''# Example 1: Single Agent Math Solver (Pure Reasoning)
from workflow import AgentNode, Workflow, ToolRegistry

# No tools needed - pure reasoning
tool_registry = ToolRegistry()

# Create a math solver agent
math_agent = AgentNode(
    name="MathSolver",
    system_prompt=(
        "You are an expert mathematician who solves problems through careful reasoning.\\n\\n"

        "APPROACH:\\n"
        "1. Read the problem carefully and identify what is being asked\\n"
        "2. Identify the key constraints and relationships\\n"
        "3. Choose an appropriate strategy (direct computation, algebraic manipulation, case analysis, etc.)\\n"
        "4. Execute the solution step by step, showing all work\\n"
        "5. Verify your answer satisfies ALL original constraints\\n\\n"

        "OUTPUT FORMAT:\\n"
        "- Show your complete reasoning process\\n"
        "- After reaching an answer, verify it against the problem statement\\n"
        "- Put your final numerical answer in \\\\boxed{} format\\n"
        "- Example: The answer is \\\\boxed{42}\\n\\n"

        "IMPORTANT: Your final answer MUST be in \\\\boxed{answer} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Create workflow
workflow = Workflow(name="math_single_solver")
workflow.add_node(math_agent)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },

    "math_ensemble_vote": {
        "category": "Math Ensemble with Majority Vote + Judge",
        "description": "Three strategy-diverse solvers solve independently, majority-voted; if votes split, a judge reviews reasoning and picks the best answer",
        "code": '''# Example 2: Math Ensemble — Vote first, Judge on disagreement
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode

# No tools needed - pure reasoning
tool_registry = ToolRegistry()

# --- Strategy-based agent differentiation ---

solver_forward = AgentNode(
    name="ForwardSolver",
    system_prompt=(
        "You are a mathematician who solves problems by FORWARD REASONING.\\n\\n"
        "STRATEGY: Start from the given conditions and derive the answer step by step.\\n"
        "- Translate the problem into equations or expressions\\n"
        "- Simplify and compute forward until you reach the answer\\n"
        "- Double-check each algebraic step\\n\\n"
        "Show your complete reasoning. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_backward = AgentNode(
    name="BackwardSolver",
    system_prompt=(
        "You are a mathematician who solves problems by BACKWARD REASONING.\\n\\n"
        "STRATEGY: Think about what the answer must look like, then work backwards.\\n"
        "- What form must the answer take? What constraints must it satisfy?\\n"
        "- Work backwards from the desired conclusion to the given conditions\\n"
        "- Verify the chain of reasoning is reversible and complete\\n\\n"
        "Show your complete reasoning. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_cases = AgentNode(
    name="CaseAnalyzer",
    system_prompt=(
        "You are a mathematician who solves problems by SYSTEMATIC CASE ANALYSIS.\\n\\n"
        "STRATEGY: Identify the key branching points and enumerate cases.\\n"
        "- What are the critical cases or boundary conditions?\\n"
        "- Enumerate and analyze each case separately\\n"
        "- Combine results or identify which case yields the answer\\n"
        "- Verify no cases are missed\\n\\n"
        "Show your complete reasoning. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Judge agent — only called when solvers disagree
judge_agent = AgentNode(
    name="MathJudge",
    system_prompt=(
        "You are a senior mathematician acting as a judge.\\n\\n"
        "You receive the ORIGINAL QUESTION and solutions from multiple solvers "
        "who gave DIFFERENT answers. Your task is to:\\n"
        "1. Re-read the original question carefully\\n"
        "2. Check each solution's reasoning and calculations step by step\\n"
        "3. Identify which solution (if any) is correct\\n"
        "4. If none is correct, derive the right answer yourself\\n\\n"
        "Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Majority vote with judge fallback:
#   - If all solvers agree → accept immediately (fast path)
#   - If votes split → judge reviews all solutions and decides
ensemble = EnsembleNode(
    name="MathEnsemble",
    agents=[solver_forward, solver_backward, solver_cases],
    strategy="majority_vote",
    consensus_agent=judge_agent
)

# Create workflow
workflow = Workflow(name="math_ensemble_vote")
workflow.add_node(ensemble)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },

    "math_ensemble_judge": {
        "category": "Math Ensemble with Judge",
        "description": "Three strategy-diverse solvers solve independently, then a judge reviews all solutions with the original question and selects the best",
        "code": '''# Example 3: Math Ensemble with Judge Selection
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode

# No tools needed - pure reasoning
tool_registry = ToolRegistry()

# --- Strategy-based agent differentiation ---

solver_forward = AgentNode(
    name="ForwardSolver",
    system_prompt=(
        "You are a mathematician who solves problems by FORWARD REASONING.\\n\\n"
        "STRATEGY: Start from the given conditions and derive the answer step by step.\\n"
        "- Translate the problem into equations or expressions\\n"
        "- Simplify and compute forward until you reach the answer\\n"
        "- Double-check each algebraic step\\n\\n"
        "Show your complete reasoning. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_backward = AgentNode(
    name="BackwardSolver",
    system_prompt=(
        "You are a mathematician who solves problems by BACKWARD REASONING.\\n\\n"
        "STRATEGY: Think about what the answer must look like, then work backwards.\\n"
        "- What form must the answer take? What constraints must it satisfy?\\n"
        "- Work backwards from the desired conclusion to the given conditions\\n"
        "- Verify the chain of reasoning is reversible and complete\\n\\n"
        "Show your complete reasoning. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_cases = AgentNode(
    name="CaseAnalyzer",
    system_prompt=(
        "You are a mathematician who solves problems by SYSTEMATIC CASE ANALYSIS.\\n\\n"
        "STRATEGY: Identify the key branching points and enumerate cases.\\n"
        "- What are the critical cases or boundary conditions?\\n"
        "- Enumerate and analyze each case separately\\n"
        "- Combine results or identify which case yields the answer\\n"
        "- Verify no cases are missed\\n\\n"
        "Show your complete reasoning. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Judge reviews all solutions WITH the original question
judge_agent = AgentNode(
    name="MathJudge",
    system_prompt=(
        "You are a senior mathematician acting as a judge.\\n\\n"

        "You will receive the ORIGINAL QUESTION together with solutions from "
        "multiple solvers. Your task is to:\\n"
        "1. Re-read the original question carefully\\n"
        "2. Check each solution's reasoning and calculations step by step\\n"
        "3. Identify errors, unjustified leaps, or missing cases\\n"
        "4. Select the correct answer (or fix it if all solutions have errors)\\n\\n"

        "OUTPUT FORMAT:\\n"
        "- Point out key errors or strengths in each solution\\n"
        "- State your chosen (or corrected) answer with brief justification\\n"
        "- Put the final answer in \\\\boxed{} format\\n\\n"

        "IMPORTANT: You must output exactly ONE answer in \\\\boxed{answer} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Consensus strategy passes original question to judge
ensemble = EnsembleNode(
    name="MathEnsemble",
    agents=[solver_forward, solver_backward, solver_cases],
    strategy="consensus",
    consensus_agent=judge_agent
)

# Create workflow
workflow = Workflow(name="math_ensemble_judge")
workflow.add_node(ensemble)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },

    "math_solver_critic_reflection": {
        "category": "Math Solver with Critic Reflection",
        "description": "Solver solves, Verifier checks against ALL constraints, Solver refines - repeat 2 rounds",
        "code": '''# Example 4: Math Solver + Verifier Reflection (2 rounds)
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import ReflectionNode

# No tools needed - pure reasoning
tool_registry = ToolRegistry()

# Create the main solver agent
solver_agent = AgentNode(
    name="MathSolver",
    system_prompt=(
        "You are an expert mathematician who solves problems through careful reasoning.\\n\\n"

        "APPROACH:\\n"
        "1. Read the problem carefully and list ALL constraints\\n"
        "2. Choose a strategy and execute step by step\\n"
        "3. Show all intermediate calculations\\n"
        "4. Before finalizing, verify your answer against every constraint\\n\\n"

        "If you receive feedback from a verifier, carefully consider their points and:\\n"
        "- Re-examine the specific steps they flagged\\n"
        "- Fix any errors they identified\\n"
        "- Re-verify the corrected solution against ALL constraints\\n\\n"

        "OUTPUT: Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Verifier checks the solution against ALL original constraints
critic_agent = AgentNode(
    name="Verifier",
    system_prompt=(
        "You are a meticulous math verifier. Your ONLY job is to check whether "
        "a proposed solution is correct.\\n\\n"

        "VERIFICATION PROCESS:\\n"
        "1. Re-read the ORIGINAL QUESTION (provided at the top)\\n"
        "2. List every constraint the problem states\\n"
        "3. BACK-SUBSTITUTE: plug the proposed answer back into EACH "
        "original constraint and verify it holds numerically\\n"
        "4. Re-derive key calculations independently (compute from scratch, "
        "don't just re-read the solver's work)\\n"
        "5. Check common errors: off-by-one in ranges, sign errors, "
        "forgotten edge cases, integer vs fraction\\n"
        "6. Verify the answer is a non-negative integer (AIME format)\\n\\n"

        "OUTPUT FORMAT:\\n"
        "- For each constraint: PASS or FAIL with the substitution result\\n"
        "- List specific calculation errors (if any) with corrections\\n"
        "- Final verdict: CORRECT or INCORRECT\\n"
        "- If INCORRECT, state what needs to be fixed\\n\\n"

        "Be rigorous. Do NOT rubber-stamp the solution."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Solver -> Verifier -> Solver refines -> Verifier -> final
reflection = ReflectionNode(
    name="SolverVerifierReflection",
    agent=solver_agent,
    critic_agent=critic_agent,
    num_iterations=2
)

# Create workflow
workflow = Workflow(name="math_solver_verifier")
workflow.add_node(reflection)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },

    "math_self_consistency": {
        "category": "Math Self-Consistency (Same Solver x3 + Vote + Judge)",
        "description": "Run the same solver agent 3 times independently; if unanimous accept, if split a judge decides. High accuracy with low design overhead.",
        "code": '''# Example 5: Self-Consistency (Same Solver x3 + Vote, Judge on split)
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode

# No tools needed - pure reasoning
tool_registry = ToolRegistry()

# Three instances of the same solver (will produce diverse reasoning via sampling)
solver_a = AgentNode(
    name="Solver_A",
    system_prompt=(
        "You are an expert mathematician. Solve the problem step by step.\\n"
        "Show your complete reasoning and verify your answer.\\n"
        "Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_b = AgentNode(
    name="Solver_B",
    system_prompt=(
        "You are an expert mathematician. Solve the problem step by step.\\n"
        "Show your complete reasoning and verify your answer.\\n"
        "Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_c = AgentNode(
    name="Solver_C",
    system_prompt=(
        "You are an expert mathematician. Solve the problem step by step.\\n"
        "Show your complete reasoning and verify your answer.\\n"
        "Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Judge — only invoked when the three runs disagree
judge_agent = AgentNode(
    name="MathJudge",
    system_prompt=(
        "You are a senior mathematician acting as a judge.\\n\\n"
        "Multiple solvers produced DIFFERENT answers for the same problem.\\n"
        "1. Check each solution's reasoning step by step\\n"
        "2. Identify calculation errors or logical gaps\\n"
        "3. Select the correct answer or derive it yourself\\n\\n"
        "Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Vote + judge fallback: unanimous → accept; split → judge decides
ensemble = EnsembleNode(
    name="SelfConsistency",
    agents=[solver_a, solver_b, solver_c],
    strategy="majority_vote",
    consensus_agent=judge_agent
)

# Create workflow
workflow = Workflow(name="math_self_consistency")
workflow.add_node(ensemble)

# Run workflow
print("================================================")
print("FINAL ANSWER:")
result = workflow.run(question)
print(result.content)
print("================================================")
'''
    },

    "math_ensemble_verify": {
        "category": "Math Ensemble + Verifier Pipeline",
        "description": "Three solvers -> judge picks best -> verifier checks -> if wrong, solver refines. Most thorough pattern.",
        "code": '''# Example 6: Ensemble + Judge + Verifier (Comprehensive)
from workflow import AgentNode, Workflow, ToolRegistry
from workflow.nodes import EnsembleNode

# No tools needed - pure reasoning
tool_registry = ToolRegistry()

# --- Three diverse solvers ---

solver_forward = AgentNode(
    name="ForwardSolver",
    system_prompt=(
        "You are a mathematician who solves problems by FORWARD REASONING.\\n"
        "Start from given conditions, derive step by step. Show all work.\\n"
        "Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_backward = AgentNode(
    name="BackwardSolver",
    system_prompt=(
        "You are a mathematician who solves problems by BACKWARD REASONING.\\n"
        "Think about what the answer must satisfy, then work backwards to confirm.\\n"
        "Show all work. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

solver_cases = AgentNode(
    name="CaseAnalyzer",
    system_prompt=(
        "You are a mathematician who solves problems by CASE ANALYSIS.\\n"
        "Enumerate key cases or boundary conditions systematically.\\n"
        "Show all work. Put your final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# --- Judge selects best answer (sees original question + all solutions) ---

judge_agent = AgentNode(
    name="MathJudge",
    system_prompt=(
        "You are a senior mathematician judge.\\n\\n"
        "You receive the ORIGINAL QUESTION and multiple solutions.\\n"
        "1. Check each solution's reasoning step by step\\n"
        "2. Identify calculation errors or logical gaps\\n"
        "3. Select the correct answer or fix errors if needed\\n\\n"
        "Output your chosen/corrected answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

ensemble = EnsembleNode(
    name="MathEnsemble",
    agents=[solver_forward, solver_backward, solver_cases],
    strategy="consensus",
    consensus_agent=judge_agent
)

# --- Verifier checks the judge's answer ---

verifier = AgentNode(
    name="Verifier",
    system_prompt=(
        "You are a meticulous math verifier.\\n\\n"
        "You will see the original question and a proposed solution from previous agents.\\n"
        "1. List ALL constraints from the original problem\\n"
        "2. BACK-SUBSTITUTE: plug the proposed answer into each constraint numerically\\n"
        "3. Re-derive key calculations independently (compute from scratch)\\n"
        "4. Check: is the answer a non-negative integer? (AIME requires 0-999)\\n"
        "5. If correct: restate the answer in \\\\boxed{} format\\n"
        "6. If incorrect: provide the corrected answer in \\\\boxed{} format\\n\\n"
        "IMPORTANT: Always output a final answer in \\\\boxed{} format."
    ),
    tool_registry=tool_registry,
    max_turns=1
)

# Pipeline: Ensemble(3 solvers -> judge) -> Verifier
# The Verifier automatically receives the original question + the ensemble's
# <delivery> summary (not full reasoning), keeping its input compact.
workflow = Workflow(name="math_ensemble_verify")
workflow.add_node(ensemble)
workflow.add_node(verifier)

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
CODE_GENERATION_PROMPT_TEMPLATE = """You are an expert Python developer specializing in multi-agent workflow systems. Your task is to generate workflow code for solving math problems.

## Problem Type: MATH

All problems are mathematical and should be solved through pure reasoning (NO code execution tools).

## Available Workflow Patterns

Choose the most appropriate pattern based on problem complexity:

1. **math_single_agent**: Single agent solves through reasoning (for straightforward problems)
2. **math_ensemble_vote**: Three strategy-diverse solvers + vote; if unanimous accept, if split a judge reviews reasoning and picks the best (for moderate problems)
3. **math_ensemble_judge**: Three strategy-diverse solvers + judge always reviews all solutions (for hard problems)
4. **math_solver_critic_reflection**: Solver + Verifier with 2 rounds of constraint-checking refinement (for problems needing careful verification)
5. **math_self_consistency**: Same solver x3 + vote + judge on disagreement (high accuracy, good when diverse strategies aren't needed)
6. **math_ensemble_verify**: Three solvers + judge + verifier pipeline (most thorough, for competition-level problems)

## Agent Role Design Principles

Differentiate agents by **solving strategy**, NOT by mathematical subfield:
- **ForwardSolver**: derives answer from given conditions step by step
- **BackwardSolver**: reasons backwards from what the answer must satisfy
- **CaseAnalyzer**: enumerates and checks key cases / boundary conditions
- **Verifier**: checks whether a proposed answer satisfies ALL original constraints (does NOT solve from scratch)

## Key Requirements

1. **NO code execution tools** - solve through pure mathematical reasoning
2. **Final answer MUST be in \\boxed{{}} format** - e.g., \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}
3. **Show reasoning steps clearly** before the final answer
4. **Every agent sees the original question** via shared context (the framework handles this automatically)
5. **Inter-agent communication**: The framework automatically asks each agent to output a `<delivery>` block at the end of its response summarising key findings and answer. Downstream agents receive ONLY the `<delivery>` content from upstream agents (not the full reasoning), keeping prompts compact. You do NOT need to add `<delivery>` instructions to system prompts — the framework injects them automatically.

## Few-Shot Examples

{examples}

## Task

Given:
- **Specific Question**: {question}

Generate complete, runnable Python code that:
1. Imports necessary modules
2. Sets up the appropriate workflow pattern
3. Configures agents with strategy-based system prompts
4. Runs the workflow and prints the result

## Output Format

First analyze the problem, then provide code:

Problem Type: MATH

Problem Analysis: [What is this question asking? What mathematical concepts are involved?]

Workflow Pattern: [Selected pattern and why]

Agent Design: [What agents are needed and their strategic roles]

Workflow Flow: [How agents will interact]

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
        include_examples: List of example categories to include (default: all math examples)
        use_simple_format: If True, return simple format without detailed guide (for SFT data)
        random_sample_examples: If True, randomly sample examples for diversity (default: True)
        force_nested: If True, instruct LLM to create nested/combined workflows

    Returns:
        Tuple of (prompt_string, selected_examples)
    """
    import random

    # Simple format for SFT data collection (just question)
    if use_simple_format:
        return f"Design Multi Agent System for the Question: {question}", []

    # For math, always use all three examples
    if include_examples is None:
        all_examples = list(WORKFLOW_EXAMPLES.keys())

        if random_sample_examples:
            # Randomly sample 2-3 examples
            num_to_sample = random.randint(2, 3)
            include_examples = random.sample(all_examples, num_to_sample)
        else:
            include_examples = all_examples

    # Build examples section
    examples_text = ""
    for ex_name in include_examples:
        if ex_name in WORKFLOW_EXAMPLES:
            ex = WORKFLOW_EXAMPLES[ex_name]
            examples_text += f"\n### {ex['category']}\n"
            examples_text += f"**Description**: {ex['description']}\n\n"
            examples_text += f"```python\n{ex['code']}\n```\n"

    # Format the prompt
    prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
        examples=examples_text,
        question=question
    )

    return prompt, include_examples


# Category selection prompt (simplified for math only)
CATEGORY_SELECTION_PROMPT = """Given a math question, determine the most appropriate workflow pattern.

Available patterns:
- **math_single_agent**: Single agent solves through reasoning (for straightforward problems)
- **math_ensemble_vote**: Three strategy-diverse solvers + vote; if unanimous accept, if split a judge decides (for moderate problems)
- **math_ensemble_judge**: Three strategy-diverse solvers + judge always reviews all solutions (for hard problems where reasoning quality matters)
- **math_solver_critic_reflection**: Solver + Verifier with 2 rounds of refinement (for problems needing careful verification)
- **math_self_consistency**: Same solver x3 + vote + judge on disagreement (high accuracy with low overhead, good default for moderate problems)
- **math_ensemble_verify**: Three solvers + judge + verifier pipeline (most thorough, for competition-level problems)

Question: {question}

Reply with ONLY the pattern name (one of: math_single_agent, math_ensemble_vote, math_ensemble_judge, math_solver_critic_reflection, math_self_consistency, math_ensemble_verify).
"""


def get_category_selection_prompt(question: str) -> str:
    """Get prompt for selecting the appropriate workflow category."""
    return CATEGORY_SELECTION_PROMPT.format(question=question)
