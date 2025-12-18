import asyncio
import re
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

from autogen_agentchat.messages import BaseChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pettingllms.mas_graph.math_graph.math_env import MathEnv, MathEnvBatch


def extract_answer(text: str) -> str:
    """
    Extract the final answer from solution text.
    Looks for patterns like:
    - "The answer is X"
    - "Final answer: X"
    - "Answer: X"
    - Last boxed expression \\boxed{X}
    
    Args:
        text: Solution text
        
    Returns:
        Extracted answer or empty string
    """
    if not text:
        return ""
    
    # Try to find boxed answer (LaTeX style)
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # Try to find explicit answer statements
    answer_patterns = [
        r'[Ff]inal [Aa]nswer:?\s*(.+?)(?:\n|$)',
        r'[Tt]he answer is:?\s*(.+?)(?:\n|$)',
        r'[Aa]nswer:?\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    
    # If no pattern matched, try to extract last line with numbers
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Check if last line contains numbers
        if re.search(r'\d', last_line):
            return last_line
    
    return ""


def normalize_answer(answer: str) -> str:
    """
    Normalize answer string for comparison.
    - Remove extra whitespace
    - Convert to lowercase
    - Remove common punctuation
    - Extract numbers if present
    
    Args:
        answer: Answer string
        
    Returns:
        Normalized answer
    """
    if not answer:
        return ""
    
    # Convert to lowercase and strip
    normalized = answer.lower().strip()
    
    # Remove common punctuation and symbols
    normalized = re.sub(r'[,\$\s]+', '', normalized)
    
    # Try to extract numeric value if present
    numeric_match = re.search(r'-?\d+\.?\d*', normalized)
    if numeric_match:
        return numeric_match.group(0)
    
    return normalized


def check_answer_correctness(generated_answer: str, ground_truth_answer: str) -> bool:
    """
    Check if generated answer matches ground truth.
    
    Args:
        generated_answer: Generated answer string
        ground_truth_answer: Ground truth answer string
        
    Returns:
        True if answers match, False otherwise
    """
    if not generated_answer or not ground_truth_answer:
        return False
    
    # Normalize both answers
    gen_norm = normalize_answer(generated_answer)
    gt_norm = normalize_answer(ground_truth_answer)
    
    # Direct comparison
    if gen_norm == gt_norm:
        return True
    
    # Try comparing as floats if both are numeric
    try:
        gen_float = float(gen_norm)
        gt_float = float(gt_norm)
        # Allow small floating point differences
        return abs(gen_float - gt_float) < 1e-6
    except (ValueError, TypeError):
        pass
    
    return False


async def math_graph(env: Optional[MathEnv] = None, model_client_dict: dict = None, model_client: OpenAIChatCompletionClient = None):
    """
    Main function for math problem solving workflow using autogen.

    This workflow:
    1. Math solver generates a step-by-step solution
    2. Verifier checks the solution and provides feedback
    3. Loop continues until solution is approved or max iterations reached
    4. Extract final answer and compare with ground truth
    5. Assign final_reward (1.0 if correct, 0.0 otherwise)

    Args:
        env: Optional MathEnv instance with problem and ground truth
        model_client_dict: Dictionary of model clients for each agent {agent_name: client}
        model_client: Single model client (fallback for backward compatibility)

    Returns:
        env: Updated environment with final_reward
    """

    task = env.state.problem

    # Define solver agent
    solver = AssistantAgent(
        "reasoning_generator",
        model_client=model_client_dict.get("reasoning_generator"),
        system_message=(
            "You are an expert mathematician. "
            "Given a mathematical problem, provide a detailed step-by-step solution. "
            "Show your reasoning clearly and conclude with the final answer in the format:\n"
            "Final Answer: <your answer>\n"
            "Or use LaTeX boxed notation: \\boxed{<your answer>}"
        ),
    )

    # Define verifier agent
    verifier = AssistantAgent(
        "tool_generator",
        model_client=model_client_dict.get("tool_generator"),
        system_message=(
            "You are a strict mathematics verifier. "
            "Review the solution provided and check for logical errors, calculation mistakes, or unclear reasoning. "
            "If the solution is correct and complete, reply with exactly:\n"
            "APPROVE\n"
            "Otherwise, reply with:\n"
            "NEEDS_REVISION: <brief explanation of the issue>\n"
            "Suggest how to fix the problem."
        ),
    )

    # Define a simple end agent to mark completion
    end_agent = AssistantAgent(
        "end",
        model_client=model_client_dict.get("reasoning_generator"),
        system_message="You are a completion marker. Just acknowledge the approved solution.",
    )

    # Build graph: solver -> verifier -> (solver [if needs revision] or end [if approved])
    builder = DiGraphBuilder()

    # Add nodes
    builder.add_node(solver)
    builder.add_node(verifier)
    builder.add_node(end_agent)

    # Set solver as the entry point (required for graphs with cycles)
    builder.set_entry_point(solver)

    # Add edge from solver to verifier (unconditional)
    builder.add_edge(solver, verifier)

    # Define condition functions that accept BaseChatMessage
    def needs_revision(msg) -> bool:
        """Check if verifier requests revision"""
        try:
            return "NEEDS_REVISION" in msg.to_model_text()
        except Exception:
            return False

    def approved(msg) -> bool:
        """Check if verifier approves the solution"""
        try:
            return "APPROVE" in msg.to_model_text()
        except Exception:
            return False

    # Add conditional edges from verifier:
    # - If needs revision -> back to solver (creates a loop)
    # - If approved -> to end_agent (exit the loop)
    builder.add_edge(verifier, solver, condition=needs_revision)
    builder.add_edge(verifier, end_agent, condition=approved)

    # Build the graph
    graph = builder.build()
    
    team = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=MaxMessageTermination(15),
    )
    
    # Run the workflow and capture the autogen TaskResult (holds all messages)
    run_result = await Console(team.run_stream(task=task))
    
    # Extract the final solution from the solver's last recorded message
    final_solution: Optional[str] = None
    for msg in reversed(getattr(run_result, "messages", []) or []):
        if isinstance(msg, BaseChatMessage) and msg.source == solver.name:
            final_solution = msg.to_model_text()
            break
    
    # If env is provided, evaluate the solution
    if env is not None:
        try:
            ground_truth = env.state.ground_truth_answer or ""
            extracted_answer = ""
            is_correct = False
            
            if final_solution:
                # Extract answer from solution
                extracted_answer = extract_answer(final_solution)
                
                # Check correctness
                is_correct = check_answer_correctness(extracted_answer, ground_truth)
                
                # Update env state
                env.state.reasoning_generated_solution = final_solution
                env.state.reasoning_generated_solution_history.append(final_solution)
                env.state.reasoning_extracted_answer = extracted_answer
                env.state.reasoning_extracted_answer_history.append(extracted_answer)
                env.state.reasoning_is_correct = is_correct
            
            # Assign final reward: 1.0 if correct, 0.0 otherwise
            final_reward = 1.0 if is_correct else 0.0
            env.state.final_reward = final_reward
            env.final_reward = final_reward
            
        except Exception as e:
            # In case of any evaluation failure, assign zero reward
            print(f"Warning: Failed to evaluate math solution: {e}")
            env.final_reward = 0.0
    
    # Return env with final_reward
    if env is not None:
        return env
