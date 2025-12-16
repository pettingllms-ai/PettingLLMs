import asyncio
import re
from typing import Optional

from autogen import AssistantAgent, ConversableAgent
from ag2.models.openai import OpenAIChatCompletionClient

from pettingllms.mas_graph.math_graph.math_env import MathEnv, MathEnvBatch
from pettingllms.multi_agent_env.math.math_utils import extract_code


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
    Main function for math problem solving workflow using AG2.

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

    # Define solver agent using AG2's AssistantAgent
    solver = AssistantAgent(
        name="reasoning_generator",
        llm_config={"config_list": [{"model": "gpt-4", "api_key": "dummy"}]},
        system_message=(
            "You are an expert mathematician. "
            "Given a mathematical problem, provide a detailed step-by-step solution. "
            "Show your reasoning clearly and conclude with the final answer in the format:\n"
            "Final Answer: <your answer>\n"
            "Or use LaTeX boxed notation: \\boxed{<your answer>}"
        ),
    )

    # Define verifier agent using AG2's AssistantAgent
    verifier = AssistantAgent(
        name="tool_generator",
        llm_config={"config_list": [{"model": "gpt-4", "api_key": "dummy"}]},
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

    # AG2 uses sequential chat instead of graph flow
    max_rounds = 15
    final_solution: Optional[str] = None

    for round_num in range(max_rounds):
        # Solver generates solution
        solver_response = await solver.a_generate_reply(
            messages=[{"role": "user", "content": task}]
        )

        if solver_response:
            final_solution = solver_response.get("content", "")

        # Verifier checks solution
        verifier_response = await verifier.a_generate_reply(
            messages=[{"role": "user", "content": final_solution}]
        )

        verifier_content = verifier_response.get("content", "") if verifier_response else ""

        # Check if approved
        if "APPROVE" in verifier_content:
            break

        # If needs revision, continue loop with feedback
        if "NEEDS_REVISION" in verifier_content:
            task = f"{task}\n\nPrevious attempt:\n{final_solution}\n\nFeedback:\n{verifier_content}"
        else:
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
