"""
Python Code Interpreter Tool for Math Tasks

Uses AG2's LocalCommandLineCodeExecutor for full Python environment access.
This allows executing any Python code with all installed packages (numpy, scipy, etc.)
"""

import os
import sys
import tempfile
from typing import Annotated
from pathlib import Path

# Use AG2's built-in code executor
from autogen.coding import LocalCommandLineCodeExecutor
from autogen.coding.base import CodeBlock


# Create a shared executor instance with a persistent work directory
_work_dir = Path(tempfile.mkdtemp(prefix="ag2_code_"))
_executor = LocalCommandLineCodeExecutor(
    timeout=120,  # 2 minutes timeout
    work_dir=_work_dir,
)


def python_execute(
    code: Annotated[str, "The Python code to execute. Should be valid Python code that computes the answer."],
) -> str:
    """
    Execute Python code using the current Python environment.

    This function executes Python code to solve mathematical problems.
    The code has full access to all installed packages (numpy, scipy, sympy, etc.)
    The code should compute the answer and either:
    1. Print the result using print()
    2. Store the result in a variable named 'result' or 'answer'

    Args:
        code: The Python code to execute

    Returns:
        The output of the code execution or the computed result

    Example:
        >>> output = python_execute("import numpy as np\\nprint(np.sum([1,2,3]))")
        >>> print(output)  # "6"
    """
    try:
        # Execute the code using AG2's LocalCommandLineCodeExecutor
        code_block = CodeBlock(language="python", code=code)
        result = _executor.execute_code_blocks([code_block])
        
        if result.exit_code == 0:
            output = result.output.strip() if result.output else ""
            if output:
                return f"Code executed successfully.\nOutput: {output}"
            else:
                return "Code executed successfully. No output produced."
        else:
            # Execution failed
            error_output = result.output.strip() if result.output else "Unknown error"
            return f"Error executing code (exit code {result.exit_code}):\n{error_output}"
    
    except Exception as e:
        return f"Error executing code: {type(e).__name__}: {str(e)}"


def register_code_interpreter(agent, executor=None):
    """
    Register the Python code interpreter tool with AG2 agents.

    This function registers the python_execute tool that allows agents
    to execute Python code for mathematical computations.
    The code runs in the current Python environment with full package access.

    Args:
        agent: The ConversableAgent that will call the tool (has LLM)
        executor: The ConversableAgent that will execute the tool (no LLM).
                  If None, agent will execute tools itself (for GroupChat scenarios)

    Example:
        >>> from autogen import ConversableAgent
        >>> from ag2_core import get_deepseek_config
        >>>
        >>> agent = ConversableAgent("MathAgent", llm_config=get_deepseek_config())
        >>> executor = ConversableAgent("Executor", llm_config=False)
        >>> register_code_interpreter(agent, executor)
        >>>
        >>> # Or for GroupChat (agent executes tools itself)
        >>> register_code_interpreter(agent)
    """
    # Register python_execute
    agent.register_for_llm(
        name="python_execute",
        description="Execute Python code to solve mathematical problems. Write Python code that computes the answer and prints the result. You have full access to numpy, scipy, sympy, and all installed packages."
    )(python_execute)

    if executor is not None:
        executor.register_for_execution(
            name="python_execute"
        )(python_execute)
    else:
        # Agent executes tools itself (for GroupChat)
        agent.register_for_execution(
            name="python_execute"
        )(python_execute)

    if executor is not None:
        print(f"✓ Registered code interpreter for {agent.name} (executor: {executor.name})")
    else:
        print(f"✓ Registered code interpreter for {agent.name} (self-executing)")
