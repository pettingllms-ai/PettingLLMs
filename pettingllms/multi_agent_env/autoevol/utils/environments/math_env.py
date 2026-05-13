"""
Math Environment with Python Code Executor

This module provides a Python code execution environment for solving
mathematical problems. Uses AG2's LocalCommandLineCodeExecutor for
safe code execution.
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated




class MathEnvironment:
    """Environment for executing Python code to solve math problems."""
    
    def __init__(self, timeout: int = 120, work_dir: str = None):
        """
        Initialize the math environment.
        
        Args:
            timeout: Execution timeout in seconds (default: 120)
            work_dir: Working directory for code execution (default: temp dir)
        """
        self.timeout = timeout
        
        if work_dir:
            self._work_dir = Path(work_dir)
            self._work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._work_dir = Path(tempfile.mkdtemp(prefix="math_env_"))
        
        self._executor = None
    
    def execute(
        self,
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
            >>> output = math_env.execute("import numpy as np\\nprint(np.sum([1,2,3]))")
            >>> print(output)  # "Code executed successfully.\\nOutput: 6"
        """
        
        return self._fallback_execute(code)
        
    def _fallback_execute(self, code: str) -> str:
        """
        Fallback execution using subprocess when AG2 is not available.
        
        Args:
            code: The Python code to execute
            
        Returns:
            The output of the code execution
        """
        import subprocess
        import sys
        
        # Write code to a temporary file
        code_file = self._work_dir / "temp_code.py"
        with open(code_file, "w") as f:
            f.write(code)
        
        try:
            result = subprocess.run(
                [sys.executable, str(code_file)],
                cwd=self._work_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0:
                output = result.stdout.strip() if result.stdout else ""
                if output:
                    return f"Code executed successfully.\nOutput: {output}"
                else:
                    return "Code executed successfully. No output produced."
            else:
                error_output = result.stderr.strip() if result.stderr else "Unknown error"
                return f"Error executing code (exit code {result.returncode}):\n{error_output}"
        
        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {self.timeout} seconds"
        except Exception as e:
            return f"Error executing code: {type(e).__name__}: {str(e)}"
        finally:
            # Clean up temp file
            if code_file.exists():
                code_file.unlink()


# Global instance for convenience
_default_math_env = None


def get_math_environment(timeout: int = 120) -> MathEnvironment:
    """Get or create a default math environment instance."""
    global _default_math_env
    if _default_math_env is None:
        _default_math_env = MathEnvironment(timeout=timeout)
    return _default_math_env


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
    env = get_math_environment()
    return env.execute(code)

