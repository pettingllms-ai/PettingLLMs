"""Math graph workflow module for autogen-based math problem solving."""

from .math_env import MathEnv, MathEnvBatch, MathEnvState

# Lazy import math_graph to avoid autogen_agentchat dependency during env registration
def __getattr__(name):
    if name == "math_graph":
        from .math_graph import math_graph
        return math_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MathEnv",
    "MathEnvBatch", 
    "MathEnvState",
    "math_graph",
]
