"""
Registration module for turn-based multi-agent systems.
Imports classes from pettingllms/multi_agent_env/
"""

def safe_import(module_path, class_name):
    """Safely import a class from a module, returning None if import fails."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


# ============================================================================
# Environment Classes (Turn-based)
# ============================================================================
ENV_CLASSES = {
    "base_env": safe_import("pettingllms.multi_agent_env.base.env", "BaseEnv"),
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_env", "CodeEnv"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_env", "MathEnv"),
    "math_aggretion_env": safe_import("pettingllms.multi_agent_env.math_aggretion.math_env", "MathEnv"),
    "search_env": safe_import("pettingllms.multi_agent_env.search.search_env", "SearchEnv"),
    "stateful_env": safe_import("pettingllms.multi_agent_env.stateful.stateful_env", "StatefulEnv"),
    "stateful_vision_env": safe_import("pettingllms.multi_agent_env.stateful_vision.stateful_env", "StatefulEnv"),
    "pychecker_env": safe_import("pettingllms.multi_agent_env.pychecker_rl.pychecker_env", "PyCheckerEnv"),
}

ENV_BATCH_CLASSES = {
    "base_env": safe_import("pettingllms.multi_agent_env.base.env", "EnvBatch"),
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_env", "CodeEnvBatch"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_env", "MathEnvBatch"),
    "math_aggretion_env": safe_import("pettingllms.multi_agent_env.math_aggretion.math_env", "MathEnvBatch"),
    "search_env": safe_import("pettingllms.multi_agent_env.search.search_env", "SearchEnvBatch"),
    "stateful_env": safe_import("pettingllms.multi_agent_env.stateful.stateful_env", "StatefulEnvBatch"),
    "stateful_vision_env": safe_import("pettingllms.multi_agent_env.stateful_vision.stateful_env", "StatefulEnvBatch"),
    "pychecker_env": safe_import("pettingllms.multi_agent_env.pychecker_rl.pychecker_env", "PyCheckerEnvBatch"),
}

# ============================================================================
# Agent Classes (Turn-based)
# ============================================================================
AGENT_CLASSES = {
    # Base agent
    "base_agent": safe_import("pettingllms.multi_agent_env.base.agent", "BaseAgent"),

    # Code agents
    "code_generator": safe_import("pettingllms.multi_agent_env.code.agents.code_agent", "CodeGenerationAgent"),
    "multiagent_code_agent": safe_import("pettingllms.multi_agent_env.code.agents.code_agent", "CodeGenerationAgent"),
    "test_generator": safe_import("pettingllms.multi_agent_env.code.agents.unit_test_agent", "UnitTestGenerationAgent"),
    "unit_test_agent": safe_import("pettingllms.multi_agent_env.code.agents.unit_test_agent", "UnitTestGenerationAgent"),
    "code_selfverify_single_agent": safe_import("pettingllms.multi_agent_env.code.agents.selfverify_single_agent", "CodeGenerationAgent"),

    # Math agents
    "reasoning_generator": safe_import("pettingllms.multi_agent_env.math.agents.reasoning_agent", "ReasoningAgent"),
    "tool_generator": safe_import("pettingllms.multi_agent_env.math.agents.tool_agent", "ToolAgent"),
    "math_selfverify_single_agent": safe_import("pettingllms.multi_agent_env.math.agents.selfverify_single_agent", "ReasoningAgent"),

    # Math aggregation agents
    "aggreted_agent": safe_import("pettingllms.multi_agent_env.math_aggretion.agents.aggreted_agent", "AggregationAgent"),
    "sample_tool_agent": safe_import("pettingllms.multi_agent_env.math_aggretion.agents.sample_tool_agent", "ToolAgent"),
    "sample_reasoning_agent": safe_import("pettingllms.multi_agent_env.math_aggretion.agents.sample_reasoning_agent", "ReasoningAgent"),

    # Search agents
    "search_reasoning_agent": safe_import("pettingllms.multi_agent_env.search.agents.reasoning_agent", "ReasoningAgent"),
    "web_search_agent": safe_import("pettingllms.multi_agent_env.search.agents.web_search_agent", "WebSearchAgent"),

    # Stateful agents
    "plan_agent": safe_import("pettingllms.multi_agent_env.stateful.agents.plan_agent", "PlanAgent"),
    "tool_call_agent": safe_import("pettingllms.multi_agent_env.stateful.agents.tool_agent", "ToolAgent"),

    # Stateful Vision agents
    "plan_agent_vision": safe_import("pettingllms.multi_agent_env.stateful_vision.agents.plan_agent", "PlanAgent"),
    "tool_call_agent_vision": safe_import("pettingllms.multi_agent_env.stateful_vision.agents.tool_agent", "ToolAgent"),

    # PyChecker agents
    "pychecker_agent": safe_import("pettingllms.multi_agent_env.pychecker_rl.agents.pychecker_agent", "PyCheckerAgent"),
    "gen_tb_agent": safe_import("pettingllms.multi_agent_env.pychecker_rl.agents.gen_tb_agent", "GenTBAgent"),
}

# ============================================================================
# Environment Worker Classes (Turn-based)
# ============================================================================
ENV_WORKER_CLASSES = {
    "code_env": safe_import("pettingllms.mu.code.code_worker", "get_ray_docker_worker_cls"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "math_aggretion_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "search_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "stateful_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "stateful_vision_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "pychecker_env": safe_import("pettingllms.multi_agent_env.pychecker_rl.pychecker_worker", "get_ray_docker_worker_cls"),
}

# ============================================================================
# Filter out None values and create final mappings
# ============================================================================
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
ENV_BATCH_CLASS_MAPPING = {k: v for k, v in ENV_BATCH_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
ENV_WORKER_CLASS_MAPPING = {k: v for k, v in ENV_WORKER_CLASSES.items() if v is not None}
