def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


# Import environment classes
ENV_CLASSES = {
   
    # Multi-agent system environments
    "web_env": safe_import("pettingllms.multi_agent_env.frontend.websight_env", "WebEnv"),
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_test_env", "CodeTestEnv"),
    "base_env": safe_import("pettingllms.multi_agent_env.base.env", "BaseEnv"),
    "multi_turn_env": safe_import("pettingllms.multi_agent_env.base.env", "MultiTurnEnvironment"),
    "code_env_single_agent": safe_import("pettingllms.multi_agent_env.code_single_agent.code_test_env", "CodeTestEnv"),
    "math_env_single_agent": safe_import("pettingllms.multi_agent_env.math.math_test_env", "MathTestEnv"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_test_env", "MathTestEnv"),
    "alfworld_env": safe_import("pettingllms.multi_agent_env.alfworld.alfworld_env", "AlfworldEnv"),
}

ENV_BATCH_CLASSES = {
    "base_env": safe_import("pettingllms.multi_agent_env.base.env", "EnvBatch"),
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_test_env", "CodeTestEnvBatch"),
    "code_env_single_agent": safe_import("pettingllms.multi_agent_env.code_single_agent.code_test_env", "CodeTestEnvBatch"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_test_env", "MathTestEnvBatch"),
    "math_env_single_agent": safe_import("pettingllms.multi_agent_env.math.math_test_env", "MathTestEnvBatch"),
    "alfworld_env": safe_import("pettingllms.multi_agent_env.alfworld.alfworld_env", "AlfWorldEnvBatch"),
}

# Import agent classes
AGENT_CLASSES = {
    "alfworld_agent": safe_import("pettingllms.multi_agent_env.alfworld.alf_agent", "AlfWorldAgent"),
    # Multi-agent system agents
    "base_agent": safe_import("pettingllms.multi_agent_env.base.agent", "BaseAgent"),
    "review_agent": safe_import("pettingllms.multi_agent_env.frontend.agents.review_agent", "ReviewAgent"),
    "frontend_code_agent": safe_import("pettingllms.multi_agent_env.frontend.agents.code_genaration_agent", "CodeGenerationAgent"),
    "multiagent_code_agent": safe_import("pettingllms.multi_agent_env.code.agents.code_agent", "CodeGenerationAgent"),
    "unit_test_agent": safe_import("pettingllms.multi_agent_env.code.agents.unit_test_agent", "UnitTestGenerationAgent"),
    # Aliases aligned with config.multi_agent_interaction.turn_order values
    "code_generator": safe_import("pettingllms.multi_agent_env.code.agents.code_agent", "CodeGenerationAgent"),
    "test_generator": safe_import("pettingllms.multi_agent_env.code.agents.unit_test_agent", "UnitTestGenerationAgent"),
    "code_generator_single_agent": safe_import("pettingllms.multi_agent_env.code_single_agent.agents.code_agent", "CodeGenerationAgent"),
    "reasoning_agent": safe_import("pettingllms.multi_agent_env.math.agents.reasoning_agent", "ReasoningAgent"),
    "tool_agent": safe_import("pettingllms.multi_agent_env.math.agents.tool_agent", "ToolAgent"),
    "math_agent_single_agent": safe_import("pettingllms.multi_agent_env.math_single_agent.agents.math_agent", "MathGenerationAgent"),
}

ENV_WORKER_CLASSES = {
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_utils", "get_ray_docker_worker_cls"),
    "code_env_single_agent": safe_import("pettingllms.multi_agent_env.code_single_agent.code_utils", "get_ray_docker_worker_cls"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_utils", "get_ray_docker_worker_cls")
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
ENV_BATCH_CLASS_MAPPING = {k: v for k, v in ENV_BATCH_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
ENV_WORKER_CLASS_MAPPING = {k: v for k, v in ENV_WORKER_CLASSES.items() if v is not None}