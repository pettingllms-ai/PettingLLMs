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
}

ENV_BATCH_CLASSES = {
    "base_env": safe_import("pettingllms.multi_agent_env.base.env", "EnvBatch"),
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_test_env", "CodeTestEnvBatch"),
}

# Import agent classes
AGENT_CLASSES = {
    
    # Multi-agent system agents
    "base_agent": safe_import("pettingllms.multi_agent_env.base.agent", "BaseAgent"),
    "review_agent": safe_import("pettingllms.multi_agent_env.frontend.agents.review_agent", "ReviewAgent"),
    "frontend_code_agent": safe_import("pettingllms.multi_agent_env.frontend.agents.code_genaration_agent", "CodeGenerationAgent"),
    "multiagent_code_agent": safe_import("pettingllms.multi_agent_env.code.agents.code_agent", "CodeGenerationAgent"),
    "unit_test_agent": safe_import("pettingllms.multi_agent_env.code.agents.unit_test_agent", "UnitTestGenerationAgent"),
    # Aliases aligned with config.multi_agent_interaction.turn_order values
    "code_generator": safe_import("pettingllms.multi_agent_env.code.agents.code_agent", "CodeGenerationAgent"),
    "test_generator": safe_import("pettingllms.multi_agent_env.code.agents.unit_test_agent", "UnitTestGenerationAgent"),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
ENV_BATCH_CLASS_MAPPING = {k: v for k, v in ENV_BATCH_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
