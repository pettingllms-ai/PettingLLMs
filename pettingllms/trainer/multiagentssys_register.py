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
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_env", "CodeEnv"),
    "base_env": safe_import("pettingllms.multi_agent_env.base.env", "BaseEnv"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_env", "MathEnv"),
    "math_aggretion_env": safe_import("pettingllms.multi_agent_env.math_aggretion.math_env", "MathEnv"),
    "alfworld_env": safe_import("pettingllms.multi_agent_env.alfworld.alfworld_env", "AlfworldEnv"),
    "search_env": safe_import("pettingllms.multi_agent_env.search.search_env", "SearchEnv"),
    "stateful_env": safe_import("pettingllms.multi_agent_env.stateful.stateful_env", "StatefulEnv"),
    "pychecker_rl_env": safe_import("pettingllms.multi_agent_env.pychecker_rl.pychecker_env", "PyCheckerEnv"),
}

ENV_BATCH_CLASSES = {
    "base_env": safe_import("pettingllms.multi_agent_env.base.env", "EnvBatch"),
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_env", "CodeEnvBatch"),
    "code_env_single_agent": safe_import("pettingllms.multi_agent_env.code_single_agent.code_test_env", "CodeTestEnvBatch"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_env", "MathEnvBatch"),
    "math_aggretion_env": safe_import("pettingllms.multi_agent_env.math_aggretion.math_env", "MathTestEnvBatch"),
    "alfworld_env": safe_import("pettingllms.multi_agent_env.alfworld.alfworld_env", "AlfWorldEnvBatch"),
    "search_env": safe_import("pettingllms.multi_agent_env.search.search_env", "SearchEnvBatch"),
    "stateful_env": safe_import("pettingllms.multi_agent_env.stateful.stateful_env", "StatefulEnvBatch"),
    "pychecker_rl_env": safe_import("pettingllms.multi_agent_env.pychecker_rl.pychecker_env", "PyCheckerEnvBatch"),
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
    "code_single_agent": safe_import("pettingllms.multi_agent_env.code.agents.code_single_agent", "CodeGenerationAgent"),
    "reasoning_generator": safe_import("pettingllms.multi_agent_env.math.agents.reasoning_agent", "ReasoningAgent"),
    "reasoning_single_agent": safe_import("pettingllms.multi_agent_env.math.agents.reasoning_single_agent", "ReasoningAgent"),
    "tool_generator": safe_import("pettingllms.multi_agent_env.math.agents.tool_agent", "ToolAgent"),
    "math_agent_single_agent": safe_import("pettingllms.multi_agent_env.math_single_agent.agents.math_agent", "MathGenerationAgent"),
    "aggreted_agent": safe_import("pettingllms.multi_agent_env.math_aggretion.agents.aggreted_agent", "AggregationAgent"),
    "aggregation_agent": safe_import("pettingllms.multi_agent_env.math_aggretion.agents.aggreted_agent", "AggregationAgent"),
    "sample_tool_agent": safe_import("pettingllms.multi_agent_env.math_aggretion.agents.sample_tool_agent", "SampleToolAgent"),
    "sample_reasoning_agent": safe_import("pettingllms.multi_agent_env.math_aggretion.agents.sample_reasoning_agent", "SampleReasoningAgent"),
    # Search agents (benchmarks: bamboogle, 2wiki, hotpotqa, musique)
    "search_reasoning_agent": safe_import("pettingllms.multi_agent_env.search.agents.reasoning_agent", "ReasoningAgent"),
    "web_search_agent": safe_import("pettingllms.multi_agent_env.search.agents.web_search_agent", "WebSearchAgent"),
    # Stateful agents
    "plan_agent": safe_import("pettingllms.multi_agent_env.stateful.agents.plan_agent", "PlanAgent"),
    "tool_call_agent": safe_import("pettingllms.multi_agent_env.stateful.agents.tool_agent", "ToolAgent"),
    # PyChecker RL agents
    "pychecker_agent": safe_import("pettingllms.multi_agent_env.pychecker_rl.agents.pychecker_agent", "PyCheckerAgent"),
    "gen_tb_agent": safe_import("pettingllms.multi_agent_env.pychecker_rl.agents.gen_tb_agent", "GenTBAgent"),
}

ENV_WORKER_CLASSES = {
    "code_env": safe_import("pettingllms.multi_agent_env.code.code_worker", "get_ray_docker_worker_cls"),
    "math_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "math_aggretion_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "search_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "stateful_env": safe_import("pettingllms.multi_agent_env.math.math_worker", "get_ray_docker_worker_cls"),
    "pychecker_rl_env": safe_import("pettingllms.multi_agent_env.pychecker_rl.pychecker_worker", "get_ray_docker_worker_cls")
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
ENV_BATCH_CLASS_MAPPING = {k: v for k, v in ENV_BATCH_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
ENV_WORKER_CLASS_MAPPING = {k: v for k, v in ENV_WORKER_CLASSES.items() if v is not None}