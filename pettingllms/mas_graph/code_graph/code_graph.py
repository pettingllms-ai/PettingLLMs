import asyncio
import sys
from typing import Optional

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console

from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from pettingllms.mas_graph.code_graph.code_env import CodeEnv, CodeEnvBatch
from pettingllms.multi_agent_env.code.code_utils import (
    extract_code_from_response,
    evaluate_code_against_tests,
)
from pettingllms.multi_agent_env.code.code_worker import get_ray_docker_worker_cls


async def code_graph(env: Optional[CodeEnv] = None, model_client_dict: dict = None, model_client: OpenAIChatCompletionClient = None):
    """
    Main function for code generation and testing workflow.

    Args:
        env: Optional CodeEnv instance with problem and test cases
        model_client_dict: Dictionary of model clients for each agent {agent_name: client}
        model_client: Single model client (fallback for backward compatibility)
    """

    task = env.state.problem

    # Get agent names from model_client_dict
    agent_names = list(model_client_dict.keys())
    coder_name = agent_names[0] if len(agent_names) > 0 else "coder"
    reviewer_name = agent_names[1] if len(agent_names) > 1 else agent_names[0]

    # 1. Docker code executor
    code_executor = DockerCommandLineCodeExecutor(work_dir="code_workdir")
    await code_executor.start()

    # 2. 定义 Coder / Executor / Reviewer
    coder = AssistantAgent(
        coder_name,
        model_client=model_client_dict.get(coder_name),
        system_message=(
            "You are a senior Python engineer. "
            "Given a feature request and review feedback, "
            "write or update code in ```python``` blocks only. "
            "Include minimal comments; make code runnable as-is."
        ),
    )

    executor = CodeExecutorAgent(
        "code_executor",
        model_client=model_client_dict.get(coder_name),
        code_executor=code_executor,
        sources=[coder_name],
    )

    reviewer = AssistantAgent(
        reviewer_name,
        model_client=model_client_dict.get(reviewer_name),
        system_message=(
            "You are a strict code reviewer and tester. "
            "Read the latest code and execution output. "
            "If everything passes, reply with exactly:\n"
            "APPROVE\n"
            "Otherwise, reply with:\n"
            "NEEDS_CHANGE: <one-sentence reason>\n"
            "Optionally propose a short fix suggestion."
        ),
    )

    # 3. GraphFlow：coder -> executor -> reviewer -> (coder or end)
    builder = DiGraphBuilder()

    builder.add_node(coder)
    builder.add_node(executor)
    builder.add_node(reviewer)

    builder.add_edge(coder, executor)
    builder.add_edge(executor, reviewer)

    def approved(msg):
        return "APPROVE" in msg.to_model_text()

    def needs_change(msg):
        return "NEEDS_CHANGE" in msg.to_model_text()

    builder.add_edge(reviewer, coder, condition=needs_change)

    graph = builder.build()

    team = GraphFlow(
        participants=builder.get_participants(),
        graph=graph,
        termination_condition=MaxMessageTermination(20),
    )

    # Run the graph and stream to console
    await Console(team.run_stream(task=task))
    await code_executor.stop()

    # Try to extract the latest generated code from coder's messages
    last_code_block: Optional[str] = None
    try:
        # Common message containers on autogen agents
        possible_histories = []
        for attr in ("messages", "_messages", "chat_history", "history"):
            if hasattr(coder, attr):
                possible_histories.append(getattr(coder, attr))
        # Flatten and scan text contents
        for hist in possible_histories:
            if not hist:
                continue
            # hist could be a list of message objects or dicts
            for m in hist:
                try:
                    text = (
                        m.to_model_text() if hasattr(m, "to_model_text") else (
                            m.get("content", "") if isinstance(m, dict) else str(m)
                        )
                    )
                except Exception:
                    text = str(m)
                code_candidate = extract_code_from_response(text)
                if code_candidate:
                    last_code_block = code_candidate
    except Exception:
        # If we fail to introspect messages, continue with None
        last_code_block = last_code_block

    # If env is provided, compute final reward against golden tests
    if env is not None:
        try:
            gt_inputs = env.state.ground_truth_test_input or []
            gt_outputs = env.state.ground_truth_test_output or []
            passed_ratio = 0.0
            passed_cases = []
            failed_cases = []

            if last_code_block and gt_inputs and gt_outputs:
                # Create a ray docker worker to execute tests safely
                try:
                    RayDockerWorker = get_ray_docker_worker_cls(num_workers=1)
                    ray_actor = RayDockerWorker.remote(0)
                except Exception:
                    ray_actor = None

                # Evaluate code against golden tests (async, ray actor if available)
                passed_ratio, passed_cases, failed_cases = await evaluate_code_against_tests(
                    code=last_code_block,
                    test_inputs=gt_inputs,
                    test_outputs=gt_outputs,
                    timeout=40.0,
                    ray_actor=ray_actor,
                    rollout_idx=getattr(env, "rollout_idx", 0),
                )

                # Update env.state detailed fields for downstream consumers
                env.state.ground_truth_test_vs_generated_code_match_cases = passed_cases
                env.state.ground_truth_test_vs_generated_code_mismatch_cases = failed_cases
                env.state.ground_truth_test_vs_generated_code_match_ratio = float(passed_ratio)
                env.state.generated_code = last_code_block
                env.state.generated_code_history.append(last_code_block)

            # Assign final reward: 1.0 only if all golden tests pass, else 0.0
            final_reward = 1.0 if passed_ratio == 1.0 else 0.0
            
            env.final_reward = final_reward
        except Exception:
            # In case of any evaluation failure, assign zero reward
            env.final_reward = 0.0
    
    # Return env with final_reward if provided
    if env is not None:
        return env

