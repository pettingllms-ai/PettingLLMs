"""
Core OpenAI API Patch Module for PettingLLMs

Patch autogen and langchain LLM engines to use llm_async_generate from async_generate.py
"""

import asyncio
import functools
from typing import Dict, List, Optional, Union, Callable
import random
import numpy as np


# Model client factory functions for different agent frameworks
def _create_autogen_client(policy_name: str, address: str, **kwargs):
    """Create AutoGen OpenAIChatCompletionClient."""
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    from autogen_core.models import ModelInfo, ModelFamily

    model_info = ModelInfo(
        vision=False,
        function_calling=True,
        json_output=True,
        family=ModelFamily.UNKNOWN,
        structured_output=False  # Add this to avoid warning
    )

    client = OpenAIChatCompletionClient(
        model=policy_name,
        api_key="dummy",
        base_url=address,
        model_info=model_info,
    )
    return client


def _create_ag2_client(policy_name: str, address: str, **kwargs):
    """Create AG2 OpenAIChatCompletionClient."""
    from ag2.models.openai import OpenAIChatCompletionClient
    return OpenAIChatCompletionClient(
        model=policy_name,
        api_key="dummy",
        base_url=address,
    )


def _create_langchain_client(policy_name: str, address: str, **kwargs):
    """Create LangChain ChatOpenAI client."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model_name=policy_name,
        openai_api_key="dummy",
        openai_api_base=address,
    )


def _create_llamaindex_client(policy_name: str, address: str, **kwargs):
    """Create LlamaIndex OpenAI client."""
    from llama_index.llms.openai import OpenAI
    return OpenAI(
        model=policy_name,
        api_key="dummy",
        api_base=address,
    )


# Registry of model client factory functions by framework
MODEL_CLIENT_FACTORY: Dict[str, Callable] = {
    "autogen": _create_autogen_client,
    "ag2": _create_ag2_client,
    "langchain": _create_langchain_client,
    "langgraph": _create_langchain_client,  # LangGraph uses same client as LangChain
    "llamaindex": _create_llamaindex_client,
}


def create_model_client(agent_framework: str, policy_name: str, address: str, **kwargs):
    """
    Create a model client based on the agent framework.
    
    Args:
        agent_framework: Framework name ('autogen', 'ag2', 'langchain', 'langgraph', 'llamaindex')
        policy_name: Policy/model name
        address: vLLM server address
        **kwargs: Additional arguments to pass to the client factory
        
    Returns:
        Model client instance for the specified framework
        
    Raises:
        ValueError: If agent_framework is not supported
    """
    framework_lower = agent_framework.lower()
    factory_func = MODEL_CLIENT_FACTORY.get(framework_lower)
    
    if factory_func is None:
        raise ValueError(
            f"Unsupported agent_framework: {agent_framework}. "
            f"Supported frameworks: {list(MODEL_CLIENT_FACTORY.keys())}"
        )
    
    return factory_func(policy_name, address, **kwargs)


def create_dummy_model_client(agent_framework: str):
    """
    Create a dummy model client that will be intercepted by patch.
    Uses fake model name (gpt-4o) and dummy address since they will be overridden.
    
    Args:
        agent_framework: Framework name ('autogen', 'ag2', 'langchain', 'langgraph', 'llamaindex')
        
    Returns:
        Model client instance with dummy parameters
        
    Raises:
        ValueError: If agent_framework is not supported
    """
    return create_model_client(
        agent_framework=agent_framework,
        policy_name="gpt-4o",  # Dummy model name, will be intercepted by patch
        address="http://dummy"  # Dummy address, will be overridden by agent_address_mapping
    )


def build_agent_address_mapping(
    agent_names: List[str],
    agent_policy_mapping: Dict[str, str],
    server_address_dict: Dict[str, Union[str, List[str]]]
) -> Dict[str, str]:
    """
    Build agent to vLLM address mapping.

    Args:
        agent_names: List of agent names
        agent_policy_mapping: {agent_name: policy_name}
        server_address_dict: {policy_name: address or [addresses]}

    Returns:
        Dict mapping agent_name to vLLM address
    """
    print(f"[build_agent_address_mapping] agent_names: {agent_names}")
    print(f"[build_agent_address_mapping] agent_policy_mapping: {agent_policy_mapping}")
    print(f"[build_agent_address_mapping] server_address_dict: {server_address_dict}")

    agent_address_mapping = {}
    for agent_name in agent_names:
        policy_name = agent_policy_mapping.get(agent_name)
        print(f"[build_agent_address_mapping] agent_name={agent_name} -> policy_name={policy_name}")

        if policy_name is None:
            print(f"[build_agent_address_mapping] WARNING: No policy_name found for agent '{agent_name}', skipping")
            continue

        _addresses = server_address_dict.get(policy_name)
        print(f"[build_agent_address_mapping] policy_name={policy_name} -> addresses={_addresses}")

        if _addresses is None:
            print(f"[build_agent_address_mapping] WARNING: No address found for policy '{policy_name}', skipping agent '{agent_name}'")
            continue

        if isinstance(_addresses, (list, tuple)):
            _address = random.choice(_addresses) if len(_addresses) > 0 else _addresses[0]
        else:
            _address = _addresses

        agent_address_mapping[agent_name] = _address
        print(f"[build_agent_address_mapping] Mapped agent '{agent_name}' to address '{_address}'")

    print(f"[build_agent_address_mapping] Final mapping: {agent_address_mapping}")
    return agent_address_mapping


# Global state
_server_address_dict: Dict[str, Union[str, List[str]]] = {}
_tokenizer_dict: Dict[str, any] = {}
_ppo_trainer_config_dict: Dict[str, any] = {}
_processor_dict: Dict[str, any] = {}
_agent_policy_mapping: Dict[str, str] = {}
_agent_address_mapping: Dict[str, str] = {}  # Maps agent_name -> vLLM address
_agent_lora_mapping: Dict[str, str] = {}  # Maps agent_name -> lora_id
_agent_config_dict: Dict[str, any] = {}  # Maps agent_name -> agent_config
_current_hop_idx: int = 0  # Renamed from turn_idx to hop_idx
_current_rollout_idx: int = 0
_current_env_idx: int = 0
_patched: bool = False

# Trajectory storage: {(rollout_idx, hop_idx, policy_name): (output_dpr, response)}
_trajectory_store: Dict[tuple, tuple] = {}


def init_patch_context(
    server_address_dict: Dict[str, Union[str, List[str]]],
    tokenizer_dict: Dict[str, any],
    ppo_trainer_config_dict: Dict[str, any],
    agent_policy_mapping: Dict[str, str],
    agent_address_mapping: Optional[Dict[str, str]] = None,
    agent_lora_mapping: Optional[Dict[str, str]] = None,
    agent_config_dict: Optional[Dict[str, any]] = None,
    processor_dict: Optional[Dict[str, any]] = None
):
    """
    Initialize patch context with engine attributes.

    Args:
        server_address_dict: {policy_name: address or [addresses]}
        tokenizer_dict: {policy_name: tokenizer}
        ppo_trainer_config_dict: {policy_name: ppo_config}
        agent_policy_mapping: {agent_name: policy_name}
        agent_address_mapping: {agent_name: vLLM_address} - direct mapping for routing
        agent_lora_mapping: {agent_name: lora_id} - mapping for LoRA adapters
        agent_config_dict: {agent_name: agent_config} - agent configurations
        processor_dict: {policy_name: processor} - processors for multimodal models
    """
    global _server_address_dict, _tokenizer_dict, _ppo_trainer_config_dict, _agent_policy_mapping, _agent_address_mapping, _agent_lora_mapping, _agent_config_dict, _processor_dict
    _server_address_dict = server_address_dict
    _tokenizer_dict = tokenizer_dict
    _ppo_trainer_config_dict = ppo_trainer_config_dict
    _agent_policy_mapping = agent_policy_mapping
    _agent_address_mapping = agent_address_mapping or {}
    _agent_lora_mapping = agent_lora_mapping or {}
    _agent_config_dict = agent_config_dict or {}
    _processor_dict = processor_dict or {}
    print(f"[Patch] Initialized context with policies: {list(server_address_dict.keys())}")
    print(f"[Patch] Agent address mapping: {agent_address_mapping}")
    print(f"[Patch] Agent lora mapping: {agent_lora_mapping}")




def get_rollout_idx() -> int:
    """Get current rollout index."""
    return _current_rollout_idx


def get_env_idx() -> int:
    """Get current environment index."""
    return _current_env_idx


def get_hop_idx() -> int:
    """Get current hop index (incremented each time an LLM request is made)."""
    return _current_hop_idx


def increment_hop_idx():
    """Increment hop index when an LLM request is made."""
    global _current_hop_idx
    _current_hop_idx += 1
    print(f"[Patch] Hop index incremented to {_current_hop_idx}")


def reset_hop_idx():
    """Reset hop index to 0 for new graph execution."""
    global _current_hop_idx, _trajectory_store
    _current_hop_idx = 0
    _trajectory_store = {}  # Clear trajectory store for new rollout
    print(f"[Patch] Hop index reset to 0, trajectory store cleared")


def clear_trajectory_store():
    """Clear the trajectory store."""
    global _trajectory_store
    _trajectory_store = {}
    print(f"[Patch] Trajectory store cleared")


def get_trajectory_store() -> Dict[tuple, tuple]:
    """
    Get collected trajectories for current rollout.

    Returns:
        Dict mapping (rollout_idx, hop_idx, policy_name) to (output_dpr, response)
    """
    return _trajectory_store.copy()


def get_server_address(policy_name: str) -> str:
    """Get vLLM server address for policy."""
    addresses = _server_address_dict[policy_name]
    if isinstance(addresses, list):
        return random.choice(addresses)
    return addresses


async def _patched_generate(
    messages: List[Dict[str, str]],
    agent_name: Optional[str] = None,
    **kwargs
) -> tuple[str, int, int]:
    """
    Core patched generate function that calls llm_async_generate.
    Each call to this function represents one "hop" in the agent graph.

    This function is framework-agnostic - it only takes messages and agent_name.
    All other mappings (policy, address, lora_id, config) are resolved internally.

    Args:
        messages: Chat messages
        agent_name: Agent name for routing and reward attribution
        **kwargs: Additional generation parameters

    Returns:
        Tuple of (response_text, prompt_tokens, completion_tokens)
    """
    from pettingllms.trainer.async_generate import llm_async_generate, convert_prompt_to_dpr

    # Resolve policy_name from agent_name
    policy_name = _agent_policy_mapping.get(agent_name)
    if policy_name is None:
        # Fallback: use first available policy
        policy_name = list(_server_address_dict.keys())[0]
        print(f"[Patch] Warning: agent_name '{agent_name}' not in agent_policy_mapping, using fallback policy '{policy_name}'")

    address = _agent_address_mapping[agent_name]
    print(f"[Patch] Using agent_address_mapping: agent={agent_name} -> address={address}")

    lora_id = None
    # Resolve lora_id from agent_lora_mapping
    if _agent_lora_mapping:
        lora_id = _agent_lora_mapping.get(agent_name)
    if lora_id is not None:
        print(f"[Patch] Using lora_id={lora_id} for agent={agent_name}")

    # Resolve agent_config from agent_config_dict
    agent_config = _agent_config_dict.get(agent_name)

    # Get tokenizer, processor, and ppo_config
    tokenizer = _tokenizer_dict[policy_name]
    processor = _processor_dict.get(policy_name)
    ppo_config = _ppo_trainer_config_dict[policy_name]

    # Get hop/turn/rollout/env indices
    hop_idx = get_hop_idx()
    rollout_idx = get_rollout_idx()
    env_idx = get_env_idx()

    print(f"[Patch] Starting LLM request: rollout={rollout_idx}, hop={hop_idx}, agent={agent_name}")

    # Convert messages to prompt
    if isinstance(messages, list) and len(messages) > 0:
        if isinstance(messages[0], dict):
            # Chat format
            prompt_text = messages[-1].get('content', '')
        else:
            prompt_text = str(messages[-1])
    else:
        prompt_text = str(messages)

    # Determine enable_thinking and enable_multimodal from agent_config
    enable_thinking = getattr(agent_config, 'enable_thinking', False) if agent_config else False
    enable_multimodal = getattr(agent_config, 'enable_multimodal', False) if agent_config else False

    # Create prompt DataProto
    prompt_dpr = convert_prompt_to_dpr(
        tokenizer=tokenizer,
        processor=processor,
        prompts={"text": prompt_text, "image": None},
        max_prompt_length=ppo_config.data.max_prompt_length,
        multi_modal=enable_multimodal,
        enable_thinking=enable_thinking
    )

    # Get model_name from ppo_config
    model_path = ppo_config.actor_rollout_ref.model.path
    if "checkpoint" in str(model_path):
        model_name = str(model_path)
    else:
        model_name = "/".join(str(model_path).split("/")[-2:])

    # Call llm_async_generate
    output_dpr, response = await llm_async_generate(
        rollout_idx=rollout_idx,
        turn_idx=hop_idx,  # Pass hop_idx as turn_idx parameter
        agent_idx=0,
        prompt_dpr=prompt_dpr,
        ppo_trainer_config=ppo_config,
        address=address,
        model_name=model_name,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        image_data=None,
        application_id=f"graph_r{rollout_idx}_h{hop_idx}",
        env_idx=env_idx,
        policy_name=policy_name,
        timeout=kwargs.get('timeout', 60.0),
        mode=kwargs.get('mode', 'inference'),
        lora_id=lora_id,
        agent_config=agent_config,
    )

    # Calculate token counts from output_dpr
    prompt_tokens = 0
    completion_tokens = 0

    if output_dpr is not None and hasattr(output_dpr, 'batch'):
        # Extract prompt and response lengths
        # Note: TensorDict requires using .keys() for membership checks
        batch_keys = output_dpr.batch.keys() if hasattr(output_dpr.batch, 'keys') else []

        if 'input_ids' in batch_keys:
            # Count non-padding tokens in prompts
            prompt_ids = output_dpr.batch['input_ids']
            if hasattr(prompt_ids, 'shape'):
                # Assuming padding tokens are excluded or minimal
                prompt_tokens = int(prompt_ids.shape[-1]) if len(prompt_ids.shape) > 0 else 0

        if 'responses' in batch_keys:
            # Count non-padding tokens in responses
            response_ids = output_dpr.batch['responses']
            if hasattr(response_ids, 'shape'):
                # Count actual tokens (excluding padding)
                if len(response_ids.shape) > 1:
                    completion_tokens = int(response_ids.shape[-1])
                elif len(response_ids.shape) > 0:
                    completion_tokens = int(response_ids.shape[0])

    # Store trajectory with agent_name for later reward attribution
    # Note: reward will be added later by the environment step
    if output_dpr is not None:
        # Add agent_name and hop_idx to output_dpr for trajectory collection
        output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name or "unknown"], dtype=object)
        output_dpr.non_tensor_batch["hop_idx"] = np.array([hop_idx], dtype=np.int32)

        # Store in trajectory store
        global _trajectory_store
        key = (rollout_idx, hop_idx, policy_name)
        _trajectory_store[key] = (output_dpr, response)
        print(f"[Patch] Stored trajectory for key={key}, hop_idx={hop_idx}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")

    # Increment hop index after generation (this represents completing one hop)
    increment_hop_idx()

    return response, prompt_tokens, completion_tokens


def patch_autogen():
    """
    Patch autogen's OpenAIChatCompletionClient to use llm_async_generate.
    """
    global _patched
    if _patched:
        return
    
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    
    
    original_create = OpenAIChatCompletionClient.create

    @functools.wraps(original_create)
    async def patched_create(self, messages, **kwargs):
        # Try to get agent_name directly from the client (set during model_client_dict creation)
        agent_name = getattr(self, '_agent_name', None)

        # Call patched generate with only messages and agent_name
        response_text, prompt_tokens, completion_tokens = await _patched_generate(
            messages=messages,
            agent_name=agent_name,
            **kwargs
        )

        # Return in autogen format
        from autogen_core.models import CreateResult, RequestUsage
        return CreateResult(
            content=response_text,
            usage=RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            finish_reason="stop",
            cached=False,
        )
    
    OpenAIChatCompletionClient.create = patched_create
    _patched = True
    print("[Patch] Patched autogen OpenAIChatCompletionClient.create")


def patch_ag2():
    """
    Patch ag2's OpenAIChatCompletionClient to use llm_async_generate.
    AG2 is a fork of autogen, so the patching mechanism is similar.
    """
    global _patched
    if _patched:
        return
    
    try:
        from ag2.models.openai import OpenAIChatCompletionClient
    except ImportError:
        print("[Patch] ag2 not available, skipping ag2 patch")
        return
    
    original_create = OpenAIChatCompletionClient.create
    
    @functools.wraps(original_create)
    async def patched_create(self, messages, **kwargs):
        # Get model name and infer agent_name from policy mapping
        model_name = kwargs.get('model') or self._model_id

        agent_name = None
        for a_name, p_name in _agent_policy_mapping.items():
            if p_name == model_name:
                agent_name = a_name
                break

        # Call patched generate with only messages and agent_name
        response_text, prompt_tokens, completion_tokens = await _patched_generate(
            messages=messages,
            agent_name=agent_name,
            **kwargs
        )

        # Return in ag2 format (same as autogen)
        from ag2.models import CreateResult, RequestUsage
        return CreateResult(
            content=response_text,
            usage=RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            finish_reason="stop",
            cached=False,
        )
    
    OpenAIChatCompletionClient.create = patched_create
    _patched = True
    print("[Patch] Patched ag2 OpenAIChatCompletionClient.create")


def patch_langchain():
    """
    Patch langchain's ChatOpenAI to use llm_async_generate.
    """
    global _patched
    if _patched:
        return
        
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("[Patch] langchain_openai not available, skipping langchain patch")
        return
    
    original_agenerate = ChatOpenAI._agenerate
    
    @functools.wraps(original_agenerate)
    async def patched_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # Convert langchain messages
        msg_dicts = [{"role": m.type, "content": m.content} for m in messages]

        # Get model name and infer agent_name from policy mapping
        model_name = self.model_name

        # Infer agent_name
        agent_name = None
        for a_name, p_name in _agent_policy_mapping.items():
            if p_name == model_name:
                agent_name = a_name
                break

        # Call patched generate with only messages and agent_name
        response_text, prompt_tokens, completion_tokens = await _patched_generate(
            messages=msg_dicts,
            agent_name=agent_name,
            **kwargs
        )

        # Return in langchain format
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.messages import AIMessage

        message = AIMessage(content=response_text)
        generation = ChatGeneration(
            message=message,
            generation_info={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )
        return ChatResult(generations=[generation])
    
    ChatOpenAI._agenerate = patched_agenerate
    _patched = True
    print("[Patch] Patched langchain ChatOpenAI._agenerate")


def patch_langgraph():
    """
    Patch langgraph's LLM calls to use llm_async_generate.
    LangGraph uses LangChain models underneath, so patching LangChain should work.
    """
    # LangGraph uses LangChain models, so just apply langchain patch
    patch_langchain()
    print("[Patch] LangGraph uses LangChain models, applied langchain patch")


def patch_llamaindex():
    """
    Patch llamaindex's OpenAI to use llm_async_generate.
    """
    global _patched
    if _patched:
        return
        
    try:
        from llama_index.llms.openai import OpenAI
    except ImportError:
        print("[Patch] llama_index not available, skipping llamaindex patch")
        return
    
    original_achat = OpenAI.achat
    
    @functools.wraps(original_achat)
    async def patched_achat(self, messages, **kwargs):
        # Convert llamaindex messages
        if hasattr(messages, '__iter__') and not isinstance(messages, str):
            msg_dicts = [{"role": getattr(m, 'role', 'user'), "content": getattr(m, 'content', str(m))} for m in messages]
        else:
            msg_dicts = [{"role": "user", "content": str(messages)}]

        # Get model name and infer agent_name from policy mapping
        model_name = self.model

        # Infer agent_name
        agent_name = None
        for a_name, p_name in _agent_policy_mapping.items():
            if p_name == model_name:
                agent_name = a_name
                break

        # Call patched generate with only messages and agent_name
        response_text, prompt_tokens, completion_tokens = await _patched_generate(
            messages=msg_dicts,
            agent_name=agent_name,
            **kwargs
        )

        # Return in llamaindex format
        from llama_index.core.base.llms.types import ChatResponse, ChatMessage
        message = ChatMessage(role="assistant", content=response_text)
        # LlamaIndex ChatResponse supports raw field for additional metadata
        return ChatResponse(
            message=message,
            raw={"usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}
        )
    
    OpenAI.achat = patched_achat
    _patched = True
    print("[Patch] Patched llamaindex OpenAI.achat")


def patch_all(
    server_address_dict: Dict[str, Union[str, List[str]]],
    tokenizer_dict: Dict[str, any],
    ppo_trainer_config_dict: Dict[str, any],
    agent_policy_mapping: Dict[str, str],
    agent_framework: str = "autogen",
    agent_address_mapping: Optional[Dict[str, str]] = None,
    agent_lora_mapping: Optional[Dict[str, str]] = None,
    agent_config_dict: Optional[Dict[str, any]] = None,
    processor_dict: Optional[Dict[str, any]] = None
):
    """
    Apply patches for specified agent framework.

    Args:
        server_address_dict: {policy_name: address or [addresses]}
        tokenizer_dict: {policy_name: tokenizer}
        ppo_trainer_config_dict: {policy_name: ppo_config}
        agent_policy_mapping: {agent_name: policy_name}
        agent_framework: Framework to patch - "autogen"/"ag2"/"langchain"/"langgraph"/"llamaindex"
        agent_address_mapping: {agent_name: vLLM_address} - direct mapping for routing
        agent_lora_mapping: {agent_name: lora_id} - mapping for LoRA adapters
        agent_config_dict: {agent_name: agent_config} - agent configurations
        processor_dict: {policy_name: processor} - processors for multimodal models
    """
    global _server_address_dict, _tokenizer_dict, _ppo_trainer_config_dict, _agent_policy_mapping, _agent_address_mapping, _agent_lora_mapping, _agent_config_dict, _processor_dict, _patched
    _server_address_dict = server_address_dict
    _tokenizer_dict = tokenizer_dict
    _ppo_trainer_config_dict = ppo_trainer_config_dict
    _agent_policy_mapping = agent_policy_mapping
    _agent_address_mapping = agent_address_mapping or {}
    _agent_lora_mapping = agent_lora_mapping or {}
    _agent_config_dict = agent_config_dict or {}
    _processor_dict = processor_dict or {}

    print(f"[Patch] Initialized context with policies: {list(server_address_dict.keys())}")
    print(f"[Patch] Agent policy mapping: {agent_policy_mapping}")
    print(f"[Patch] Agent address mapping: {agent_address_mapping}")
    print(f"[Patch] Agent lora mapping: {agent_lora_mapping}")
    print(f"[Patch] Agent framework: {agent_framework}")
    
    # Reset patched flag to allow re-patching
    _patched = False
    
    # Apply framework-specific patch
    framework_lower = agent_framework.lower()
    if framework_lower == "autogen":
        patch_autogen()
    elif framework_lower == "ag2":
        patch_ag2()
    elif framework_lower == "langchain":
        patch_langchain()
    elif framework_lower == "langgraph":
        patch_langgraph()
    elif framework_lower == "llamaindex":
        patch_llamaindex()
    else:
        raise ValueError(f"Unsupported agent_framework: {agent_framework}. "
                        f"Supported: autogen, ag2, langchain, langgraph, llamaindex")
    
    print(f"[Patch] All patches applied for framework: {agent_framework}")


def wrap_autogen_graph(graph_callable):
    """
    Wrap an autogen graph callable to track LLM request hops.

    Args:
        graph_callable: The graph's main() function or callable

    Returns:
        Wrapped callable with hop_idx tracking
    """
    @functools.wraps(graph_callable)
    async def wrapped_graph(*args, **kwargs):
        reset_hop_idx()
        print(f"[Patch] Starting autogen graph execution")
        result = await graph_callable(*args, **kwargs)
        print(f"[Patch] Graph completed after {get_hop_idx()} hops (LLM requests)")
        return result

    return wrapped_graph
