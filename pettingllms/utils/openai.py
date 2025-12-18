"""
Core OpenAI API Patch Module for PettingLLMs

Patch autogen and langchain LLM engines to use llm_async_generate from async_generate.py
"""

import asyncio
import contextvars
import functools
from typing import Dict, List, Optional, Union, Callable
import random
import numpy as np


# Model client factory functions for different agent frameworks
def _create_autogen_client(policy_name: str, address: str, **kwargs):
    """Create AutoGen OpenAIChatCompletionClient with dataproto container."""
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

    # Create dataproto container for collecting generated dataprotos
    dataproto_container = {
        'dataprotos': [],  # List to store dataprotos from llm_async_generate
        'env_final_reward': None  # Will store the final reward from environment
    }

    # Attach the container to the client
    client._dataproto_container = dataproto_container

    return client



def _create_langchain_client(policy_name: str, address: str, **kwargs):
    """Create LangChain ChatOpenAI client with dataproto container."""
    from langchain_openai import ChatOpenAI
    client = ChatOpenAI(
        model_name=policy_name,
        openai_api_key="dummy",
        openai_api_base=address,
    )

    # Create dataproto container for collecting generated dataprotos
    dataproto_container = {
        'dataprotos': [],  # List to store dataprotos from llm_async_generate
        'env_final_reward': None  # Will store the final reward from environment
    }

    # Attach the container to the client
    client._dataproto_container = dataproto_container

    return client


def _create_llamaindex_client(policy_name: str, address: str, **kwargs):
    """Create LlamaIndex OpenAI client with dataproto container."""
    from llama_index.llms.openai import OpenAI
    client = OpenAI(
        model=policy_name,
        api_key="dummy",
        api_base=address,
    )

    # Create dataproto container for collecting generated dataprotos
    dataproto_container = {
        'dataprotos': [],  # List to store dataprotos from llm_async_generate
        'env_final_reward': None  # Will store the final reward from environment
    }

    # Attach the container to the client
    client._dataproto_container = dataproto_container

    return client


# Registry of model client factory functions by framework
MODEL_CLIENT_FACTORY: Dict[str, Callable] = {
    "autogen": _create_autogen_client,
    "langchain": _create_langchain_client,
    "langgraph": _create_langchain_client,  # LangGraph uses same client as LangChain
    "llamaindex": _create_llamaindex_client,
}


def create_model_client(agent_framework: str, policy_name: str, address: str, **kwargs):
    """
    Create a model client based on the agent framework.

    Args:
        agent_framework: Framework name ('autogen', 'langchain', 'langgraph', 'llamaindex')
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
        agent_framework: Framework name ('autogen', 'langchain', 'langgraph', 'llamaindex')

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
_patched: bool = False

# Per-flow state (isolated per asyncio task)
_hop_idx_var: contextvars.ContextVar[int] = contextvars.ContextVar("hop_idx", default=0)
_rollout_idx_var: contextvars.ContextVar[int] = contextvars.ContextVar("rollout_idx", default=0)
_env_idx_var: contextvars.ContextVar[int] = contextvars.ContextVar("env_idx", default=0)
_trajectory_store_var: contextvars.ContextVar[Optional[Dict[tuple, tuple]]] = contextvars.ContextVar("trajectory_store", default=None)


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
    return _rollout_idx_var.get()


def get_env_idx() -> int:
    """Get current environment index."""
    return _env_idx_var.get()


def get_hop_idx() -> int:
    """Get current hop index (incremented each time an LLM request is made)."""
    return _hop_idx_var.get()


def increment_hop_idx():
    """Increment hop index when an LLM request is made."""
    current_idx = _hop_idx_var.get()
    next_idx = current_idx + 1
    _hop_idx_var.set(next_idx)
    rollout_idx = get_rollout_idx()
    print(f"[Patch] Hop index incremented from {current_idx} to {next_idx} for rollout={rollout_idx}")


def reset_hop_idx():
    """Reset hop index to 0 for new graph execution."""
    rollout_idx = get_rollout_idx()
    env_idx = get_env_idx()
    _hop_idx_var.set(0)
    _trajectory_store_var.set({})  # Clear trajectory store for new rollout
    print(f"[Patch] Hop index reset to 0 for rollout={rollout_idx}, env={env_idx}, trajectory store cleared")


def clear_trajectory_store():
    """Clear the trajectory store."""
    _trajectory_store_var.set({})
    print(f"[Patch] Trajectory store cleared")


def start_flow_context(rollout_idx: int, env_idx: int):
    """
    Initialize per-flow context (rollout/env) and reset hop/trajectory state for this task.
    Each rollout tracks its own hop count independently.
    """
    _rollout_idx_var.set(rollout_idx)
    _env_idx_var.set(env_idx)
    _hop_idx_var.set(0)
    _trajectory_store_var.set({})
    print(f"[Patch] New flow context started: rollout={rollout_idx}, env={env_idx}, hop=0")



def _get_trajectory_store_ref() -> Dict[tuple, tuple]:
    store = _trajectory_store_var.get()
    if store is None:
        store = {}
        _trajectory_store_var.set(store)
    return store


def get_trajectory_store() -> Dict[tuple, tuple]:
    """
    Get collected trajectories for current rollout.

    Returns:
        Dict mapping (rollout_idx, hop_idx, policy_name) to (output_dpr, response)
    """
    return _get_trajectory_store_ref().copy()


def get_server_address(policy_name: str) -> str:
    """Get vLLM server address for policy."""
    addresses = _server_address_dict[policy_name]
    if isinstance(addresses, list):
        return random.choice(addresses)
    return addresses


def _auto_init_from_env():
    """Auto-initialize patch context from environment variables."""
    import os
    global _server_address_dict, _tokenizer_dict, _ppo_trainer_config_dict, _agent_policy_mapping, _agent_address_mapping
    
    api_base = os.getenv("API_BASE", "http://localhost:8000")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4")
    
    from transformers import AutoTokenizer
    
    class DummyTokenizer:
        def __init__(self, model_name):
            self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        def __getattr__(self, name):
            return getattr(self._tokenizer, name)
    
    class DummyConfig:
        def __init__(self, model_path):
            self.actor_rollout_ref = type('obj', (object,), {
                'model': type('obj', (object,), {'path': model_path})()
            })()
            self.data = type('obj', (object,), {'max_prompt_length': 4096})()
    
    tokenizer = DummyTokenizer(chat_model)
    ppo_config = DummyConfig(chat_model)
    
    _server_address_dict = {"default_policy": api_base}
    _tokenizer_dict = {"default_policy": tokenizer}
    _ppo_trainer_config_dict = {"default_policy": ppo_config}
    _agent_policy_mapping = {}
    _agent_address_mapping = {}
    
    print(f"[Patch] Auto-initialized from env: API_BASE={api_base}, CHAT_MODEL={chat_model}")


async def _patched_generate(
    messages: List[Dict[str, str]],
    agent_name: Optional[str] = None,
    client: Optional[any] = None,
    **kwargs
) -> tuple[str, int, int, Optional[List[int]]]:
    """
    Core patched generate function that calls llm_async_generate.
    Each call to this function represents one "hop" in the agent graph.

    This function is framework-agnostic - it only takes messages and agent_name.
    All other mappings (policy, address, lora_id, config) are resolved internally.

    Args:
        messages: Chat messages
        agent_name: Agent name for routing and reward attribution
        client: The model client object (to access dataproto container)
        **kwargs: Additional generation parameters

    Returns:
        Tuple of (response_text, prompt_tokens, completion_tokens, token_ids)
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

    # Get current hop index and increment it BEFORE generation
    # This ensures each LLM call gets a unique hop_idx for this rollout
    hop_idx = get_hop_idx()
    increment_hop_idx()

    rollout_idx = get_rollout_idx()
    env_idx = get_env_idx()

    print(f"[Patch] Starting LLM request: rollout={rollout_idx}, env={env_idx}, hop={hop_idx}, agent={agent_name}")

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
        application_id=f"graph_r{rollout_idx}_h{hop_idx}",
        env_idx=env_idx,
        policy_name=policy_name,
        timeout=kwargs.get('timeout', 60.0),
        mode=kwargs.get('mode', 'inference'),
        lora_id=lora_id,
        agent_config=agent_config,
    )

    # Calculate token counts and extract token IDs from output_dpr
    prompt_tokens = 0
    completion_tokens = 0
    token_ids = None

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
            # Count non-padding tokens in responses and extract token IDs
            response_ids = output_dpr.batch['responses']
            if hasattr(response_ids, 'shape'):
                # Count actual tokens (excluding padding)
                if len(response_ids.shape) > 1:
                    completion_tokens = int(response_ids.shape[-1])
                    # Extract token IDs (first sample if batch)
                    token_ids = response_ids[0].tolist() if response_ids.shape[0] > 0 else []
                elif len(response_ids.shape) > 0:
                    completion_tokens = int(response_ids.shape[0])
                    token_ids = response_ids.tolist()

    # Store trajectory with agent_name for later reward attribution
    # Note: reward will be added later by the environment step
    if output_dpr is not None:
        # Add agent_name and hop_idx to output_dpr for trajectory collection
        output_dpr.non_tensor_batch["agent_name"] = np.array([agent_name or "unknown"], dtype=object)
        output_dpr.non_tensor_batch["hop_idx"] = np.array([hop_idx], dtype=np.int32)

        # Store in trajectory store
        _trajectory_store = _get_trajectory_store_ref()
        key = (rollout_idx, hop_idx, policy_name)
        _trajectory_store[key] = (output_dpr, response)
        print(f"[Patch] Stored trajectory for key={key}, rollout={rollout_idx}, env={env_idx}, hop={hop_idx}, agent={agent_name}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
        client._dataproto_container['dataprotos'].append(output_dpr)
        print(f"[Patch] Added dataproto to client container for agent={agent_name}, total dataprotos={len(client._dataproto_container['dataprotos'])}")

    return response, prompt_tokens, completion_tokens, token_ids


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

        # Call patched generate with messages, agent_name, and client
        response_text, prompt_tokens, completion_tokens, token_ids = await _patched_generate(
            messages=messages,
            agent_name=agent_name,
            client=self,
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

        # Call patched generate with messages, agent_name, and client
        response_text, prompt_tokens, completion_tokens, token_ids = await _patched_generate(
            messages=msg_dicts,
            agent_name=agent_name,
            client=self,
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

        # Call patched generate with messages, agent_name, and client
        response_text, prompt_tokens, completion_tokens, token_ids = await _patched_generate(
            messages=msg_dicts,
            agent_name=agent_name,
            client=self,
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
        agent_framework: Framework to patch - "autogen"/"langchain"/"langgraph"/"llamaindex"
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
    elif framework_lower == "langchain":
        patch_langchain()
    elif framework_lower == "langgraph":
        patch_langgraph()
    elif framework_lower == "llamaindex":
        patch_llamaindex()
    else:
        raise ValueError(f"Unsupported agent_framework: {agent_framework}. "
                        f"Supported: autogen, langchain, langgraph, llamaindex")
    
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
        # Reset hop counter for this rollout
        reset_hop_idx()
        rollout_idx = get_rollout_idx()
        env_idx = get_env_idx()
        print(f"[Patch] Starting autogen graph execution for rollout={rollout_idx}, env={env_idx}")
        result = await graph_callable(*args, **kwargs)
        final_hops = get_hop_idx()
        print(f"[Patch] Graph completed after {final_hops} hops (LLM requests) for rollout={rollout_idx}, env={env_idx}")
        return result

    return wrapped_graph


def get_client_dataprotos(client) -> List:
    """
    Get all dataprotos collected by a client.

    Args:
        client: The model client object

    Returns:
        List of dataprotos collected from llm_async_generate calls
    """
    if hasattr(client, '_dataproto_container'):
        return client._dataproto_container['dataprotos']
    return []


def get_client_env_reward(client) -> Optional[float]:
    """
    Get the environment final reward stored in a client.

    Args:
        client: The model client object

    Returns:
        The environment final reward, or None if not set
    """
    if hasattr(client, '_dataproto_container'):
        return client._dataproto_container['env_final_reward']
    return None


def set_client_env_reward(client, reward: float):
    """
    Set the environment final reward for a client.

    Args:
        client: The model client object
        reward: The environment final reward to store
    """
    if hasattr(client, '_dataproto_container'):
        client._dataproto_container['env_final_reward'] = reward
        print(f"[Patch] Set env_final_reward={reward} for client")
    else:
        print(f"[Patch] Warning: Client does not have _dataproto_container")


def clear_client_dataprotos(client):
    """
    Clear all dataprotos collected by a client.

    Args:
        client: The model client object
    """
    if hasattr(client, '_dataproto_container'):
        client._dataproto_container['dataprotos'] = []
        client._dataproto_container['env_final_reward'] = None
        print(f"[Patch] Cleared dataproto container for client")


def merge_dataprotos_with_reward(dataprotos: List, reward: float):
    """
    Merge a list of dataprotos and add the environment final reward to each.

    Args:
        dataprotos: List of DataProto objects
        reward: The environment final reward

    Returns:
        List of dataprotos with env_final_reward added
    """
    for dpr in dataprotos:
        if dpr is not None and hasattr(dpr, 'non_tensor_batch'):
            dpr.non_tensor_batch["env_final_reward"] = np.array([reward], dtype=np.float32)
    return dataprotos
