from openai import OpenAI
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import asyncio
from pettingllms.trainer.async_generate import convert_prompt_to_dpr, llm_async_generate

class AIClient:
    def __init__(
        self,
        api_base: str,
        api_key: str = "dummy",
        chat_model: str = "default",
        max_answer_tokens: int = 2048,
        tokenizer_path: Optional[str] = None,
        server_address: Optional[str] = None,
        max_prompt_length: Optional[int] = None,
        max_response_length: Optional[int] = None,
        enable_thinking: bool = False,
        workflow: Optional[Any] = None
    ):
        # Use "dummy" as default api_key for vLLM local server
        if not api_key:
            api_key = "dummy"
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.chat_model = chat_model
        self.max_answer_tokens = max_answer_tokens

        # verl integration parameters
        self.tokenizer = None
        self.server_address = server_address
        self.max_prompt_length = max_prompt_length or 2048
        self.max_response_length = max_response_length or 2048
        self.enable_thinking = bool(enable_thinking)
        self.workflow = workflow

        # Load tokenizer if path is provided
        #print(f"[DEBUG] AIClient.__init__: tokenizer_path = {tokenizer_path}")
        #print(f"[DEBUG] AIClient.__init__: tokenizer_path type = {type(tokenizer_path)}")

        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                #print(f"[DEBUG] AIClient: Loading tokenizer from {tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
               
            except Exception as e:
                #print(f"[ERROR] AIClient: Failed to load tokenizer from {tokenizer_path}: {e}")
                import traceback
                traceback.print_exc()
                raise
     
    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: Optional[int] = None, tools: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, int, int]:
        if self.tokenizer is None or self.server_address is None:
            return self._chat_openai(messages, temperature, max_tokens, tools)

        # Check if we should use verl vllm interface
        return self._chat_verl(messages, temperature, max_tokens)

    def _chat_openai(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int], tools: Optional[List[Dict[str, Any]]]) -> Tuple[str, int, int]:
        """OpenAI-compatible chat path used by demo/inference servers."""
        kwargs = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.max_answer_tokens,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking,
                },
            },
        }
        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        return content, prompt_tokens, completion_tokens


    def _chat_verl(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]) -> Tuple[str, int, int]:
        """verl vllm interface with DataProto."""
        # Pass the full multi-turn messages list directly so apply_chat_template
        # handles all roles (system/user/assistant/tool) with proper special tokens.
        # This avoids the old manual flattening that garbled tool results and
        # multi-turn history into a single user message.
        prompt_dict = {
            "messages": messages,  # full conversation — apply_chat_template handles formatting
            "text": "",            # required key for validation in convert_prompt_to_dpr
            "image": None,
        }

        # Convert to DataProto
        prompt_dpr = convert_prompt_to_dpr(
            tokenizer=self.tokenizer,
            processor=None,
            prompts=prompt_dict,
            max_prompt_length=self.max_prompt_length,
            multi_modal=False,
            enable_thinking=self.enable_thinking
        )

        # Create a minimal ppo_trainer_config-like object for llm_async_generate
        class PPOConfig:
            class DataConfig:
                def __init__(self, max_prompt_length, max_response_length):
                    self.max_prompt_length = max_prompt_length
                    self.max_response_length = max_response_length

            def __init__(self, max_prompt_length, max_response_length):
                self.data = self.DataConfig(max_prompt_length, max_response_length)

        ppo_config = PPOConfig(self.max_prompt_length, self.max_response_length)

        # Call llm_async_generate
        # Run async function in sync context
        try:
            loop = asyncio.get_event_loop()
            output_dpr, response = loop.run_until_complete(
                    llm_async_generate(
                        rollout_idx=0,
                        turn_idx=0,
                        agent_idx=0,
                        prompt_dpr=prompt_dpr,
                        ppo_trainer_config=ppo_config,
                        address=self.server_address,
                        model_name=self.chat_model,
                        tokenizer=self.tokenizer,
                        enable_thinking=self.enable_thinking,
                        timeout=300.0,
                        mode="test",
                        sample_num=1
                    )
                )
            #print(f"Agent response: {response}")
        except RuntimeError:
            # Create new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.close()

        # Store actual response token count for length-penalty bookkeeping.
        import numpy as np
        response_token_count = len(self.tokenizer.encode(response))
        batch_size = output_dpr.batch.batch_size[0]
        output_dpr.non_tensor_batch["response_token_count"] = np.array([response_token_count] * batch_size)

        self.workflow.dataproto_list.append(output_dpr)

        # Calculate token usage from DataProto
        prompt_tokens = int(output_dpr.batch["prompts"].shape[1])
        completion_tokens = int(output_dpr.batch["responses"].shape[1])

        return response, prompt_tokens, completion_tokens
