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
        self.enable_thinking = False
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
        # Check if we should use verl vllm interface
        if self.tokenizer is None:
            raise ValueError(
                "AIClient.tokenizer is None. Please provide a valid tokenizer_path when creating AIClient. "
                "Check that the tokenizer was properly loaded in the AIClient.__init__ method."
            )
        if self.server_address is None:
            raise ValueError(
                "AIClient.server_address is None. Please provide a valid server_address when creating AIClient."
            )

        return self._chat_verl(messages, temperature, max_tokens)
        


    def _chat_verl(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]) -> Tuple[str, int, int]:
        """verl vllm interface with DataProto."""
        # Import verl functions
        

        # Convert messages to prompt dict
        # Extract system and user messages
        system_message = None
        user_message = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_message = content
            elif role == "user":
                user_message += content + "\n"
            elif role == "assistant":
                user_message += f"Assistant: {content}\n"

        user_message = user_message.strip()

        # Create prompt dict
        prompt_dict = {
            "text": user_message,
            "image": None,
        }
        if system_message:
            prompt_dict["system"] = system_message

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

        self.workflow.dataproto_list.append(output_dpr)

        # Calculate token usage from DataProto
        prompt_tokens = int(output_dpr.batch["prompts"].shape[1])
        completion_tokens = int(output_dpr.batch["responses"].shape[1])

        return response, prompt_tokens, completion_tokens

