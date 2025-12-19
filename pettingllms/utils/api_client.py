"""
API Client Module for External LLM APIs

Supports:
- OpenAI API
- DeepSeek API
- Claude API (Anthropic)
"""

import os
import asyncio
import logging
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API client"""
    api_type: str  # "openai", "deepseek", "claude"
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = 60.0


class BaseAPIClient:
    """Base class for API clients"""

    def __init__(self, config: APIConfig):
        self.config = config

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate response from API

        Args:
            messages: List of chat messages in format [{"role": "user", "content": "..."}]
            **kwargs: Additional generation parameters

        Returns:
            Generated response text
        """
        raise NotImplementedError


class OpenAIClient(BaseAPIClient):
    """OpenAI API client"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API"""
        try:
            temperature = kwargs.get('temperature', self.config.temperature)
            max_tokens = kwargs.get('max_tokens', self.config.max_tokens)

            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class DeepSeekClient(BaseAPIClient):
    """DeepSeek API client (compatible with OpenAI API)"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            from openai import AsyncOpenAI
            # DeepSeek uses OpenAI-compatible API
            self.client = AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url or "https://api.deepseek.com",
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using DeepSeek API"""
        try:
            temperature = kwargs.get('temperature', self.config.temperature)
            max_tokens = kwargs.get('max_tokens', self.config.max_tokens)

            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise


class ClaudeClient(BaseAPIClient):
    """Claude API (Anthropic) client"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(
                api_key=config.api_key,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Claude API"""
        try:
            temperature = kwargs.get('temperature', self.config.temperature)
            max_tokens = kwargs.get('max_tokens', self.config.max_tokens)

            # Extract system message if present
            system_message = None
            user_messages = []
            for msg in messages:
                if msg.get('role') == 'system':
                    system_message = msg.get('content')
                else:
                    user_messages.append(msg)

            # Claude API format
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=user_messages,
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


def create_api_client(
    api_type: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: float = 60.0
) -> BaseAPIClient:
    """
    Factory function to create API client

    Args:
        api_type: Type of API ("openai", "deepseek", "claude")
        api_key: API key (can also be set via environment variables)
        model: Model name
        base_url: Base URL for API (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds

    Returns:
        API client instance

    Environment Variables:
        - OPENAI_API_KEY: OpenAI API key
        - DEEPSEEK_API_KEY: DeepSeek API key
        - ANTHROPIC_API_KEY: Claude/Anthropic API key
    """
    api_type_lower = api_type.lower()

    # Get API key from environment if not provided
    if api_key is None:
        if api_type_lower == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif api_type_lower == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
        elif api_type_lower == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")

    # For local vLLM server (custom base_url), API key is not required
    # Use a dummy key to bypass validation
    if api_key is None and base_url is not None:
        # Check if this is a local vLLM server (localhost or 127.0.0.1)
        if "localhost" in base_url or "127.0.0.1" in base_url:
            api_key = "dummy-key-for-local-vllm"
        else:
            raise ValueError(f"API key not provided for {api_type}. Set via argument or environment variable.")
    elif api_key is None:
        raise ValueError(f"API key not provided for {api_type}. Set via argument or environment variable.")

    # Set default models if not provided
    if model is None:
        if api_type_lower == "openai":
            model = "gpt-4o"
        elif api_type_lower == "deepseek":
            model = "deepseek-chat"
        elif api_type_lower == "claude":
            model = "claude-3-5-sonnet-20241022"

    config = APIConfig(
        api_type=api_type_lower,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )

    if api_type_lower == "openai":
        return OpenAIClient(config)
    elif api_type_lower == "deepseek":
        return DeepSeekClient(config)
    elif api_type_lower == "claude":
        return ClaudeClient(config)
    else:
        raise ValueError(f"Unsupported API type: {api_type}. Supported: openai, deepseek, claude")


async def batch_generate(
    client: BaseAPIClient,
    prompts: List[str],
    system_prompt: Optional[str] = None,
    **kwargs
) -> List[str]:
    """
    Generate responses for multiple prompts in parallel

    Args:
        client: API client instance
        prompts: List of user prompts
        system_prompt: Optional system prompt to prepend
        **kwargs: Additional generation parameters

    Returns:
        List of generated responses
    """
    tasks = []
    for prompt in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        task = client.generate(messages, **kwargs)
        tasks.append(task)

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(f"Error generating response for prompt {i}: {response}")
            results.append("")
        else:
            results.append(response)

    return results


# Example usage
if __name__ == "__main__":
    import asyncio

    async def test_api_clients():
        # Test OpenAI
        try:
            openai_client = create_api_client(
                api_type="openai",
                model="gpt-4o-mini"
            )
            messages = [{"role": "user", "content": "Hello, how are you?"}]
            response = await openai_client.generate(messages)
            print(f"OpenAI response: {response}")
        except Exception as e:
            print(f"OpenAI test failed: {e}")

        # Test DeepSeek
        try:
            deepseek_client = create_api_client(
                api_type="deepseek",
                model="deepseek-chat"
            )
            messages = [{"role": "user", "content": "Hello, how are you?"}]
            response = await deepseek_client.generate(messages)
            print(f"DeepSeek response: {response}")
        except Exception as e:
            print(f"DeepSeek test failed: {e}")

        # Test Claude
        try:
            claude_client = create_api_client(
                api_type="claude",
                model="claude-3-5-sonnet-20241022"
            )
            messages = [{"role": "user", "content": "Hello, how are you?"}]
            response = await claude_client.generate(messages)
            print(f"Claude response: {response}")
        except Exception as e:
            print(f"Claude test failed: {e}")

    asyncio.run(test_api_clients())
