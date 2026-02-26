"""Agent node that wraps an AI agent with tool calling capabilities."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import re
from typing import List, Dict, Any, Optional
import json
from workflow.core import WorkflowNode, Context, Message, MessageType, ToolRegistry
from utils.BaseOpenAI import AIClient
from utils.conversation_logger import get_global_tracker


# Instruction appended to every agent's system prompt so they produce
# a final answer in a parseable format.
# NOTE: Keep this minimal.  Verbose summary requests (Approach / Confidence /
# Key-steps) can hurt accuracy, and multi-agent language ("Inter-Agent",
# "downstream agents") triggers designer-mode in fine-tuned models.
DELIVERY_INSTRUCTION = (
    "\n\nAfter solving, restate your final answer clearly as: "
    "FINAL ANSWER: \\boxed{your_answer}"
)


def _extract_delivery(text: str) -> Optional[str]:
    """Extract the compact delivery block from an agent response.

    Tries several formats (newest first):
      1. ``FINAL ANSWER: \\boxed{...}``  (V3 – current default)
      2. ``Approach: / Answer: / Confidence:`` trailing block (V1 – legacy)
      3. ``<delivery>...</delivery>`` XML tags (oldest legacy)

    Returns the extracted string, or None if nothing matched.
    """
    # V3: FINAL ANSWER: \boxed{...}
    match = re.search(r"FINAL ANSWER:\s*(.*)", text)
    if match:
        return match.group(1).strip()

    # V1 legacy: trailing Approach/Answer/Confidence block
    match = re.search(
        r"(Approach:.*(?:\nAnswer:.*)?(?:\nConfidence:.*)?)\s*$",
        text,
    )
    if match:
        return match.group(1).strip()

    # Oldest legacy: <delivery>...</delivery> XML tags
    match = re.search(r"<delivery>(.*?)</delivery>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


class AgentNode(WorkflowNode):
    """A node that wraps an AI agent with tool calling capabilities.

    This node:
    - Uses structured message passing instead of string parsing
    - Automatically handles tool calls through ToolRegistry
    - Manages conversation history
    - Provides clean interface for custom agents
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tool_registry: Optional[ToolRegistry] = None,
        ai_client: Optional[AIClient] = None,
        max_turns: int = 10,
        enable_conversation_logging: bool = True,
        **kwargs
    ):
        """Initialize the agent node.

        Args:
            name: Node name
            system_prompt: System prompt for the agent
            tool_registry: Registry of available tools
            ai_client: AI client for making API calls
            max_turns: Maximum number of turns for tool calling
            enable_conversation_logging: Whether to log conversations to ShareGPT format
            **kwargs: Additional configuration
        """
        super().__init__(name, **kwargs)

        self.system_prompt = system_prompt
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_turns = max_turns
        self.enable_conversation_logging = enable_conversation_logging

        # Store ai_client if provided, otherwise will use workflow's client
        self.ai_client = ai_client
        self._client_from_workflow = (ai_client is None)

    def _build_initial_messages(self, context: Context) -> List[Dict[str, str]]:
        """Build initial message list from context.

        Builds a shared-context view: the agent always sees the original question
        plus all previous agents' outputs, enabling full information sharing.

        Args:
            context: Workflow context

        Returns:
            List of messages in OpenAI format
        """
        # Check if we need to add tool descriptions to system prompt (for vLLM)
        system_prompt = self.system_prompt

        # In evaluate mode, use prompt-based tool calling instead of native API tools
        if os.getenv("EVALUATE_MODE") == "True" and self.tool_registry.list_tools():
            tools = self.tool_registry.get_tool_schemas()
            system_prompt = (
                self.system_prompt
                + f"\n\nYou have access to the following tools: {json.dumps(tools, indent=2)}.\n"
                "When you need to call a tool, output it in the following format:\n"
                "<tool_call>{\n"
                '  "name": "tool_name",\n'
                '  "arguments": {\n'
                '    "argument_name": "argument_value"\n'
                "  }\n"
                "}</tool_call>\n"
                "Make sure to include ALL required arguments in the tool call.\n"
            )

        # Auto-inject delivery format instruction
        system_prompt += DELIVERY_INSTRUCTION

        # In evaluate mode (AIME), remind agents the answer must be an integer
        # and prevent designer-format output
        if os.getenv("EVALUATE_MODE") == "True":
            system_prompt += (
                "\nIMPORTANT: The answer MUST be an integer between 0 and 999."
                " If you get a non-integer, re-examine your work."
                "\nDo NOT output Problem Type, Problem Analysis, or Workflow Pattern."
                " You are a math solver, not a designer. Solve the problem directly."
            )

        # Add /no_think suffix if enabled (only during execution, not design)
        if os.getenv("EXECUTOR_NO_THINK") == "True":
            system_prompt = system_prompt + " /no_think"

        messages = [{"role": "system", "content": system_prompt}]

        # --- Shared Context: original question + delivery-extracted prior outputs ---
        original_input = None
        agent_responses = []
        for msg in context.messages:
            if msg.message_type == MessageType.USER_INPUT and original_input is None:
                original_input = msg
            elif msg.message_type == MessageType.AGENT_RESPONSE:
                agent_responses.append(msg)

        if original_input:
            if isinstance(original_input.content, str):
                content = original_input.content
            elif isinstance(original_input.content, dict):
                content = json.dumps(original_input.content, ensure_ascii=False)
            else:
                content = str(original_input.content)

            if agent_responses:
                # Build shared-context: original question + extracted deliveries
                parts = [f"**Original Question:**\n{content}\n"]
                parts.append("**Previous Agents' Analysis:**\n")
                for resp in agent_responses:
                    sender = resp.sender or "Agent"
                    resp_content = resp.content if isinstance(resp.content, str) else str(resp.content)
                    # Prefer <delivery> block; fall back to tail truncation
                    delivery = _extract_delivery(resp_content)
                    if delivery:
                        summary = delivery
                    elif len(resp_content) > 1200:
                        summary = (
                            resp_content[:200] + "\n...(reasoning omitted)...\n"
                            + resp_content[-800:]
                        )
                    else:
                        summary = resp_content
                    parts.append(f"[{sender}]:\n{summary}\n")
                parts.append(
                    "Based on the original question and the analysis above, "
                    "provide your own solution."
                )
                messages.append({"role": "user", "content": "\n".join(parts)})
            else:
                # First agent in the chain — just the question
                messages.append({"role": "user", "content": content})
        else:
            # Fallback: use latest message (for non-workflow direct calls)
            latest_msg = context.get_latest_message()
            if latest_msg:
                if isinstance(latest_msg.content, str):
                    content = latest_msg.content
                elif isinstance(latest_msg.content, dict):
                    content = json.dumps(latest_msg.content, ensure_ascii=False)
                else:
                    content = str(latest_msg.content)
                messages.append({"role": "user", "content": content})

        return messages

    @staticmethod
    def _extract_python_code_block(response: str):
        """Extract Python code from ```python ... ``` blocks in the response.

        Returns the first matched code string, or None if no block found.
        """
        match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _handle_tool_calls(self, messages: List[Dict[str, str]]) -> tuple[str, int, List[Dict[str, str]]]:
        """Handle agent tool calls in a loop.

        Supports two tool-calling formats:
          1. ```python ... ``` blocks  –  code is extracted and executed
             directly via the ``python_execute`` tool in the registry (preferred).
          2. <tool_call>{...}</tool_call> JSON tags  –  legacy / generic tools.

        Args:
            messages: Current message history

        Returns:
            Tuple of (final_response, total_tokens, full_messages)
        """
        total_tokens = 0

        # Get tool schemas if available
        tools = None
        evaluate_mode = os.getenv("EVALUATE_MODE") == "True"

        # Only use native tool calling if NOT in evaluate mode (vLLM doesn't support it)
        if not evaluate_mode and self.tool_registry.list_tools():
            tools = self.tool_registry.get_tool_schemas()

        for turn in range(self.max_turns):
            # Make API call
            # In evaluate mode, never pass tools (use prompt-based approach instead)
            if tools and not evaluate_mode:
                response, prompt_tokens, completion_tokens = self.ai_client.chat(
                    messages,
                    tools=tools
                )
            else:
                response, prompt_tokens, completion_tokens = self.ai_client.chat(messages)

            total_tokens += prompt_tokens + completion_tokens

            # ----- Format 1: ```python code blocks (preferred for Python tool) -----
            python_code = self._extract_python_code_block(response)
            if python_code is not None:
                messages.append({"role": "assistant", "content": response})

                try:
                    # Execute via python_execute tool if registered, else direct call
                    if self.tool_registry.get_tool("python_execute"):
                        tool_result = self.tool_registry.call_tool(
                            "python_execute", {"code": python_code}
                        )
                    else:
                        # Fallback: try to import and run MathEnvironment directly
                        from pettingllms.multi_agent_env.autoevol.utils.environments.math_env import python_execute
                        tool_result = python_execute(python_code)

                    self.logger.info(f"Executed ```python block ({len(python_code)} chars)")
                except Exception as e:
                    self.logger.error(f"Error executing python code block: {e}")
                    tool_result = f"Error executing code: {type(e).__name__}: {str(e)}"

                messages.append({
                    "role": "user",
                    "content": tool_result
                })
                continue

            # ----- Format 2: <tool_call> JSON tags (legacy / generic tools) -----
            if "<tool_call>" in response and "</tool_call>" in response:
                # Add assistant's tool call to messages
                messages.append({"role": "assistant", "content": response})

                try:
                    tool_call_str = response.split("<tool_call>")[1].split("</tool_call>")[0].strip()

                    # Check for empty tool call
                    if not tool_call_str:
                        self.logger.error("Empty tool call content received")
                        messages.append({
                            "role": "user",
                            "content": "Error: Empty tool call. Please provide a valid tool call with proper JSON format, or provide your final answer without using tools."
                        })
                        continue

                    tool_call = json.loads(tool_call_str)
                    tool_name = tool_call.get("name")
                    # Accept both "parameters" and "arguments" (OpenAI uses "arguments")
                    tool_params = tool_call.get("parameters") or tool_call.get("arguments") or {}

                    # Validate tool call structure
                    if not tool_name:
                        self.logger.error("Tool call missing 'name' field")
                        messages.append({
                            "role": "user",
                            "content": "Error: Tool call is missing the 'name' field. Please provide a valid tool call in the format: <tool_call>{\"name\": \"tool_name\", \"parameters\": {...}}</tool_call>"
                        })
                        continue

                    self.logger.info(f"Calling tool: {tool_name} with params: {tool_params}")

                    # Execute tool
                    tool_result = self.tool_registry.call_tool(tool_name, tool_params)

                    # Add tool result to messages
                    tool_response = f"Tool '{tool_name}' returned: {tool_result}"
                    messages.append({
                        "role": "user",
                        "content": tool_response
                    })

                    self.logger.debug(f"Tool response: {tool_response[:200]}...")

                except json.JSONDecodeError as e:
                    self.logger.error(f"Invalid JSON in tool call: {e}")
                    messages.append({
                        "role": "user",
                        "content": f"Error: Invalid JSON format in tool call. Please ensure your tool call uses valid JSON format: <tool_call>{{\"name\": \"tool_name\", \"parameters\": {{...}}}}</tool_call>\nJSON error: {str(e)}"
                    })
                except Exception as e:
                    self.logger.error(f"Error executing tool: {e}")
                    error_msg = f"Error executing tool: {str(e)}"
                    messages.append({
                        "role": "user",
                        "content": error_msg
                    })

                # Continue to next turn
                continue

            # No tool call in response - this is the final answer
            # Add final response to messages
            messages.append({"role": "assistant", "content": response})

            # Return final response and full message history
            return response, total_tokens, messages

        # Max turns reached, force final answer
        messages.append({
            "role": "user",
            "content": "Please provide your final answer now without using any tools."
        })
        response, pt, ct = self.ai_client.chat(messages)
        total_tokens += pt + ct
        messages.append({"role": "assistant", "content": response})

        return response, total_tokens, messages

    def process(self, context: Context) -> Message:
        """Process the context and generate agent response.

        Args:
            context: Workflow context

        Returns:
            Message containing agent response
        """
        workflow = context.state.get('workflow')
        self.ai_client = workflow.ai_client

        # Build message history
        initial_messages = self._build_initial_messages(context)

        # Handle tool calls and get final response with full message history
        response, total_tokens, full_messages = self._handle_tool_calls(initial_messages)

        # Print agent node details to stdout (captured in output.txt)
        try:
            sys_prompt = next((m.get("content", "") for m in initial_messages if m.get("role") == "system"), "")
            user_input = next((m.get("content", "") for m in initial_messages if m.get("role") == "user"), "")
            print(f"\n========== AGENT NODE: {self.name} ==========")
            print(f"[SYSTEM PROMPT]: {sys_prompt[:500]}")
            print(f"[USER INPUT]: {user_input[:500]}")
            print(f"[AGENT RESPONSE]: {response[:2000]}")
            print(f"[TOKENS]: {total_tokens}")
            print(f"=============================================\n")
        except Exception:
            pass

        # Log conversation to ShareGPT format if enabled
        if self.enable_conversation_logging:
            try:
                tracker = get_global_tracker()
                logger = tracker.get_logger(self.name)

                # Log all messages in the conversation (including tool calls and responses)
                for msg in full_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")

                    # Map OpenAI roles to ShareGPT format
                    if role == "system":
                        sharegpt_role = "system"
                    elif role == "user":
                        sharegpt_role = "human"
                    elif role == "assistant":
                        sharegpt_role = "gpt"
                    else:
                        sharegpt_role = "human"

                    logger.add_message(sharegpt_role, content)

            except Exception as e:
                self.logger.warning(f"Failed to log conversation: {e}")

        # Store token usage in metadata
        self.logger.info(f"Total tokens used: {total_tokens}")

        # Create result message
        result = Message(
            content=response,
            message_type=MessageType.AGENT_RESPONSE,
            metadata={
                "tokens": total_tokens,
                "agent_name": self.name
            }
        )

        return result
