"""Agent node that wraps an AI agent with tool calling capabilities."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import re
from typing import List, Dict, Any, Optional, Tuple
import json
from workflow.core import WorkflowNode, Context, Message, MessageType, ToolRegistry
from utils.BaseOpenAI import AIClient
from utils.conversation_logger import get_global_tracker


# Instruction appended to every agent's system prompt so they produce
# a final answer in a parseable format.
# NOTE: Keep this minimal.  Verbose summary requests (Approach / Confidence /
# Key-steps) can hurt accuracy, and multi-agent language ("Inter-Agent",
# "downstream agents") triggers designer-mode in fine-tuned models.
DELIVERY_INSTRUCTION_MATH = (
    "\n\nAfter solving, output two things:\n"
    "1. Wrap the key reasoning steps you want to share with the next agent "
    "in <delivery>...</delivery> tags.\n"
    "2. State your final answer as: FINAL ANSWER: \\boxed{your_answer}"
)

DELIVERY_INSTRUCTION_CODE = (
    "\n\nAfter solving, output two things:\n"
    "1. Wrap the key reasoning steps you want to share with the next agent "
    "in <delivery>...</delivery> tags.\n"
    "2. Wrap your complete solution code in <solution>...</solution> tags. "
    "Only include the final runnable code inside <solution>, no explanations."
)


def _get_delivery_instruction() -> str:
    """Return the delivery instruction appropriate for the current task type."""
    task_type = os.getenv("TASK_TYPE", "math").lower()
    if task_type == "code":
        return DELIVERY_INSTRUCTION_CODE
    return DELIVERY_INSTRUCTION_MATH


def _extract_delivery(text: str) -> Optional[str]:
    """Extract the compact delivery block from an agent response.

    Priority (agent-chosen content first, then fixed-format fallbacks):
      1. ``<delivery>...</delivery>`` — agent explicitly chose what to share
      2. ``FINAL ANSWER: \\boxed{...}`` (math fallback)
      3. ``<solution>...</solution>`` (code fallback)

    Returns the extracted string, or None if nothing matched.
    """
    # 1. Agent-chosen delivery (highest priority)
    match = re.search(r"<delivery>(.*?)</delivery>", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Math fallback: FINAL ANSWER: \boxed{...}
    match = re.search(r"FINAL ANSWER:\s*(.*)", text)
    if match:
        return match.group(1).strip()

    # 3. Code fallback: <solution>...</solution>
    match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if match:
        return match.group(0).strip()

    return None


# ---------------------------------------------------------------------------
# Inter-node response compaction
# ---------------------------------------------------------------------------
# Caps on what we embed inline when one node's output is fed into another
# node's prompt (Reflection / Ensemble._consensus / Debate). Without
# compression these prompts blow past max_prompt_length, triggering silent
# left-truncation that drops the original question. Both task types share the
# same path: <delivery> is the canonical inter-node channel; deliverable
# salvage (boxed answer / code block) handles the case where the model
# skipped the <delivery> tag.

_RESPONSE_COMPACT_MAX_CHARS = 3000
_CRITIQUE_COMPACT_MAX_CHARS = 1500
_PROMPT_LENGTH_WARN_CHARS = 6000
_TAIL_REASONING_CHARS = 800
_LAST_RESORT_TAIL_CHARS = 4000


def _salvage_deliverable(text: str, task_type: str) -> Tuple[Optional[str], int]:
    """Find the final deliverable in an agent response, ignoring <delivery>.

    For math: last ``\\boxed{...}`` with brace-counting (handles nested {}).
    For code: last ``<solution>...</solution>``, ``<code>...</code>``,
              ``` ```python ... ``` ```, or generic ``` ``` ... ``` ``` block.

    Returns (deliverable_str, start_index_in_text), or (None, -1).
    """
    if not text:
        return None, -1

    if task_type == "code":
        for pattern, wrap in [
            (r"<solution>\s*(.*?)\s*</solution>", "<solution>\n{}\n</solution>"),
            (r"<code>\s*(.*?)\s*</code>", "<code>\n{}\n</code>"),
            (r"```python\s*(.*?)```", "```python\n{}\n```"),
            (r"```\s*(.*?)```", "```\n{}\n```"),
        ]:
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if matches:
                last = matches[-1]
                inner = last.group(1).strip()
                if inner:
                    return wrap.format(inner), last.start()
        return None, -1

    # math: brace-counting on last \boxed{...}
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None, -1
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        content = text[start:i - 1]
        return "\\boxed{" + content + "}", idx
    return None, -1


def _last_resort_tail(text: str) -> str:
    """Keep the tail of the response when nothing structured was found.

    Most-recent reasoning is more useful than the head (which usually
    restates the question, already in the prompt). Char-based budget;
    the downstream hard cap enforces the final size.
    """
    if len(text) <= _LAST_RESORT_TAIL_CHARS:
        return text
    return "...[earlier content omitted]...\n" + text[-_LAST_RESORT_TAIL_CHARS:]


def _compact_response(text: str) -> str:
    """Compress an agent response for embedding in another node's prompt.

    Layered salvage (works for both math and code):
      Layer 0 — short responses pass through unchanged.
      Layer 1 — extract the <delivery> block (the agent's chosen summary).
      Layer 2 — independently salvage the final deliverable (\\boxed{} for
                math, <solution>/code-block for code).
      Layer 3 — when no <delivery> but a deliverable was salvaged, prepend
                the last _TAIL_REASONING_CHARS chars before it.
      Layer 4 — when nothing structured was found, keep just the response tail.

    The result is hard-capped at _RESPONSE_COMPACT_MAX_CHARS.
    """
    if not text:
        return ""

    text = text.strip()

    if len(text) <= 10000:
        return text

    task_type = os.getenv("TASK_TYPE", "math").lower()

    delivery = _extract_delivery(text)
    # If <delivery> matched the math/code fallback inside _extract_delivery
    # (boxed-only or <solution>-only), treat as "no real delivery" so that
    # Layer 3 can attach tail reasoning.
    if delivery and (
        delivery.startswith("\\boxed{")
        or delivery.startswith("<solution>")
    ):
        delivery = None

    deliverable, deliverable_pos = _salvage_deliverable(text, task_type)

    if delivery and deliverable:
        compact = delivery + "\n\n" + deliverable
    elif delivery:
        compact = delivery
    elif deliverable:
        tail_start = max(0, deliverable_pos - _TAIL_REASONING_CHARS)
        tail = text[tail_start:deliverable_pos].strip()
        prefix = "...[earlier reasoning omitted]...\n" if tail_start > 0 else ""
        compact = (prefix + tail + "\n" + deliverable).strip()
    else:
        compact = _last_resort_tail(text)

    if len(compact) > _RESPONSE_COMPACT_MAX_CHARS:
        half = _RESPONSE_COMPACT_MAX_CHARS // 2
        compact = compact[:half] + "\n...[truncated]...\n" + compact[-half:]

    return compact.strip()


def _compact_critique(text: str) -> str:
    """Cap a critic's critique for embedding in a downstream prompt.

    Keep head (where major errors typically appear) and tail (where the
    final verdict typically appears).
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= _CRITIQUE_COMPACT_MAX_CHARS:
        return text
    return text[:500] + "\n...[truncated critique]...\n" + text[-1000:]


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

        # Auto-inject delivery format instruction (task-type aware)
        system_prompt += _get_delivery_instruction()

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

        # max_turns semantics: number of REACTIVE turns after the first tool call.
        # Loop runs (max_turns + 1) iterations: 1 initial call + max_turns reactive
        # turns. If the agent is still tool-calling when the loop exits, the
        # forced-finalization branch below adds one final non-tool answer.
        for turn in range(self.max_turns + 1):
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

        # Max turns reached, force final answer.
        # Use task-aware phrasing: for code, hint at <solution> wrap if the agent's
        # role is to produce code (critic/verifier agents are free to give critique
        # text instead — the wording is permissive on purpose).
        _task_type = os.getenv("TASK_TYPE", "math").lower()
        if _task_type == "code":
            _force_final_msg = (
                "Stop calling tools and produce your final response now. "
                "If your role is to provide the solution code, wrap it in "
                "<solution>...</solution> tags."
            )
        else:
            _force_final_msg = "Please provide your final answer now without using any tools."
        messages.append({
            "role": "user",
            "content": _force_final_msg,
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
            print(f"[AGENT RESPONSE]: {response}")
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
