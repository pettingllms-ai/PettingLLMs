# Split Policy Training: Complete Investigation

## Overview

The split policy system trains **two separate LLM policies** cooperatively:

| Role | Model | Thinking | Purpose |
|------|-------|----------|---------|
| **Designer** (policy_0) | `Mercury7353/masrl0206_notool` | Enabled | Generates Python workflow code (MAS design) |
| **AgentNode (executor)** | Same Designer vLLM server | Disabled | The workflow agents that actually solve problems. They ARE the executor -- no separate model needed. |

> **Key insight**: The "executor" is not a standalone model. It is the `AgentNode`
> instances running inside the Designer-generated workflow code. They call back into
> the Designer's vLLM server via `AIClient`. The Executor PPO trainer collects the
> DataProtos produced by those AgentNode LLM calls and optimizes accordingly.

---

## 1. End-to-End Rollout Flow

```
Math Problem (e.g. AIME24)
        |
        v
  +-----------+
  | DESIGNER  |  LLM generates Python workflow code
  +-----------+  (output wrapped in <code>```python...```</code>)
        |
        v
  +---------------------+
  | CODE PATCH & EXEC   |  Code patched (imports, AIClient injection), executed in Ray worker
  +---------------------+
        |
        v
  +------------------+
  | AGENT NODES      |  Each AgentNode calls Designer's vLLM via AIClient
  | (the executor)   |  Writes ```python code blocks -> extracted & executed -> result feedback
  +------------------+  Multi-turn loop until final answer
        |
        v
  Final Answer extracted (\boxed{...}) -> Reward computed -> PPO update
```

---

## 2. Full Rollout Example

Below is a **complete rollout** showing every step from problem to reward.

### Phase 1: Designer Generates Workflow Code

**Input to Designer LLM:**
```
System: "You are an expert in designing Multi-Agent System workflows."
User:   "Design Multi Agent System for the Question: Let S be the set of
         positive integers k such that 2 divides kC(k,2). Find |S| for
         k in {1,2,...,30}."
```

**Designer Output** (with `enable_thinking=True`):
```
<think>
This is a number theory problem. I'll use a single solver with Python code
execution -- the agent can enumerate and check divisibility with sympy.
</think>

<code>
```python
from workflow import AgentNode, Workflow, ToolRegistry
from utils.environments.math_env import MathEnvironment

# Setup Python execution tool
math_env = MathEnvironment(timeout=120)
tool_registry = ToolRegistry()
tool_registry.register(
    name="python_execute",
    func=math_env.execute,
    description="Execute Python code. Has access to numpy, scipy, sympy.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        },
        "required": ["code"]
    }
)

solver = AgentNode(
    name="MathSolver",
    system_prompt=(
        "You are an expert mathematician. Solve problems using Python code.\n\n"
        "When you need to compute something, write Python code in a ```python block.\n"
        "The code will be executed and the output returned to you.\n"
        "After getting the result, state your final answer as: FINAL ANSWER: \\boxed{answer}"
    ),
    tool_registry=tool_registry,
    max_turns=5
)

workflow = Workflow(name="math_solver")
workflow.add_node(solver)

result = workflow.run(question)
print(result.content)
```
</code>
```

**Code extraction**: `MASGenerator.update_from_model()` extracts code from `<code>```python...```</code>` tags.

### Phase 2: Code Patching

The extracted code is patched in 3 steps:

1. **`_patch_imports()`**: `from workflow import ...` -> `from pettingllms.multi_agent_env.autoevol.workflow import ...`
2. **`_patch_string_escapes()`**: LaTeX strings -> raw strings
3. **`_patch_workflow_init()`**: `Workflow(name="math_solver")` -> `Workflow(name="math_solver", ai_client=ai_client)` + `ai_client.workflow = workflow`

Setup code (AIClient creation) is prepended, DataProto save code is appended.

### Phase 3: AgentNode Executes (Multi-Turn with Python Tool)

The workflow runs. The `AgentNode` ("MathSolver") starts the conversation:

---

**Turn 1: AgentNode calls LLM (via AIClient -> Designer's vLLM)**

```
messages = [
    {"role": "system", "content":
        "You are an expert mathematician. Solve problems using Python code.\n\n"
        "When you need to compute something, write Python code in a ```python block.\n"
        "The code will be executed and the output returned to you.\n"
        "After getting the result, state your final answer as: FINAL ANSWER: \\boxed{answer}\n\n"
        "After solving, restate your final answer clearly as: FINAL ANSWER: \\boxed{your_answer}"},
    {"role": "user", "content":
        "Let S be the set of positive integers k such that 2 divides kC(k,2). "
        "Find |S| for k in {1,2,...,30}."}
]
```

**LLM Response (Turn 1):**
```
I need to find all k in {1,...,30} where 2 divides k * C(k,2).

Let me compute this with Python:

```python
from math import comb

count = 0
results = []
for k in range(1, 31):
    val = k * comb(k, 2)
    if val % 2 == 0:
        count += 1
        results.append(k)

print(f"Values of k: {results}")
print(f"|S| = {count}")
```
```

**`_handle_tool_calls()` detects the `` ```python `` block:**
- Extracts the Python code
- Calls `MathEnvironment.execute(code)` which runs it via subprocess
- Gets result: `"Code executed successfully.\nOutput: Values of k: [1, 2, 3, 4, ...]\n|S| = 22"`

**Feedback injected as user message:**
```python
messages.append({"role": "assistant", "content": "<the LLM response above>"})
messages.append({"role": "user", "content":
    "Tool 'python_execute' returned: Code executed successfully.\n"
    "Output: Values of k: [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28]\n"
    "|S| = 22"})
```

> **DataProto #1 collected**: prompt tokens + response tokens from this LLM call are saved to `workflow.dataproto_list`.

---

**Turn 2: AgentNode calls LLM again (with execution result)**

**LLM Response (Turn 2):**
```
The Python code shows that there are 22 values of k in {1,...,30} where
2 divides k * C(k,2).

FINAL ANSWER: \boxed{22}
```

**No `` ```python `` block detected** -> loop exits. This is the final answer.

> **DataProto #2 collected**: prompt (now includes Turn 1 + tool result) + response tokens saved.

---

### Phase 4: Answer Extraction & Reward

```
Output text (captured from Ray worker stdout):
  "... Values of k: [1, 2, ...]\n|S| = 22\n...\nFINAL ANSWER: \boxed{22}"

_extract_final_answer() -> "22"    (extracted from \boxed{22})
ground_truth_answer    -> "22"

correctness_reward = 1.0   (exact match)
format_reward      = 0.1   (\boxed{} present, not a placeholder)
length_penalty     = 0.0   (responses < 3072 tokens)
code_block_penalty = 0.0   (no <code> or ```python in agent's FINAL output)

designer_reward = 1.0 + 0.1 = 1.1
agent_reward    = 1.0 + 0.1 + 0.0 + 0.0 = 1.1
```

### Phase 5: PPO Update

```
Designer PPO Trainer:
  - Input: DataProto from Designer's code generation response
  - Reward: designer_reward = 1.1
  - Update: Reinforce generating good workflow designs

AgentNode (Executor) PPO Trainer:
  - Input: DataProto #1 (Turn 1: wrote Python code) + DataProto #2 (Turn 2: final answer)
  - Reward: agent_reward = 1.1 (applied to last token position)
  - Update: Reinforce writing correct Python code and producing correct answers
```

---

## 3. Code Patching Pipeline

The `MASGenerator.step()` method (`gen_agent.py:115-365`) assembles the final `mas.py`:

### 3.1 Setup Code (Injected Before Designer's Code)

```python
import sys, os
from pettingllms.multi_agent_env.autoevol.utils.BaseOpenAI import AIClient

ai_client = AIClient(
    api_base="http://localhost:8000",     # Designer's vLLM server
    api_key="dummy",
    chat_model="Mercury7353/masrl0206_notool",
    tokenizer_path="/path/to/tokenizer",
    server_address="localhost:8000",
    max_prompt_length=2048,
    max_response_length=4096,
    enable_thinking=False,                 # AgentNodes don't use thinking
    workflow=None                          # Set after workflow creation
)
```

### 3.2 Three Patching Steps

| Patch | Method | What It Does |
|-------|--------|-------------|
| `_patch_imports()` | `gen_agent.py:367-412` | `from workflow import ...` -> `from pettingllms.multi_agent_env.autoevol.workflow import ...` |
| `_patch_string_escapes()` | `gen_agent.py:414-467` | LaTeX strings (`\frac`, `\sqrt`) -> raw strings to avoid unicode errors |
| `_patch_workflow_init()` | `gen_agent.py:469-510` | Injects `ai_client=ai_client` into `Workflow(...)` and adds `ai_client.workflow = workflow` |

### 3.3 DataProto Save Code (Appended)

```python
import pickle
if 'workflow' in globals() and hasattr(workflow, 'dataproto_list') and workflow.dataproto_list:
    with open('/path/to/dataproto.pkl', 'wb') as f:
        pickle.dump(workflow.dataproto_list, f)
    print("[SUCCESSSAVED]")
```

---

## 4. How the Python Tool Works (`` ```python `` Block Pattern)

### 4.1 The Key Difference: Designer vs AgentNode

| | Designer | AgentNode (Executor) |
|---|---------|---------------------|
| **Output format** | `<code>```python...```</code>` | `` ```python...``` `` directly in response |
| **What happens** | Code extracted by `MASGenerator.update_from_model()` | Code extracted by `_handle_tool_calls()`, executed via `MathEnvironment`, result fed back |
| **Purpose** | Generate workflow structure | Solve the math problem with code |

### 4.2 Python Tool Prompting (in AgentNode system prompt)

The Designer generates a system prompt for the AgentNode that tells it to use `` ```python `` blocks:

```
"You are an expert mathematician. Solve problems using Python code.

When you need to compute something, write Python code in a ```python block.
The code will be executed and the output returned to you.
After getting the result, state your final answer as: FINAL ANSWER: \boxed{answer}"
```

### 4.3 Code Extraction and Execution

Inside `_handle_tool_calls()`, when the LLM response contains a `` ```python `` block:

1. **Extract**: Parse the code between `` ```python\n `` and `` ``` ``
2. **Execute**: Call `MathEnvironment.execute(code)` which writes to temp file and runs via `subprocess.run()`
3. **Feedback**: Inject execution result as `{"role": "user", "content": "Tool 'python_execute' returned: ..."}`
4. **Loop**: LLM sees the result and either writes more code or produces final answer

```python
# MathEnvironment._fallback_execute() (math_env.py:64-108)
code_file = self._work_dir / "temp_code.py"
with open(code_file, "w") as f:
    f.write(code)
result = subprocess.run([sys.executable, str(code_file)],
                        capture_output=True, text=True, timeout=self.timeout)
if result.returncode == 0:
    return f"Code executed successfully.\nOutput: {result.stdout.strip()}"
else:
    return f"Error executing code (exit code {result.returncode}):\n{result.stderr.strip()}"
```

### 4.4 Multi-Turn Feedback Loop

```
Turn 1: LLM writes ```python code block
         |  -> code extracted
         |  -> executed via subprocess
         v
Turn 2: Execution result fed back as user message -> LLM sees output
         |  -> writes more ```python if needed, OR produces final answer
         v
Turn N: LLM outputs FINAL ANSWER: \boxed{...} (no ```python) -> loop exits
```

### 4.5 Error Handling

All errors are fed back as user messages, giving the LLM a chance to self-correct:

| Scenario | Feedback |
|----------|----------|
| Code runs successfully | `"Code executed successfully.\nOutput: 42"` |
| Python error (SyntaxError, etc.) | `"Error executing code (exit code 1):\nTraceback..."` |
| Timeout (>120s) | `"Error: Code execution timed out after 120 seconds"` |
| Max turns reached | `"Please provide your final answer now without using any tools."` |

### 4.6 DataProto Collection

Every `ai_client.chat()` call (each turn in the loop) creates a DataProto:
- Tokenized prompt + response
- Appended to `workflow.dataproto_list`
- Later collected for Executor PPO training

---

## 5. Shared Context Between Workflow Agents

When multiple `AgentNode`s are in a workflow, each subsequent agent sees:

```python
# agent_node.py:149-189
parts = [f"**Original Question:**\n{content}\n"]
parts.append("**Previous Agents' Analysis:**\n")
for resp in agent_responses:
    delivery = _extract_delivery(resp_content)  # Extract FINAL ANSWER: \boxed{...}
    parts.append(f"[{sender}]:\n{summary}\n")
parts.append("Based on the original question and the analysis above, provide your own solution.")
```

---

## 6. Reward System

| Component | Value | Condition |
|-----------|-------|-----------|
| Correctness | 0.0 or 1.0 | Math answer matches ground truth |
| Format bonus | +0.1 | Output contains `\boxed{real_answer}` |
| Length penalty | -0.1 | Any agent response > 3072 tokens |
| Code block penalty | -0.1 | Agent FINAL output contains `<code>` or `` ```python `` (intermediate tool turns are fine) |

```python
designer_reward = correctness_reward + format_reward
agent_reward    = correctness_reward + format_reward + length_penalty + code_block_penalty
```

### Correctness Check (`reward_function.py`)

1. Exact match (case-insensitive)
2. Decoration-stripped match
3. Normalized LaTeX match
4. LaTeX fraction to plain fraction
5. Numeric tolerance (rel_tol=0.005)
6. Multiple choice letter extraction
7. `math_verify` library

---

## 7. GRPO Grouping (Tree Design Mode)

```
Problem P
  |
  +-- Design D1 (Designer generates workflow code)
  |     |-- Run E1a (AgentNodes solve) -> reward r1a
  |     +-- Run E1b                    -> reward r1b
  |
  +-- Design D2
        +-- Run E2a -> reward r2a
```

- **Designer**: D1/D2 share one GRPO group. D1's reward = mean(r1a, r1b).
- **AgentNodes**: E1a/E1b share one group. E2a in its own group.

---

## 8. Key Files

| Component | File |
|-----------|------|
| Training script | `scripts/train/autoeval/train_L1_split_policy.sh` |
| Config | `pettingllms/config/autoevol/math_L1_split_policy.yaml` |
| Entry point | `pettingllms/trainer/train.py` |
| PPO trainer | `pettingllms/trainer/multi_agents_ppo_trainer.py` |
| Execution engine | `pettingllms/trainer/multi_agents_execution_engine_autoevol.py` |
| Designer + code exec | `pettingllms/multi_agent_env/autoevol/gen_agent.py` |
| Workflow runtime | `pettingllms/multi_agent_env/autoevol/workflow/workflow.py` |
| AgentNode (tool loop) | `pettingllms/multi_agent_env/autoevol/workflow/nodes/agent_node.py` |
| Python execution env | `pettingllms/multi_agent_env/autoevol/utils/environments/math_env.py` |
| AIClient (vLLM bridge) | `pettingllms/multi_agent_env/autoevol/utils/BaseOpenAI.py` |
| Reward function | `pettingllms/multi_agent_env/autoevol/reward_function.py` |
| Prompt templates (math) | `pettingllms/multi_agent_env/autoevol/utils/prompt.py` |
| Prompt templates (general) | `pettingllms/multi_agent_env/autoevol/utils/data_process.py` |

---

## 9. Architecture Diagram

```
+------------------------------- GPU 0-3 --------------------------------+
|  Designer Model (Mercury7353/masrl0206_notool)                         |
|  - vLLM server for design generation                                   |
|  - SAME vLLM server used by AgentNodes via AIClient                    |
|  - PPO Trainer #0 updates Designer trajectories                        |
+------------------------------------------------------------------------+

+------------------------------- GPU 4-7 --------------------------------+
|  Executor Policy PPO Trainer                                           |
|  - Trained on DataProtos from AgentNode LLM calls                      |
|  - The "executor" is NOT a separate model -- it's the AgentNodes       |
+------------------------------------------------------------------------+

Per training step:
  1. Load math problems
  2. For each problem (concurrent):
     a. Designer generates workflow code (enable_thinking=True)
     b. Code extracted from <code>```python...```</code>
     c. Code patched + executed in Ray worker
     d. AgentNodes run, each:
        - Calls Designer vLLM -> gets response
        - If response has ```python block -> extract, execute, feed result back
        - Repeat until final answer (\boxed{...})
        - DataProto saved per LLM call
     e. Rewards computed
  3. PPO update Designer (design quality)
  4. PPO update Executor policy (AgentNode solving quality)
```
