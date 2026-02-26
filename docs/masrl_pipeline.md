# MASRL Training Pipeline

Multi-Agent System Reinforcement Learning (MASRL) 训练流程文档。

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│  train.py (入口)                                                │
│  └─> MultiAgentsPPOTrainer.fit()  (训练主循环)                    │
│       │                                                         │
│       ├── [Rollout 阶段]                                        │
│       │   MultiAgentsExecutionEngineAutoEvol                     │
│       │   ├── generate_tree_rollout()     (N designs × M execs) │
│       │   └── generate_single_rollout()   (单次 design+exec)    │
│       │       │                                                 │
│       │       ├── MASGenerator (Designer)                       │
│       │       │   └── LLM 生成 workflow Python 代码              │
│       │       │                                                 │
│       │       └── MASGenerator.step() (执行 MAS)                │
│       │           ├── 生成 mas.py                               │
│       │           ├── Ray Worker 执行 mas.py                    │
│       │           │   ├── Workflow → AgentNode(s)               │
│       │           │   │   ├── AIClient._chat_verl() → vLLM     │
│       │           │   │   ├── ```python block → 执行 → 反馈     │
│       │           │   │   └── 多轮 tool calling loop            │
│       │           │   └── 输出 FINAL ANSWER: \boxed{...}        │
│       │           └── 计算 reward (correctness + format + penalties)│
│       │                                                         │
│       ├── [PPO 更新阶段]                                        │
│       │   reward → token_level_scores → GRPO advantages → loss  │
│       │                                                         │
│       └── [Validation / Checkpoint]                             │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 文件结构

| 文件 | 作用 |
|------|------|
| `pettingllms/trainer/train.py` | 入口，Hydra 配置加载，初始化 Ray |
| `pettingllms/trainer/multi_agents_ppo_trainer.py` | PPO 训练主循环，reward 构建，metrics |
| `pettingllms/trainer/multi_agents_execution_engine_autoevol.py` | Rollout 生成引擎（tree design / single rollout） |
| `pettingllms/multi_agent_env/autoevol/gen_agent.py` | `MASGenerator`（Designer + 执行），`MASExecutor`（split policy 模式） |
| `pettingllms/multi_agent_env/autoevol/workflow/workflow.py` | `Workflow` 类：顺序执行节点，收集 DataProto |
| `pettingllms/multi_agent_env/autoevol/workflow/nodes/agent_node.py` | `AgentNode`：LLM agent + tool calling loop |
| `pettingllms/multi_agent_env/autoevol/utils/BaseOpenAI.py` | `AIClient`：对接 vLLM server，生成 DataProto |
| `pettingllms/multi_agent_env/autoevol/utils/data_process.py` | Few-shot examples，system prompt 模板 |
| `pettingllms/multi_agent_env/autoevol/utils/environments/math_env.py` | Python 代码执行沙箱 |
| `pettingllms/config/autoevol/math_L1_prompt.yaml` | 训练配置（模型、agent、超参） |
| `scripts/train/autoeval/train_L1.sh` | 启动脚本 |

## 3. 训练主循环

**文件**: `pettingllms/trainer/multi_agents_ppo_trainer.py` — `MultiAgentsPPOTrainer.fit()`

```
for step in range(total_training_steps):
    1. init_agents_and_envs(mode="train")          # 初始化环境和 agent
    2. wakeup rollout engines                       # 启动 vLLM server
    3. generate_multiple_rollouts_concurrent()       # 并发生成 rollout
    4. reset_prefix_cache()                          # 清理 KV cache
    5. sleep rollout engines                         # 释放 GPU 显存
    6. 构建 reward tensor → token_level_scores       # reward 放到最后一个 valid token
    7. compute old_log_probs                         # 计算旧策略概率
    8. PPO update (actor gradient step)              # 梯度更新
    9. 每 val_freq 步做 validation
    10. 每 save_freq 步保存 checkpoint
```

**Reward Tensor 构建** (line 395-420):
```python
# reward 放在 response 的最后一个有效 token 位置
reward_tensor = torch.zeros_like(responses_batch)
last_valid_positions = valid_token_counts - 1
reward_tensor[valid_batch_indices, last_valid_positions] = rewards
batch.batch["token_level_scores"] = reward_tensor
```

## 4. Rollout 生成

**文件**: `pettingllms/trainer/multi_agents_execution_engine_autoevol.py`

### 4.1 调度逻辑 (line 1376)

```python
use_tree_design = (execute_sample_num > 1 and mode != "validate")

if use_tree_design:
    # 训练模式：N designs × M executions per problem
    generate_tree_rollout(env_idx)
else:
    # 验证模式或 M=1：每个 rollout 独立
    generate_single_rollout(rollout_idx)
```

### 4.2 Tree Design 模式 (line 1163)

每个 problem 生成 N 个 design，每个 design 执行 M 次：

```
Problem (env_idx=0):
  ├── Design 0 (rollout_idx 0-7)
  │   ├── Exec 0: MAS 执行 → reward_0
  │   ├── Exec 1: MAS 执行 → reward_1
  │   └── ...Exec 7: MAS 执行 → reward_7
  ├── Design 1 (rollout_idx 8-15)
  │   └── ...
  └── Design 3 (rollout_idx 24-31)
      └── ...
```

**Designer reward** = `mean(M次execution的designer_reward)` → 赋给 Designer DataProto (agent_idx=0)
**Executor reward** = 每次 execution 的 `reward` → 赋给 Workflow DataProto (agent_idx=1)

**GRPO 分组**：
- Designer: 同一 problem 的 N 个 design 为一组
- Executor (WorkflowAgent): 同一 design 的 M 次 execution 为一组

### 4.3 Single Rollout 模式 (line 303)

验证模式使用，流程：
1. Designer 生成 MAS 代码
2. 执行 MAS 代码
3. 计算 reward
4. 返回 DataProto

## 5. Designer（MAS 代码生成）

**文件**: `pettingllms/multi_agent_env/autoevol/gen_agent.py` — `MASGenerator`

### 5.1 Prompt 构建

Designer 接收数学题目，生成 Python workflow 代码：

```python
# update_from_env() 构建 prompt
prompt = {
    "system": "You are an expert in designing Multi-Agent System workflows...",
    "text": f"Design Multi Agent System for the Question: {problem}"
}
```

Few-shot examples 在 `utils/prompt.py` 的 `WORKFLOW_EXAMPLES` 中定义。

### 5.2 代码提取 (line 87)

从 LLM response 中提取代码：
```python
# 顺序尝试：
1. <code>```python ... ```</code>   # 标准格式
2. <code>``` ... ```</code>          # 无语言标记
3. <code>raw_content</code>          # 直接内容
```

### 5.3 MAS 执行 (`step()`, line 115)

```
step() 流程:
  1. 检查 current_action 非空
  2. 构建 setup_code (import, AIClient 配置)
  3. _patch_imports() — 修复 import 路径
  4. _patch_string_escapes() — 处理转义字符
  5. _patch_workflow_init() — 注入 vLLM server 地址
  6. 拼接: setup_code + patched_code + dataproto_save_code
  7. 保存为 mas.py
  8. Ray Worker 执行 mas.py
  9. 解析 output_text，提取 final_answer
  10. 计算 reward
  11. 加载 dataproto.pkl (workflow DataProtos)
```

## 6. Workflow 执行（MAS 内部）

**文件**: `pettingllms/multi_agent_env/autoevol/workflow/workflow.py`

Designer 生成的 `mas.py` 代码实例化 `Workflow`，包含若干 `AgentNode`。

```python
# mas.py 中生成的典型代码：
workflow = Workflow(name="math_solver")
workflow.add_node(AgentNode(
    name="MathSolver",
    system_prompt="You are a math expert...",
    tool_registry=ToolRegistry([python_execute])
))
result = workflow.run(problem)
```

### 6.1 AgentNode 处理流程

**文件**: `pettingllms/multi_agent_env/autoevol/workflow/nodes/agent_node.py`

```
AgentNode.process():
  1. _build_initial_messages()
     - system prompt + DELIVERY_INSTRUCTION ("\nFINAL ANSWER: \boxed{...}")
     - 共享上下文：原始题目 + 之前 agent 的 delivery
  2. _handle_tool_calls() — 多轮 tool calling loop (最多 max_turns=10 轮)
     │
     ├── LLM 生成 response
     ├── 检测 ```python block → 提取代码 → python_execute() → 结果反馈
     ├── 检测 <tool_call> JSON → 解析 → call_tool() → 结果反馈
     └── 无 tool call → 返回 final response
  3. 记录对话到 ShareGPT 格式
  4. 返回 Message(content=response, type=AGENT_RESPONSE)
```

### 6.2 Python Tool 执行

**Agent prompt 指示** (在 `data_process.py` 的 few-shot examples 中)：
```
Write your code in a ```python block. It will be automatically executed
and the result will be returned to you.
```

**执行流程** (`agent_node.py` line 256-280):
```python
python_code = _extract_python_code_block(response)  # regex: ```python\s*\n(.*?)```
if python_code:
    tool_result = tool_registry.call_tool("python_execute", {"code": python_code})
    messages.append({"role": "user", "content": tool_result})
    continue  # 进入下一轮
```

**沙箱执行** (`math_env.py`):
```python
# MathEnvironment.execute(): 写入临时文件 → subprocess.run(timeout=30s) → 返回 stdout/stderr
```

### 6.3 DataProto 收集

**文件**: `pettingllms/multi_agent_env/autoevol/utils/BaseOpenAI.py` — `AIClient._chat_verl()`

每次 AgentNode 调用 LLM 时：
1. `convert_prompt_to_dpr()` — messages → DataProto (prompt tokens)
2. `llm_async_generate()` — vLLM server 生成 response → DataProto (response tokens)
3. 结果 append 到 `workflow.dataproto_list`

这些 DataProto 最终被 `dataproto_save_code` 保存为 `dataproto.pkl`，由 execution engine 加载并赋予 reward。

## 7. Reward 计算

**文件**: `pettingllms/multi_agent_env/autoevol/gen_agent.py` (line 311-346)

```
reward = correctness + format_bonus + length_penalty + code_block_penalty

correctness:       0 or 1     — final_answer vs ground_truth（数学等价性判断）
format_bonus:      +0.1       — 使用了 \boxed{} 且非 placeholder
length_penalty:    -0.1       — 任一 agent response > 3072 tokens
code_block_penalty: -0.1      — agent 输出包含 <code> 或 ```python（不应在 final output 中出现）
```

**Designer vs Executor reward 区分**:
- `designer_reward = correctness + format_bonus` (无 penalty，仅衡量 design 质量)
- `executor_reward = correctness + format_bonus + penalties` (完整 reward，包含执行质量)

## 8. DataProto 组装与 GRPO 分组

**文件**: `pettingllms/trainer/multi_agents_execution_engine_autoevol.py` (line 1238-1330)

每个 rollout 产生三类 DataProto：

| 类型 | agent_idx | agent_name | reward | 来源 |
|------|-----------|------------|--------|------|
| Designer DataProto | 0 | "Designer" | `designer_mean_reward` (M次平均) | Designer LLM 调用 |
| Executor DataProto | 1 | executor_name | `exec_reward` | Executor LLM 调用 (split policy only) |
| Workflow DataProto | 1 | "WorkflowAgent" | `exec_reward` | MAS 内部 AgentNode LLM 调用 |

所有 DataProto 按 `policy_name` 归入对应 PPO trainer 的 batch。

## 9. WandB Metrics

**文件**: `pettingllms/trainer/multi_agents_ppo_trainer.py` (line 862-945)

| Metric | 含义 |
|--------|------|
| `shared_model_critic/score/mean` | 整个 batch 的平均 reward（Designer + WorkflowAgent） |
| `shared_model/reward_by_agent/Designer/mean` | Designer DataProto 的平均 reward |
| `shared_model/reward_by_agent/WorkflowAgent/mean` | Workflow DataProto 的平均 reward |
| `shared_model/tree_design/designer_reward_mean` | agent_idx=0 的平均 reward |
| `shared_model/tree_design/executor_reward_mean` | agent_idx=1 的平均 reward |
| `shared_model/tree_design/inter_design_reward_mean` | 按 (env_idx, design_idx) 分组的 executor reward 均值 |
| `shared_model/tree_design/intra_design_reward_std` | 同一 design 内 M 次 execution reward 的 std |

## 10. 配置说明

**文件**: `pettingllms/config/autoevol/math_L1_prompt.yaml`

```yaml
# 关键配置项
specialization: prompt           # prompt=共享模型, lora=LoRA区分, full=完全分离
turn_order: [Designer]           # shared model: 只有 Designer
                                 # split policy: [Designer, Executor]

training:
  train_batch_size: 32           # 每步处理的 problem 数
  train_sample_num: 16           # 每个 problem 的 rollout 数 (auto: N×M)
  design_sample_num: 4           # tree design: N 个 design
  execute_sample_num: 8          # tree design: 每个 design M 次执行
  max_prompt_length: 4096
  max_response_length: 8192

base_models:
  policy_0:
    name: shared_model           # WandB metric 前缀
```

## 11. Rollout 日志

**目录**: `tmp_auto_mas/{experiment_name}/{train|validate}/rollout_{idx}/`

每个 rollout 目录包含：
- `response.txt` — Designer 的 LLM response（包含 `<code>` 标签）
- `mas.py` — 生成的 MAS 代码（经过 patch 的完整可执行文件）
- `output.txt` — mas.py 的执行输出（包含 agent 日志和 FINAL ANSWER）
- `dataproto.pkl` — Workflow 内所有 AgentNode 的 DataProto 列表
- `executor_response.txt` — Executor 的 LLM response（split policy 模式）

## 12. 已修复的 Bug

### Bug 1: Tree Design exec>0 全部失败
**文件**: `multi_agents_execution_engine_autoevol.py` line 1083-1088
**根因**: Shared model 模式下 `_execute_single_design()` 对 exec=1-7 使用了不同的 agent 实例，但 `has_executor=False` 导致 `update_from_model()` 被跳过，agent 的 `current_action` 为空，`step()` 直接返回 `reward=0`。
**修复**: 在 `else` 分支也调用 `executor_agent.update_from_model(design_result["designer_response"])`。

### Bug 2: Workflow DataProto agent_name 错误
**文件**: `multi_agents_execution_engine_autoevol.py` line 744, 1308
**根因**: Shared model 模式下 `executor_name` 回退为 `designer_name="Designer"`，导致 Workflow DataProtos 的 `agent_name="Designer"` 但 `agent_idx=1`，metrics 统计不一致。
**修复**: Workflow DataProtos 使用 `agent_name="WorkflowAgent"` 来区分。

### Bug 3: Checkpoint 命名错误
**文件**: `multi_agents_ppo_trainer.py` line 125
**根因**: `experiment_name` 赋值在 `if lora_rank > 0` 分支内，非 LoRA 模式使用默认值 `"gsm8k"`。
**修复**: 移到 if/else 外面，始终生效。

### Bug 4: Auto Resume 无法加载 checkpoint
**文件**: `verl/verl/utils/checkpoint/checkpoint_manager.py` — `is_valid_checkpoint()`
**根因**: `_save_checkpoint()`（`pettingllms/verl/ray_trainer.py:841-843`）中 `data.pt` 的保存被注释掉（MASRL 不使用 PyTorch DataLoader，而是通过 `step_idx * batch_size` 直接计算数据偏移），但 `is_valid_checkpoint()` 仍然硬性要求 `data.pt` 存在。导致 `find_latest_ckpt_path()` 判定 checkpoint 无效并跳过，训练从头开始。
**修复**: 将 `data.pt` 从必需文件改为可选文件，缺失时只打印 warning 而不返回 `False`。

### Bug 5: Rollout 日志目录混乱
**文件**: `multi_agents_execution_engine_autoevol.py` line 596, 1098
**根因**: 所有实验的 rollout 日志都平铺存放在 `tmp_auto_mas/rollout_{idx}/`，不同实验、训练/验证阶段的日志互相覆盖。
**修复**: 日志路径改为 `tmp_auto_mas/{experiment_name}/{train|validate}/rollout_{idx}/`，按实验名和模式分目录存放。
