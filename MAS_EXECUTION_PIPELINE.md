# MAS Execution Pipeline and Reward Calculation

## Overview
This document explains how the Multi-Agent System (MAS) execution pipeline works in `multi_agents_execution_engine_autoevol.py`, specifically:
1. How the Designer generates MAS code
2. How the MAS is executed
3. How the final answer is extracted
4. How rewards are calculated based on ground truth

## Pipeline Flow

### Phase 1: Designer Generates MAS Code (Lines 316-437)

```python
# Step 1: Designer gets prompt from environment
designer_agent.update_from_env(env)
designer_prompt = designer_agent.current_prompt

# Step 2: Format prompt and generate MAS code using LLM
designer_output_dpr, designer_response = await llm_async_generate(...)

# Step 3: Extract generated code
designer_agent.update_from_model(designer_response)
designer_code = designer_agent.generated_code
```

**Key Points:**
- Designer receives the problem/question from the environment
- Designer uses LLM to generate Python code that defines a Multi-Agent System
- The generated code typically creates a `Workflow` with multiple agents (sub-agents)
- The Designer itself acts as one of the sub-agents in the MAS

### Phase 2: Executor Executes MAS Code (Lines 439-636)

#### Step 1: Prepare MAS Code for Execution (Lines 584-600 in `gen_agent.py`)

The generated code is:
1. **Patched with imports** (`_patch_imports`): Fixes import paths
2. **Patched with verl integration** (`_patch_workflow_init`): Adds AIClient setup for verl integration
3. **Wrapped with setup code**: Adds necessary imports and AIClient initialization
4. **Saved to `mas.py`**: The complete code is written to `output_dir/mas.py`

**Setup Code Added (Lines 134-176):**
```python
# Creates AIClient with verl integration
ai_client = AIClient(
    api_base=_api_base,
    api_key="dummy",
    chat_model=model_name,
    max_answer_tokens=max_response_length,
    tokenizer_path=tokenizer_path,
    server_address=server_address,
    max_prompt_length=max_prompt_length,
    max_response_length=max_response_length,
    enable_thinking=enable_thinking,
    workflow=None
)
```

**DataProto Saving Code Added (Lines 188-207):**
```python
# At the end of mas.py execution, saves workflow DataProto
if 'workflow' in globals() and hasattr(workflow, 'dataproto_list'):
    with open(dataproto_file, 'wb') as f:
        pickle.dump(workflow.dataproto_list, f)
    print("[SUCCESSSAVED]")
```

#### Step 2: Execute MAS Code (Lines 224-250 in `gen_agent.py`)

```python
# Execute mas.py using Ray worker
obj_ref = env_worker.run.remote(script_content, step_timeout)
output_text = await _await_ray_object_ref(obj_ref, total_timeout)

# Save output to output.txt
with open(output_txt_path, 'w') as f:
    f.write(output_text)

# Load DataProto if execution succeeded
if os.path.exists(dataproto_pkl_path):
    with open(dataproto_pkl_path, 'rb') as f:
        workflow_dataproto_list = pickle.load(f)
```

**Key Points:**
- MAS code is executed in a Ray worker (isolated environment)
- Execution output is captured in `output_text`
- If execution succeeds, `[SUCCESSSAVED]` appears in output
- Workflow DataProto list is saved to `dataproto.pkl` and loaded back

#### Step 3: Extract Final Answer (Lines 251-252, 417-494 in `gen_agent.py`)

The `_extract_final_answer` method extracts the final answer from MAS execution output:

**Strategy 1: Look for Final Answer markers**
```python
# Pattern 1: **Final Answer** or **Final Answer**: xxx
final_answer_match = re.search(
    r'\*\*Final\s*Answer\s*\*{0,2}[:\s]*(.*?)(?:={3,}|\Z)',
    output_text, re.DOTALL | re.IGNORECASE
)

# Pattern 2: FINAL ANSWER: xxx or FINAL ANSWER:\nxxx
if not final_answer_match:
    final_answer_match = re.search(
        r'(?:^|\n)\s*FINAL\s+ANSWER\s*:?\s*\n?\s*(.*?)(?:\n\n|\n===|\n\s*\n|$|\Z)',
        output_text, re.MULTILINE | re.IGNORECASE | re.DOTALL
    )

# Extract last number from final answer section
number_pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
numbers = re.findall(number_pattern, final_answer_section)
if numbers:
    return numbers[-1]
```

**Strategy 2: Look for WORKFLOW_SUMMARY markers**
```python
workflow_match = re.search(
    r'WORKFLOW_SUMMARY_START\s*(.*?)\s*WORKFLOW_SUMMARY_END',
    output_text, re.DOTALL | re.IGNORECASE
)
# Extract last number from summary
```

**Strategy 3: Extract last number from entire output**
```python
number_pattern = r'-?\d+(?:\.\d+)?(?:/\d+)?'
numbers = re.findall(number_pattern, output_text)
if numbers:
    return numbers[-1]
```

### Phase 3: Calculate Reward (Lines 254-256, 496-521 in `gen_agent.py`)

#### Reward Calculation Flow

```python
# Step 1: Extract final answer from MAS output
final_answer = self._extract_final_answer(output_text)

# Step 2: Calculate reward based on task type
reward = self._calculate_reward(final_answer, env_data)
```

#### Reward Function Selection (Lines 496-521)

```python
def _calculate_reward(self, final_answer: str, env_data: Env) -> float:
    # Get reward function based on task_type
    reward_function = REWARD_FUNCTIONS.get(self.task_type.lower())
    
    if reward_function is None:
        return 0.0
    
    # Call reward function with final_answer and env_data
    reward = reward_function(final_answer, env_data)
    return float(reward)
```

#### Math Reward Function (Lines 63-91 in `reward_function.py`)

For math tasks, the reward is calculated by:

```python
def math_reward_function(summary: str, env_data: Env) -> float:
    # Extract predicted answer from summary
    predicted_answer = extract_answer_from_summary(summary)
    
    # Get ground truth from env_data.state
    ground_truth = env_data.state.ground_truth_answer
    
    # Parse both answers using math_verify
    parsed_gt = parse(str(ground_truth))
    
    # Verify if they match
    is_correct = verify(predicted_answer, parsed_gt)
    
    # Return 1.0 if correct, 0.0 if incorrect
    reward = 1.0 if is_correct else 0.0
    return reward
```

**Key Points:**
- Uses `math_verify.parse()` and `math_verify.verify()` for mathematical equivalence
- Handles fractions, decimals, and different representations (e.g., "1/2" == "0.5")
- Returns binary reward: 1.0 (correct) or 0.0 (incorrect)

#### QA Reward Function (Lines 115-143 in `reward_function.py`)

```python
def qa_reward_function(summary: str, env_data: Env) -> float:
    # Extract predicted answer
    predicted_answer = extract_answer_from_summary(summary)
    
    # Get ground truth
    ground_truth = getattr(env_data, 'answer', None)
    
    # Simple exact match (case-insensitive)
    pred_normalized = predicted_answer.strip().lower()
    gt_normalized = str(ground_truth).strip().lower()
    
    reward = 1.0 if pred_normalized == gt_normalized else 0.0
    return reward
```

### Phase 4: Assign Rewards to DataProtos (Lines 645-719)

After MAS execution, rewards are assigned to all DataProtos:

```python
# Designer's DataProto - reward based on final outcome
if designer_output_dpr is not None:
    designer_batch_size = len(designer_output_dpr)
    designer_output_dpr.non_tensor_batch["reward"] = np.array([final_reward] * designer_batch_size)
    designer_output_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward] * designer_batch_size)
    # ... other metadata

# Executor's DataProto - reward based on final outcome
if executor_output_dpr is not None and has_executor:
    executor_batch_size = len(executor_output_dpr)
    executor_output_dpr.non_tensor_batch["reward"] = np.array([final_reward] * executor_batch_size)
    executor_output_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward] * executor_batch_size)
    # ... other metadata

# Workflow DataProtos from MAS workflow execution
if workflow_dataproto_list:
    for workflow_dpr in workflow_dataproto_list:
        batch_size = len(workflow_dpr)
        workflow_dpr.non_tensor_batch["reward"] = np.array([final_reward] * batch_size)
        workflow_dpr.non_tensor_batch["env_final_reward"] = np.array([final_reward] * batch_size)
        # ... other metadata
```

**Key Points:**
- **All agents receive the same reward**: Designer, Executor, and all sub-agents in the MAS workflow
- **Reward is based on final outcome**: The reward is calculated from the final answer extracted from MAS execution
- **Reward is binary**: 1.0 if correct, 0.0 if incorrect (for math/QA tasks)

## Summary

### Complete Pipeline:

1. **Designer generates MAS code** → LLM generates Python code defining a Multi-Agent System
2. **Code is patched and saved** → Code is wrapped with verl integration and saved to `mas.py`
3. **MAS is executed** → `mas.py` is executed in Ray worker, creating and running the workflow
4. **Sub-agents interact** → The Designer (and other sub-agents) act within the MAS workflow
5. **Output is captured** → MAS execution output is saved to `output.txt`
6. **Final answer is extracted** → Answer is extracted from output using regex patterns
7. **Reward is calculated** → Final answer is compared with ground truth using task-specific reward function
8. **Rewards are assigned** → All DataProtos (Designer, Executor, workflow sub-agents) receive the same reward

### Reward Calculation Verification:

✅ **Answer Extraction**: Multiple strategies to extract final answer from MAS output
✅ **Ground Truth Access**: `env_data.state.ground_truth_answer` contains the correct answer
✅ **Task-Specific Verification**: Math tasks use `math_verify` for mathematical equivalence
✅ **Binary Rewards**: Clear 1.0 (correct) / 0.0 (incorrect) reward structure
✅ **Reward Propagation**: All agents (Designer, Executor, sub-agents) receive the same outcome reward

### Code Locations:

- **MAS Execution**: `gen_agent.py` lines 80-271 (`MASGenerator.step()` or `MASExecutor.step()`)
- **Answer Extraction**: `gen_agent.py` lines 417-494 (`_extract_final_answer()`)
- **Reward Calculation**: `gen_agent.py` lines 496-521 (`_calculate_reward()`)
- **Reward Functions**: `reward_function.py` (math, qa, code reward functions)
- **Reward Assignment**: `multi_agents_execution_engine_autoevol.py` lines 645-719

