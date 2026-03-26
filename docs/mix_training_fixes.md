# Mixed Training (Math + Code) Fix Summary

## Problem

Mixed-environment training (math + code) consistently crashed while math-only training worked fine. This doc explains the root causes and fixes.

## Root Cause: FA2 PTX Incompatibility → Sleep Failure Cascade

The crash chain:

1. **FA2 PTX error kills vLLM engine** — `VLLM_FLASH_ATTN_VERSION=2` triggered `RuntimeError: CUDA error: the provided PTX was compiled with an unsupported toolchain` during rollout inference.

2. **Engine death breaks sleep()** — After rollout, the system calls `sleep()` on vLLM to free KV cache GPU memory (~61 GiB) before PPO training. With a dead engine, `reset_prefix_cache` and `sleep()` both fail.

3. **KV cache memory NOT freed → OOM during training** — PPO `update_actor` runs on the same GPUs (colocated WorkerDict). Without KV cache freed, there's only ~588 MiB free out of 79 GiB. The first `torch.logsumexp(logits, dim=-1)` in entropy computation tries to allocate ~7.5 GiB and OOMs.

### Why math-only didn't crash

Math tasks are simpler — shorter sequences, fewer concurrent LLM calls in MAS execution, less pressure on the vLLM engine. The FA2 PTX error was triggered by specific memory/compute patterns that code tasks hit more frequently due to longer contexts (8192 tokens) and more workflow agent turns.

## Secondary Bug: DataProto Alignment

`_align_non_tensor_batch_keys` used a single pass to pad missing keys across DataProtos. Code tasks generate variable numbers of workflow DataProtos (2-4× more than math), so later DataProtos could introduce keys that earlier ones didn't have. A single forward pass missed these.

## Fixes Applied

### 1. Removed `VLLM_FLASH_ATTN_VERSION=2`

**File:** `train_design_tree_mix.sh`

Letting vLLM use its default Flash Attention backend eliminates the PTX compilation error entirely. The engine stays alive through rollout, `sleep()` succeeds, KV cache is freed, and training has enough GPU memory.

### 2. Two-pass DataProto alignment

**File:** `multi_agents_execution_engine_autoevol.py` (line ~1391)

```python
# Pass 1: forward alignment
for i in range(1, len(dpr_list)):
    _align_non_tensor_batch_keys(dpr_list[0], dpr_list[i])
# Pass 2: catch keys introduced by later DataProtos
for i in range(1, len(dpr_list)):
    _align_non_tensor_batch_keys(dpr_list[0], dpr_list[i])
```

This ensures all DataProtos have consistent keys regardless of which workflow generated which metadata fields.

## Remaining Considerations

- **Variable batch sizes**: Code tasks generate 240-310 samples per rollout vs math's 40-110. Combined batches range from 424-768 samples. This creates variable memory pressure during training.
- **Engine death is still possible**: Other CUDA errors (illegal memory access, TMA descriptor errors) can still kill the engine. If `sleep()` fails for any reason, the same OOM cascade will occur.
- **`gpu_memory_utilization=0.8`** is required to serve the model. Cannot reduce to free more headroom.

## Current Config (Working)

```bash
DESIGN_SAMPLE_NUM=2
EXECUTE_SAMPLE_NUM=2
TRAIN_BATCH_SIZE=8
gpu_memory_utilization=0.8
VLLM_USE_V1=1
VLLM_CUDAGRAPH_MODE=piecewise
MAX_ROLLOUT_CONCURRENCY=64
MAX_ROLLOUT_RETRIES=3
# No VLLM_FLASH_ATTN_VERSION set (use default)
```
