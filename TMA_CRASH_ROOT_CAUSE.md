# TMA Crash & EngineDeadError: Root Cause Analysis

**Date:** 2026-03-28
**Codebase:** PettingLLMs / verl
**Affected file:** `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
**Pre-fix commit:** `25c6e87`
**Fix commit:** `d61a981`

---

## 1. Observed Symptoms

During PPO training on H100 GPUs (vLLM v1, MASRL autoevol), the following crash sequence occurs every ~10 training steps:

1. `CUDA error: illegal memory access` during `compute_log_prob` (FSDP training phase)
2. `EngineDeadError` raised on next vLLM engine interaction
3. `wake_up()` attempts to reinitialize the engine → fails with `Engine core initialization failed`
4. Training process dies

---

## 2. Short Answer: Is "Timeout Too Late" the Root Cause?

**No — it is a necessary condition, but not the root cause.**

The actual root cause is a **fatal fallback in `sleep()`**: the old code explicitly proceeded to `engine.sleep()` even when `reset_prefix_cache()` had failed, making the TMA crash guaranteed regardless of how long it waited.

The short drain timeout (10 s) is a contributing trigger, but even with a longer timeout, the same crash would happen any time `reset_prefix_cache()` fails for any reason.

---

## 3. Full Causal Chain

```
MAS subprocess timeout
  → SIGKILL kills subprocess
  → TCP connection to vLLM HTTP server closes abruptly
  → vLLM non-streaming endpoint does NOT detect client disconnect
  → Orphan request continues generating in EngineCore (up to max_tokens=8192)
  → Rollout phase ends; sleep() is called
  → abort() sent, but orphan is still in output_processor
  → Old code waited only 10 s for drain (orphan takes minutes)
  → Drain check shows remaining > 0
  → reset_prefix_cache() fails: "blocks not freed"
  → [FATAL BUG] Old code logs "Proceeding with sleep anyway" → calls engine.sleep()
  → engine.sleep() frees physical GPU memory pages
  → EngineCore still holds KV block pointers to now-freed pages
  → TMA descriptors reference those freed addresses
  → FSDP forward pass accesses freed memory
  → CUDA illegal memory access → crash
  → EngineDeadError on next engine call
  → wake_up() tries init_engine() on already-running Ray workers → fails
```

---

## 4. Evidence

### 4.1 Orphan Request Creation — `math_worker.py`

When a MAS subprocess times out, `_worker_docker` sends `SIGKILL` to the entire process group:

```python
# math_worker.py:185-188
except asyncio.TimeoutError:
    if proc and proc.pid:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    result = "timeout"
```

`SIGKILL` closes the TCP connection to vLLM's HTTP server. vLLM's non-streaming completion endpoint does **not** call `abort()` on client disconnect — it continues generating until `max_tokens` is reached. This is the source of orphan requests.

The timeout is triggered via `_await_ray_object_ref` in `gen_agent.py`:

```python
# gen_agent.py:298-302
obj_ref = env_worker.run.remote(script_content, step_timeout)
try:
    output_text = await _await_ray_object_ref(obj_ref, total_timeout)
finally:
    del obj_ref  # Release Ray object reference — does NOT cancel the Ray actor
```

**Note:** `del obj_ref` releases the Python object reference but does **not** cancel the running Ray actor. The actor continues executing inside its own asyncio loop, and any in-flight vLLM HTTP requests it made are now orphaned — their TCP connection is dead but vLLM EngineCore continues generating.

### 4.2 The Fatal Bug — `sleep()` in `vllm_async_server.py` (pre-fix, commit `25c6e87`)

The old `sleep()` had a 10 s drain window and 10 × 2 s = 20 s of `reset_prefix_cache` retries:

```python
# OLD CODE — commit 25c6e87
max_wait, poll_interval, elapsed = 10.0, 0.2, 0.0   # ← 10 s drain
while elapsed < max_wait:
    await asyncio.sleep(poll_interval)
    elapsed += poll_interval
    if not self.engine.output_processor.has_unfinished_requests():
        break
remaining = self.engine.output_processor.get_num_unfinished_requests()
if remaining > 0:
    logger.warning(f"... {remaining} requests remaining after {max_wait}s wait.")
    # ← No guard: continues to reset_prefix_cache even if drain failed
```

Then `reset_prefix_cache` was retried 10 times with 2 s each:

```python
# OLD CODE — commit 25c6e87
for attempt in range(10):                              # ← only 10 × 2s = 20s
    try:
        await self.engine.reset_prefix_cache()
        cache_reset_ok = True
        break
    except Exception as e:
        if attempt < 9:
            logger.warning(f"... Retrying in 2s...")
            await asyncio.sleep(2)
        else:
            logger.warning(f"... Proceeding with sleep anyway.")  # ← FATAL
            # No raise, no guard — falls through to engine.sleep()
```

**This is the root cause.** After 20 s of failed `reset_prefix_cache`, the code said "Proceeding with sleep anyway" and then:

```python
# OLD CODE — commit 25c6e87
try:
    await self.engine.sleep()           # ← frees GPU memory with live KV blocks
    self._engine_dead = False
except Exception as e:
    ...
    self._engine_dead = True
```

At this point:
- `engine.sleep()` frees physical GPU pages via `cudaFree`
- EngineCore still holds `CUdeviceptr` values pointing to those freed pages
- H100 TMA (Tensor Memory Accelerator) descriptors cache these physical addresses
- Next FSDP training step dereferences those addresses → **CUDA illegal memory access**

### 4.3 The Broken Reinit Path — `wake_up()` (pre-fix, commit `25c6e87`)

When the TMA crash corrupted engine state, `EngineDeadError` was raised. `wake_up()` then tried to reinitialize:

```python
# OLD CODE — commit 25c6e87
async def wake_up(self, tags: Optional[list[str]] = None):
    if getattr(self, '_engine_dead', False):
        logger.warning("[AsyncvLLMServer] Engine was dead, reinitializing...")
        ...
        await self.init_engine()   # ← fails
        self._engine_dead = False
    await self.engine.wake_up(tags)
```

`init_engine()` calls `init_worker()`, `init_device()`, and `load_model()` on the `ExternalRayDistributedExecutor` workers. Those workers are already running (they were never shut down) and cannot be re-initialized — they raise `Engine core initialization failed`. Training dies.

---

## 5. Why 10 s Was Not Enough (Secondary Issue)

`abort()` in vLLM v1 sends a ZMQ message to the EngineCore subprocess. EngineCore processes it on its **next scheduler iteration** — typically after the current CUDA forward pass completes. A single forward pass for 1 token takes ~20–100 ms on 8× H100. Once the abort message is received, the request is removed and KV blocks freed within 1 step.

However, the orphan request is **still active in vLLM's output_processor** (`output_processor.has_unfinished_requests()` returns True) until EngineCore confirms the abort back over ZMQ. If the request has already finished generating in EngineCore but hasn't been "collected" from the output queue yet, the abort processing can be delayed.

In practice, orphans generating up to `max_response_length=8192` tokens at ~30 ms/token can run for ~4 minutes. The 10 s abort window is far too short for these cases.

But this is a **secondary issue**: even if the drain window were infinite, the "Proceeding with sleep anyway" fallback would cause a TMA crash for any failure mode where `reset_prefix_cache` doesn't succeed within its retries.

---

## 6. The Fix (commit `d61a981`)

Three changes were made:

### Fix 1: `reset_prefix_cache` failure is now a hard error

```python
# NEW CODE — commit d61a981
if not cache_reset_ok:
    raise RuntimeError(
        "[AsyncvLLMServer] reset_prefix_cache failed after 300s of retries. "
        "Cannot safely call engine.sleep() — KV blocks still reference GPU memory. "
        "Investigate stuck EngineCore request."
    )
# engine.sleep() only reached when cache_reset_ok = True
await self.engine.sleep()
```

This eliminates the "Proceeding with sleep anyway" path. If KV blocks cannot be freed, `sleep()` is never called and there is no TMA crash.

### Fix 2: Extended drain window from 10 s → 300 s

```python
# NEW CODE — commit d61a981
max_wait, poll_interval, elapsed = 300.0, 1.0, 0.0  # was 10.0, 0.2
```

300 s matches the generation request timeout in `async_generate.py`. An orphan generating 8192 tokens at ~30 ms/token takes at most ~250 s, so 300 s is sufficient to drain it naturally after `abort()`.

### Fix 3: Extended `reset_prefix_cache` retries from 10×2 s → 60×5 s = 300 s

```python
# NEW CODE — commit d61a981
for attempt in range(60):          # was 10
    ...
    await asyncio.sleep(5)         # was 2s
```

### Fix 4: Removed broken `init_engine()` reinit path from `wake_up()`

```python
# NEW CODE — commit d61a981
async def wake_up(self, tags: Optional[list[str]] = None):
    await self.engine.wake_up(tags)
    self._is_sleeping = False
```

`ExternalRayDistributedExecutor` workers cannot be re-initialized once running. The old reinit path always failed. Removing it allows `engine.wake_up()` (which the vLLM team designed to handle resume after sleep) to work correctly.

### Fix 5: TOCTOU guard for new HTTP requests

```python
# NEW CODE — commit d61a981
self._is_sleeping = True  # set immediately at top of sleep()

# In completions():
if self._is_sleeping:
    return JSONResponse(ErrorResponse(...).model_dump(), status_code=503)
```

Prevents new requests sneaking in between the drain check and `engine.sleep()`.

---

## 7. What Was NOT Fixed (Known Limitation)

The underlying cause — vLLM's non-streaming HTTP endpoint not aborting on client disconnect — was intentionally left unfixed. The 300 s drain is a workaround: it waits for the orphan to finish generating naturally (or until abort() propagates through EngineCore) before proceeding to sleep.

A proper fix would require patching vLLM's HTTP layer to detect client TCP disconnect and call `engine.abort()` immediately. This was judged too invasive.

---

## 8. Summary Table

| Issue | Old Behavior | New Behavior |
|---|---|---|
| Drain wait | 10 s | 300 s |
| `reset_prefix_cache` retries | 10 × 2 s = 20 s | 60 × 5 s = 300 s |
| `reset_prefix_cache` failure | Logs warning, calls `sleep()` anyway → **TMA crash** | Raises `RuntimeError`, `sleep()` never called |
| `wake_up()` on dead engine | `init_engine()` → always fails | Removed; `engine.wake_up()` handles resume |
| New requests during sleep | No guard (TOCTOU race) | 503 response while sleeping |
