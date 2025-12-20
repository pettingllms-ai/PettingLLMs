# Agent Functions

This page reflects the current base contracts in `pettingllms/multi_agent_env/base` and shows how agents combine local and team rewards in `calculate_reward`, using the **code** and **math** environments as concrete examples.

## Base Interfaces (pettingllms/multi_agent_env/base/agent.py)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pettingllms.multi_agent_env.base.env import Env

@dataclass
class AgentData:
    current_prompt: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {"text": None, "image": None}
    )
    current_action: Optional[Any] = None
    agent_reward: Optional[float] = 0.0
    success: bool = False
    done: bool = False
    skip_current_turn: bool = False


class Agent(AgentData):
    @abstractmethod
    def update_from_env(self, env_data: Env, **kwargs) -> Env: ...

    @abstractmethod
    def update_from_model(self, env_data: Env, **kwargs) -> Env: ...

    @abstractmethod
    def reset(self): ...
```

**Key Changes from Previous Version:**
- ✅ **Removed**: `answer_history`, `action_history`, `reward_history` - These are no longer maintained by agents. Reward tracking is now handled externally by the execution engine.
- ✅ **Added**: `skip_current_turn` - Allows agents to skip turns when needed

- `update_from_env` and `update_from_model` accept `**kwargs` so implementations can take extra parameters (e.g., `turn_idx` or a raw `response` string) while still aligning with the base signature.
- `reset` in the base class clears prompts/actions and success flags; derived agents typically call `super().reset()` then add any custom cleanup.

## Core Lifecycle

1) **update_from_env(env_data, \*\*kwargs)**  
   Read `env_data.state` and build the prompt or observation for the model. Some agents also take `turn_idx` to switch between “first-pass” and “refine” behaviors.

2) **update_from_model(env_data or response, \*\*kwargs)**  
   Parse model output into an actionable format and store it in `self.current_action`. Implementations may receive `response: str` directly (e.g., code/math agents) even though the base signature includes `env_data`.

3) **step(env_data, env_worker=None)**  
   Execute the action, mutate `env_data.state`, and set `self.success` / `env_data.success` when a task is solved. This is where environment-specific side effects happen (running code, executing tools, etc.).

4) **calculate_reward(env_data)**
   Calculate the reward signal and store it in `self.agent_reward`. In PettingLLMs, this is typically **local reward + team reward** so multi-agent cooperation is reflected in each agent's return. The execution engine is responsible for tracking rewards across turns.

5) **reset()**  
   Clear transient fields so the agent can start a fresh episode.

## Reward: Local + Team Examples

### Code Environment

`UnitTestGenerationAgent` combines its own test quality with the code agent’s pass ratio:

```python
# pettingllms/multi_agent_env/code/agents/unit_test_agent.py
def calculate_reward(self, env_data: Env):
    self.agent_reward = (
        env_data.state.generated_test_vs_golden_code_match_ratio   # local: how good the generated tests are
        + env_data.state.ground_truth_test_vs_generated_code_match_ratio  # team: how well the code agent passed the ground-truth tests
    )
```

`CodeGenerationAgent` mirrors the same pattern by summing the code pass ratio twice (self + team) to keep the reward additive for cooperative training:

```python
# pettingllms/multi_agent_env/code/agents/code_agent.py
def calculate_reward(self, env_data: Env):
    self.agent_reward = (
        env_data.state.ground_truth_test_vs_generated_code_match_ratio
        + env_data.state.ground_truth_test_vs_generated_code_match_ratio
    )
```

### Math Environment

`ToolAgent` first sets a local reward in `step` (1.0 for correct execution, 0.0 or -1 for errors), then adds the teammate’s reasoning correctness during `calculate_reward`:

```python
# pettingllms/multi_agent_env/math/agents/tool_agent.py
def calculate_reward(self, env_data: Env):
    self.agent_reward = self.agent_reward + int(env_data.state.reasoning_is_correct)  # team bonus from reasoning agent
```

`ReasoningAgent` follows the same additive pattern, counting reasoning correctness twice to reflect both self and team contributions:

```python
# pettingllms/multi_agent_env/math/agents/reasoning_agent.py
def calculate_reward(self, env_data: Env):
    self.agent_reward = int(env_data.state.reasoning_is_correct) + int(env_data.state.reasoning_is_correct)
```

## Minimal Turn Loop

```python
agent.update_from_env(env_data=env, turn_idx=turn)     # read shared state
agent.update_from_model(response=model_out)            # parse model output
await agent.step(env)                                  # write to env.state
agent.calculate_reward(env)                            # local + team reward
agent.reset() if env.done else None                    # cleanup between episodes
```

Use this sequence for every agent involved in a rollout so shared state and rewards stay consistent with the base interfaces.
