# Environment State

`env_data.state` is the single source of truth that every agent reads from and writes to. This page aligns the documentation with the base environment definitions in `pettingllms/multi_agent_env/base` and highlights how the state fields feed the **local + team** reward design in the code and math environments.

## Base Container (pettingllms/multi_agent_env/base/env.py)

```python
from abc import abstractmethod

class Env:
    def __init__(self, env_idx: int, rollout_idx: int, config: dict | None = None):
        """
        Initialize the multi-agents environment.

        Args:
            env_idx: Environment index
            rollout_idx: Rollout index
            config: Configuration for the system
        """
        # Save configuration
        self.config = config

        # Initialize variables required by step method
        self.done = False
        self.state = None  # domain-specific dataclass
        self.success = False

    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment based on the action.

        Returns:
            next_observation, reward, terminated, truncated, info
        """
        return NotImplementedError("Subclasses must implement this method")
```

**Key Changes from Previous Version:**
- ✅ **Removed**: `max_turns` parameter - Now read from `config.env.max_turns` instead
- ✅ **Removed**: `history`, `task`, `current_turn` - These fields were not used by the execution engine
- ✅ **Simplified**: Only 3 essential fields remain: `done`, `state`, `success`

- `state` is a **domain-specific dataclass** (see below). All agent coordination happens through this object.
- `done` and `success` are base fields every environment shares; agent implementations extend these with domain signals.

## Domain State Shapes

### Code Environment (`CodeEnvState`)

Defined in `pettingllms/multi_agent_env/code/code_env.py`. Key fields that agents read/write:

- Problem: `problem`, optional `golden_code`
- Generated artifacts: `generated_code`, `generated_test_input/output`, `generated_code_history`
- Execution traces: `exe_code_generated_test_output`, `exe_code_ground_truth_test_output`
- Evaluation (local & team rewards):
  - `ground_truth_test_vs_generated_code_match_ratio` and match/mismatch cases (set by `CodeGenerationAgent.step`)
  - `generated_test_vs_golden_code_match_ratio` and cases (set by `UnitTestGenerationAgent.step`)
  - `generated_test_vs_generated_code_match_ratio` and history (shared feedback loop)

### Math Environment (`MathEnvState`)

Defined in `pettingllms/multi_agent_env/math/math_env.py`. Key fields:

- Problem and answers: `problem`, `ground_truth_answer`
- Agent outputs: `reasoning_generated_solution`, `code_generated_solution`
- Extracted answers: `reasoning_extracted_answer`, `code_extracted_answer`
- Evaluation flags: `reasoning_is_correct`, `code_is_correct`, `code_reasoning_aligned`
- Histories: `reasoning_generated_solution_history`, `code_generated_solution_history`, corresponding extracted-answer histories

## Reward Signals from State (Local + Team)

Agents compute rewards in `calculate_reward(env_data)` by combining their own performance with a teammate’s signal stored in `env_data.state`.

- **Code environment**
  - `UnitTestGenerationAgent.calculate_reward`
    `agent_reward = generated_test_vs_golden_code_match_ratio (local) + ground_truth_test_vs_generated_code_match_ratio (team from code agent)`
  - `CodeGenerationAgent.calculate_reward`
    Adds `ground_truth_test_vs_generated_code_match_ratio` twice (self + team) to keep the cooperative reward additive.

- **Math environment**
  - `ToolAgent.calculate_reward`
    Starts with the local reward set in `step` (code correctness), then adds `int(env_data.state.reasoning_is_correct)` as the team bonus from the reasoning agent.
  - `ReasoningAgent.calculate_reward`
    Sums `int(env_data.state.reasoning_is_correct)` twice to reflect self + team contribution from the same correctness flag.

These patterns make every agent's return depend on both its own output and shared team success, encouraging coordinated policies.

**Note**: The execution engine now tracks rewards externally. Agents simply set `self.agent_reward` in `calculate_reward()`, and the engine reads this value for training.
