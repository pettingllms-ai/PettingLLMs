# PyChecker RL Environment

Standalone RL environment for training agents to generate Python GoldenDUT code for hardware verification.

## Overview

This module provides a reinforcement learning environment where agents learn to generate Python code that emulates hardware behavior described in Verilog. The agent receives rewards based on how well the generated Python code matches the expected hardware functionality.

## Reward Structure

- **1.0**: Python code runs correctly and matches the golden DUT output
- **0.3**: Python code runs successfully but output doesn't match
- **0.0**: Code fails to extract or encounters runtime errors

## Directory Structure

```
pychecker_rl/
├── pychecker_env.py          # Main RL environment
├── agents/
│   ├── pychecker_agent.py    # RL agent implementation
│   └── __init__.py
├── utils/
│   ├── flexible_parser.py    # Code extraction utilities
│   ├── simulator.py          # Python code execution & evaluation
│   └── __init__.py
├── sim_cmb/                  # Combinational circuit simulation templates
├── sim_seq/                  # Sequential circuit simulation templates
└── README.md                 # This file
```

## Usage

### Loading Dataset

The environment loads problems from `rl_4testbench/dataset/dataset.jsonl`. Each line should be a JSON object with:

```json
{
  "input": "Problem description with module header",
  "output": "Golden Verilog DUT code"
}
```

### Creating Environment

```python
from pychecker_rl.pychecker_env import PyCheckerEnvBatch

# Create batch environment
env_batch = PyCheckerEnvBatch(
    env_idx_list=[0, 1, 2],
    env_indices=[0, 1, 2],
    rollout_idx_list=[0, 1, 2],
    samples=1,
    max_turns=5,
    config=config,
    mode="train"
)
```

### Using Agent

```python
from pychecker_rl.agents.pychecker_agent import PyCheckerAgent

# Create agent
agent = PyCheckerAgent(rollout_idx=0)

# Update from environment
agent.update_from_env(turn_idx=0, env_data=env)

# Get prompt for LLM
prompt = agent.current_prompt

# After LLM response
agent.update_from_model(llm_response)

# Execute step
await agent.step(env_data=env)

# Get reward
reward = agent.agent_reward
```

## Features

- **Standalone**: All dependencies are self-contained within this directory
- **Circuit Type Detection**: Automatically detects CMB (combinational) vs SEQ (sequential) circuits
- **Code Extraction**: Flexible parser handles multiple code formats (markdown, XML tags, etc.)
- **Simulation**: Executes Python code in isolated environment with timeout protection
- **History Tracking**: Maintains history of attempts for refinement

## Requirements

- Python 3.8+
- Standard library only (no external dependencies for core functionality)
- Optional: Verilator (for actual hardware simulation comparison)

## Extending

To add actual hardware simulation comparison:

1. Implement `compare_golden_dut()` in `utils/simulator.py`
2. Add Verilator-based simulation using templates in `sim_cmb/` and `sim_seq/`
3. Compare Python GoldenDUT output with Verilator simulation output

## Notes

- Circuit type is inferred from problem description keywords (clock, clk, sequential, state)
- Can be configured via `config.env.dataset_path` to use custom dataset
- Supports both combinational and sequential circuit types
- Code execution is sandboxed with timeout protection (default 30s)

