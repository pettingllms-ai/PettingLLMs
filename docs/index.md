# PettingLLMs

<div align="center">
<img src="figs/logo.svg" alt="PettingLLMs Logo" width="800">
</div>

**Reinforcement Learning Framework for Multi LLM Agents** 🚀🌟

<div align="center">
<img src="figs/pettingllms.svg" alt="PettingLLMs Overview" width="800">
</div>

## Overview

**PettingLLMs** is an open-source framework for **on-policy reinforcement learning (RL) with multi-agent large language models (LLMs)**.  

It implements **AT-GRPO** (Agent- and Turn-wise Group Relative Policy Optimization), a novel algorithm and system design for training collaborative LLM agents across **planning, coding, and mathematical reasoning tasks**.

## Supported Training Modes

This framework supports:

- ✅ **Single-agent RL training**  
- ✅ **Multi-agent RL training (role-sharing policy)**  
- ✅ **Multi-agent RL training (role-specialized policies)**  

## Key Features

- **AT-GRPO Algorithm**: Extends GRPO with **agent- and turn-wise grouping**, tree-structured sampling, and mixed global-local rewards.  
- **Multi-Agent RL System**: GPU-pinned resource pools with **RolloutWorkers** and **UpdateWorkers** for scalable on-policy updates.  
- **Workflow Support**: Built-in MAS workflows for:
  - 🎮 **Games**: Sudoku 4×4, Sokoban 6×6  
  - 📐 **Planning**: Plan-Path (10×10 grid)  
  - 💻 **Coding**: APPS, CodeContests, LiveCodeBench  
  - 🔢 **Math**: AIME24/25, OlympiadBench  
- **Flexible Policies**: Train with either **role-sharing** or **role-specialized** policies, depending on task domain.  
- **Reproducible Benchmarks**: Predefined datasets, verifiers, and configs ensure reproducible results.

## Performance Highlights

PettingLLMs demonstrates **substantial gains** over single-agent GRPO:

- **Planning**: From **14–47% → 96–99.5%** accuracy  
- **Coding**: +3.87–7.62% improvement  
- **Math**: +9.0–17.93% improvement  

See the [Benchmark Results](results/benchmarks.md) page for full details.

## Quick Links

- [Installation Guide](getting-started/installation.md) - Get started in minutes
- [Quick Start Tutorial](getting-started/quick-start.md) - Run your first training
- [Core Concepts](core-concepts/overview.md) - Understand the framework
- [Training Guides](training/overview.md) - Train on different tasks
- [API Reference](api/index.md) - Detailed API documentation

## Citation

If you use PettingLLMs in your research, please cite our paper:

```bibtex
@article{pettingllms2025,
  title={PettingLLMs: Reinforcement Learning for Multi-Agent LLM Collaboration},
  author={Your Name and Team},
  journal={arXiv preprint},
  year={2025}
}
```

## License

Released under the Apache 2.0 license. See [LICENSE](https://github.com/NorahYujieZhao/PettingLLMs/blob/main/LICENSE) for details.

