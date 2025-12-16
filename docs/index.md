# PettingLLMs

<div align="center">
<img src="figs/logo.svg" alt="PettingLLMs Logo" width="400">
</div>

**Reinforcement Learning Framework for Multi LLM Agents** 🚀🌟

<div align="center">
<img src="figs/pettingllms.svg" alt="PettingLLMs Overview" width="800">
</div>

## Overview

**PettingLLMs** is an open-source framework for **on-policy reinforcement learning (RL) with multi-agent large language models (LLMs)**.  

It implements **AT-GRPO** (Agent- and Turn-wise Group Relative Policy Optimization), a novel algorithm and system design for training collaborative LLM agents across **planning, coding, and mathematical reasoning tasks**.

## 📚 Documentation Directory

- [Home](https://pettingllms-docs.readthedocs.io/en/latest/)
- **Getting Started**
  - [Installation](https://pettingllms-docs.readthedocs.io/en/latest/getting-started/installation/)
  - [Quick Start](https://pettingllms-docs.readthedocs.io/en/latest/getting-started/quick-start/)
  - [Datasets](https://pettingllms-docs.readthedocs.io/en/latest/getting-started/datasets/)
  - [Training](https://pettingllms-docs.readthedocs.io/en/latest/getting-started/training/)
  - [Evaluation](https://pettingllms-docs.readthedocs.io/en/latest/getting-started/evaluation/)
- **Environment Setup**
  - [Overview](https://pettingllms-docs.readthedocs.io/en/latest/environment_setup/index/)
  - [Data Preparation](https://pettingllms-docs.readthedocs.io/en/latest/environment_setup/data-preparation/)
  - [Core Architecture](https://pettingllms-docs.readthedocs.io/en/latest/environment_setup/core-architecture/)
  - [Agent Functions](https://pettingllms-docs.readthedocs.io/en/latest/environment_setup/agent-functions/)
  - [Environment State](https://pettingllms-docs.readthedocs.io/en/latest/environment_setup/environment-state/)
  - [Configuration](https://pettingllms-docs.readthedocs.io/en/latest/environment_setup/configuration/)
  - [Registration](https://pettingllms-docs.readthedocs.io/en/latest/environment_setup/registration/)
- **Core Concepts**
  - [Overview](https://pettingllms-docs.readthedocs.io/en/latest/core-concepts/overview/)
  - [AT-GRPO Algorithm](https://pettingllms-docs.readthedocs.io/en/latest/core-concepts/at-grpo/)
  - [Three-Level Specialization](https://pettingllms-docs.readthedocs.io/en/latest/core-concepts/three-level-specialization/)
  - [Training System](https://pettingllms-docs.readthedocs.io/en/latest/core-concepts/training-system/)
- **Training Guides**
  - [Overview](https://pettingllms-docs.readthedocs.io/en/latest/training/overview/)
  - [Games](https://pettingllms-docs.readthedocs.io/en/latest/training/games/)
  - [Planning](https://pettingllms-docs.readthedocs.io/en/latest/training/planning/)
  - [Code](https://pettingllms-docs.readthedocs.io/en/latest/training/code/)
  - [Math](https://pettingllms-docs.readthedocs.io/en/latest/training/math/)
- **Evaluation**
  - [Evaluation Guide](https://pettingllms-docs.readthedocs.io/en/latest/evaluation/guide/)
- **Results**
  - [Benchmarks](https://pettingllms-docs.readthedocs.io/en/latest/results/benchmarks/)
  - [Ablation Studies](https://pettingllms-docs.readthedocs.io/en/latest/results/ablations/)
- **API Reference**
  - [Python API](https://pettingllms-docs.readthedocs.io/en/latest/api/)
- **About Us**
  - [Team](https://pettingllms-docs.readthedocs.io/en/latest/About_us/)

## Supported Training Modes

This framework supports:

- ✅ **Single-agent RL training**  
- ✅ **Multi-agent RL training (role-sharing policy)**  
- ✅ **Multi-agent RL training (role-specialized policies using different LoRA adapters or different LLMs)**  

## 📰 News

- **[2025.10]** 🚀 GitHub repository open-sourced and publicly available
- **[2025.10]** 🎉 Paper released! Check out our [arxiv preprint](https://arxiv.org/pdf/2510.11062)
- **[2025.10]** 🔥 Support for different LoRA adapters per agent role - enabling efficient role-specialized training
- **[2025.09]** 🌍 Multi-environment support added: Game (Sudoku, Sokoban), Code (APPS, CodeContests), and Math (AIME, OlympiadBench)
- **[2025.08]** 🤖 Multi-agent framework implementation: support for both shared single model and role-specific models

## 🚀 Key Features

- **Multi-Level Agent Specialization**: Train and specialize agents at any level, from lightweight prompt adjustments to full model fine-tuning with LoRA or reinforcement learning.
- **Novel RL Algorithm**: Implements Agent- and turn wise GRPO- **AT-GRPO** for efficient and stable multi-agent training.
- **Built-in Multi-Turn MAS Workflows**: Comes with predefined, reproducible benchmarks and environments for a variety of domains:
  - 🎮 **Games**: Sudoku (4x4), Sokoban (6x6)
  - 📐 **Planning**: Plan-Path (10x10 grid)
  - 💻 **Coding**: APPS, CodeContests, LiveCodeBench
  - 🔢 **Math**: AIME24/25, OlympiadBench

## 🚩 Roadmap

- [ ] **More Environments**: Verilog design, web search, robotics, database query, scientific discovery
- [ ] **Multi-Modal Support**: Vision-language models, audio processing, mixed-modal tasks
- [ ] **Agentic Framework Integration**: AutoGen, LangGraph, CrewAI, and custom framework APIs

## 📊 Key Results

<div align="center">
<img src="figs/pettingllms_performance_comparison.png" alt="PettingLLMs performance" width="800">
</div>

**Table 3 · Ablation on Plan-Path (Qwen3-1.7B)**

| Method                                       | Acc.(%) |      Δ |
| -------------------------------------------- | ------: | -----: |
| Single agent                                 |    5.00 |      – |
| Training tool agent in SA, eval in SA        |   11.00 |  +6.00 |
| Training code agent in SA, eval in SA        |   14.50 |  +9.50 |
| Training in SA, eval in MAS                  |   16.00 | +11.00 |
| MAS RL (role specific policies), eval in MAS |   96.00 | +91.00 |
| w/ Swapped Policies                          |    6.00 |  +1.00 |

## 🔁 Environment Workflows (MA vs. SA)

<div align="center">
<img src="figs/workflow.png" alt="PettingLLMs worker" width="800">
</div>

## 📦 Installation

```bash
git clone https://github.com/pettingllms-ai/PettingLLMs.git
cd PettingLLMs
bash setup.bash
```

## 🎯 Quick Start

### 1. Dataset Preparation

Prepare datasets for different tasks:

```bash
# Code tasks (APPS, CodeContests, LiveCodeBench)
python scripts/dataprocess/load_code.py

# Math tasks (AIME24/25, OlympiadBench)
python scripts/dataprocess/load_math.py

# Game/Planning tasks (Sokoban, Sudoku)
python scripts/dataprocess/load_sokoban.py
```

Datasets will be saved to `datasets/code/`, `datasets/math/`, and `datasets/sudoku_environments/`.

### 2. Training

**Example: Train multi-agent system on math tasks**

```bash
bash scripts/train/math/math_L1_prompt.sh
```

Other training scripts available in `scripts/train/`:
- `code_single_policy.sh`, `code_two_policy.sh` - Code domain
- `plan_path_single.sh`, `plan_path_two_policy.sh` - Planning domain
- `sokoban_two_policy.sh`, `sokodu_single.sh` - Game domain

### 3. Evaluation

**Example: Evaluate trained model**

Edit `scripts/evaluate/evaluate.sh` to set your model path and config:
```bash
MODEL_PATHS=("/path/to/your/model")
CONFIG_NAME="math_single_policy"
```

Then run:
```bash
bash scripts/evaluate/evaluate.sh
```

## 🧱 Three Levels of Agent Specialization

PettingLLMs uses a tiered approach to define agent roles, ranging from simple instructions to deep model specialization.

| Level | Role Specialization Method | Description |
| :--- | :--- | :--- |
| **L0** | **Shared model** | Roles are defined *solely through instructions* in the prompt. The base model is identical for all agents, offering a flexible but performance-limited baseline. |
| **L1** | **LoRA** | Each role is specialized using a unique, lightweight **LoRA adapter**. This creates distinct, cost-effective agent "personalities" on top of a shared base model. |
| **L2** | **Full-Model** | The **entire model's weights** are optimized for a specific role using reinforcement learning. This creates a highly specialized expert agent for maximum performance on complex tasks. |

## Quick Links

- [Installation Guide](getting-started/installation.md) - Get started in minutes
- [Quick Start Tutorial](getting-started/quick-start.md) - Run your first training
- [Core Concepts](core-concepts/overview.md) - Understand the framework
- [Training Guides](training/overview.md) - Train on different tasks
- [API Reference](api/index.md) - Detailed API documentation



## 🔗 Acknowledgements

This work was primarily conducted by Yujie Zhao during her summer internship at **Intel Corporation**. We gratefully acknowledge Intel's support and resources that made this research possible.

- **VERL**: [VERL: Efficient RL Training for LLMs](https://github.com/volcengine/verl) - For efficient distributed RL training infrastructure
- **RLLM**: [RLLM: Reinforcement Learning with Language Models](https://github.com/mukobi/rllm) - For foundational RL algorithms for LLMs

## 📌 License

Released under the MIT license.
See LICENSE for details.
