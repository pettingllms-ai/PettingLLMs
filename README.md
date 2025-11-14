<div align="center">

<div align="center">
<img src="figs/logo.svg" alt="PettingLLMs Logo" width="400">
</div>

# PETTINGLLMS

<div>
🚀 RL Framework for training Multi Agentic LLMs.🌟
</div>

<p align="center">
  <a href="https://www.notion.so/STRONGER-TOGETHER-ON-POLICY-REINFORCEMENT-LEARNING-FOR-COLLABORATIVE-LLMS-2833b123b9438102b496d3673c8b4094?source=copy_link">📝 Blog</a>•<a href="https://arxiv.org/pdf/2510.11062">📄 Paper</a> •
  <a href="https://pettingllms.github.io">🌐 Website</a> •
  <a href="https://www.youtube.com/watch?v=8WM-gVTrSBc">🎮 Demo</a> 
  
</p>

</div>
<div>
<br>


<div align="center">
<img src="figs/pettingllms.svg" alt="PettingLLMs Logo" width="800">
</div>

</div>
PettingLLMs is an open-source framework for on-policy reinforcement learning (RL) with multi-agent large language models (LLMs).  It implements AT-GRPO (Agent- and Turn-wise Group Relative Policy Optimization), a novel algorithm and system design for training collaborative LLM agents across **planning, coding, and mathematical reasoning tasks**. 

This repo supports:
- ✅ Single-agent(SA) RL training  
- ✅ Multi-agent RL training (one role-sharing policy)  
- ✅ Multi-agent RL training (role-specialized policies using different lora adaptor or different LLMs)

---

## 📰 News
- **[2025.10]** 🚀 GitHub repository open-sourced and publicly available

- **[2025.10]** 🎉 Paper released! Check out our [arxiv preprint](hhttps://arxiv.org/pdf/2510.11062)

- **[2025.10]** 🔥 Support for different LoRA adapters per agent role - enabling efficient role-specialized training
- **[2025.09]** 🌍 Multi-environment support added: Game (Sudoku, Sokoban), Code (APPS, CodeContests), and Math (AIME, OlympiadBench)
- **[2025.08]** 🤖 Multi-agent framework implementation: support for both shared single model and role-specific models
---  


## 🚀 Key Features

-   **Multi-Level Agent Specialization**: Train and specialize agents at any level, from lightweight prompt adjustments to full model fine-tuning with LoRA or reinforcement learning.
-   **Novel RL Algorithm**: Implements Agent- and turn wise GRPO- **AT-GRPO** for efficient and stable multi-agent training.
-   **Built-in Multi-Turn MAS Workflows**: Comes with predefined, reproducible benchmarks and environments for a variety of domains:
    -   🎮 **Games**: Sudoku (4x4), Sokoban (6x6)
    -   📐 **Planning**: Plan-Path (10x10 grid)
    -   💻 **Coding**: APPS, CodeContests, LiveCodeBench
    -   🔢 **Math**: AIME24/25, OlympiadBench

---

## 🚩 Roadmap

- [ ] **More Environments**: Verilog design, web search, robotics, database query, scientific discovery
- [ ] **Multi-Modal Support**: Vision-language models, audio processing, mixed-modal tasks
- [ ] **Agentic Framework Integration**: AutoGen, LangGraph, CrewAI, and custom framework APIs




## 📊 Key Results

<div align="center">
<img src="figs/pettingllms_performance_comparison.png" alt="PettingLLMs performance" width="800">
</div>

Table 3 · Ablation on Plan-Path (Qwen3-1.7B)
---
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

---

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

---
Of course, here is a more concise version focusing on how agent roles are differentiated at each level.

***

### 🧱 Three Levels of Agent Specialization

PettingLLMs uses a tiered approach to define agent roles, ranging from simple instructions to deep model specialization.

| Level | Role Specialization Method | Description |
| :--- | :--- | :--- |
| **L0** | **Prompt** | Roles are defined *solely through instructions* in the prompt. The base model is identical for all agents, offering a flexible but performance-limited baseline. |
| **L1** | **LoRA** | Each role is specialized using a unique, lightweight **LoRA adapter**. This creates distinct, cost-effective agent "personalities" on top of a shared base model. |
| **L2** | **Full-Model** | The **entire model's weights** are optimized for a specific role using reinforcement learning. This creates a highly specialized expert agent for maximum performance on complex tasks. |

## 📚 Citation

If you find PettingLLMs useful for your research or projects, please cite:

```bibtex
@article{pettingllms2025,
  title={PettingLLMs: Reinforcement Learning for Multi-Agent Large Language Models},
  author={Your Authors},
  journal={arXiv preprint arXiv:2510.11062},
  year={2025}
}
```


## 🔗 Acknowledgements

This work was primarily conducted by **Yujie Zhao** during her summer internship at **Intel Corporation**. We gratefully acknowledge Intel's support and resources that made this research possible.


- **VERL**: [VERL: Efficient RL Training for LLMs](https://github.com/volcengine/verl) - For efficient distributed RL training infrastructure
- **RLLM**: [RLLM: Reinforcement Learning with Language Models](https://github.com/mukobi/rllm) - For foundational RL algorithms for LLMs

---

## 📌 License

Released under the MIT license.
See LICENSE for details.