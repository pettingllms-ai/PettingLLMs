<div align="center">

<div align="center">
<img src="figs/logo.svg" alt="PettingLLMs Logo" width="400">
</div>

# PETTINGLLMS

<div>
🚀 RL Framework for training Multi Agent System(MAS).🌟
</div>
</div>
<div>
<br>


<div align="center">
<img src="figs/pettingllms.svg" alt="PettingLLMs Logo" width="800">
</div>

</div>
PettingLLMs is an open-source framework for on-policy reinforcement learning (RL) with multi-agent large language models (LLMs).  It implements AT-GRPO (Agent- and Turn-wise Group Relative Policy Optimization), a novel algorithm and system design for training collaborative LLM agents across **planning, coding, and mathematical reasoning tasks. This repo supports:
- ✅ Single-agent(SA) RL training  
- ✅ Multi-agent RL training (role-sharing policy)  
- ✅ Multi-agent RL training (role-specialized policies)  
# 🚀 Key Features

-   **Multi-Level Agent Specialization**: Train and specialize agents at any level, from lightweight prompt adjustments to full model fine-tuning with LoRA or reinforcement learning.
-   **Novel RL Algorithm**: Implements Agent- and turn wise GRPO- **AT-GRPO** for efficient and stable multi-agent training.
-   **Built-in Multi-Turn MAS Workflows**: Comes with predefined, reproducible benchmarks and environments for a variety of domains:
    -   🎮 **Games**: Sudoku (4x4), Sokoban (6x6)
    -   📐 **Planning**: Plan-Path (10x10 grid)
    -   💻 **Coding**: APPS, CodeContests, LiveCodeBench
    -   🔢 **Math**: AIME24/25, OlympiadBench


---

## 🧱 Three Levels of Role Optimization

PettingLLMs offers a tiered approach to agent specialization, allowing you to balance performance with computational cost.

| Level | Optimization Target | Trainable Params | Typical Use Case | Strengths | Trade-offs |
| :--- | :--- | ---: | :--- | :--- | :--- |
| **L0: Prompt Engineering** | Prompt templates only | **0%** | Rapid baselines, ablation studies, and orchestration tuning. | No training cost; instant iteration. | Performance is highly sensitive to prompt design; limited ceiling. |
| **L1: LoRA Adaptation** | Low-rank adapters (per role) | **~0.1–2%** | Cost-effective specialization for specific roles (e.g., a "Tool User" or "Judge" agent). | Excellent balance of performance gain vs. training cost/latency. | Requires managing separate adapters for each specialized role. |
| **L2: Full-Model RL** | Full model parameters via on-policy RL (AT-GRPO) | **100%** | Achieving maximum performance on complex, long-horizon tasks. | Highest performance ceiling; enables precise credit assignment per role/turn. | Computationally intensive; requires careful hyperparameter tuning. |



## 📊 Key Results

PettingLLMs demonstrates **substantial gains** over single-agent GRPO:

- **Planning**: From **14–47% → 96–99.5%** accuracy  
- **Coding**: +3.87–7.62% improvement  
- **Math**: +9.0–17.93% improvement  

See Tables 1–3 in the paper for full results.
Table 1 · Qwen3 1.7B results (game, planning, coding, math)
| Method                            |         Sudoku |        Sokoban |      Plan-Path |  LiveCodeBench |          APPS | CodeContests |         AIME24 |        AIME25 |  OlympiadBench |
| --------------------------------- | -------------: | -------------: | -------------: | -------------: | ------------: | -----------: | -------------: | ------------: | -------------: |
| Single agent                      |   7.00 (+0.00) |   0.00 (+0.00) |   5.00 (+0.00) |  11.60 (+0.00) | 16.20 (+0.00) | 3.60 (+0.00) |  13.40 (+0.00) |  9.80 (+0.00) |  22.20 (+0.00) |
| Single agent + GRPO               | 29.00 (+22.00) |   3.00 (+3.00) |  11.00 (+6.00) |  18.80 (+7.20) | 17.00 (+0.80) | 3.00 (-0.60) |  10.00 (-3.40) |  6.70 (-3.10) |  23.80 (+1.60) |
| MAS                               | 69.00 (+62.00) |   0.00 (+0.00) |  10.00 (+5.00) |  19.00 (+7.40) | 16.60 (+0.40) | 3.60 (+0.00) | 13.30 (+-0.10) | 13.00 (+3.20) | 35.90 (+13.70) |
| MAS + AT-GRPO (shared policy)     | 99.00 (+92.00) | 10.00 (+10.00) | 96.00 (+91.00) |  20.90 (+9.30) | 17.60 (+1.40) | 4.80 (+1.20) |  16.70 (+3.30) | 16.70 (+6.90) | 39.60 (+16.80) |
| MAS + AT-GRPO (per-role policies) | 99.00 (+92.00) | 11.50 (+11.50) | 97.00 (+92.00) | 24.00 (+12.40) | 18.60 (+2.40) | 7.80 (+4.20) | 13.30 (+-0.10) | 18.30 (+8.50) | 35.20 (+13.00) |



Table 2 · Qwen3 8B results (game, planning, coding, math)
---
| Method                            |         Sudoku |        Sokoban |      Plan-Path |  LiveCodeBench |           APPS |  CodeContests |         AIME24 |         AIME25 | OlympiadBench |
| --------------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: | ------------: | -------------: | -------------: | ------------: |
| Single agent                      |  48.00 (+0.00) |   9.00 (+0.00) |  12.00 (+0.00) |  22.80 (+0.00) |  30.20 (+0.00) | 15.75 (+0.00) |  18.30 (+0.00) |  20.00 (+0.00) | 55.00 (+0.00) |
| Single agent + GRPO               |  54.00 (+6.00) |  14.00 (+5.00) | 47.00 (+35.00) |  25.70 (+2.90) |  37.00 (+6.80) | 12.12 (-3.63) |  18.30 (+0.00) |  26.67 (+6.67) | 54.80 (-0.20) |
| MAS                               | 72.00 (+24.00) |  16.00 (+7.00) | 71.00 (+59.00) |  28.00 (+5.20) | 44.40 (+14.20) | 17.60 (+1.85) | 36.60 (+18.30) | 30.00 (+10.00) | 56.50 (+1.50) |
| MAS + AT-GRPO (shared policy)     | 99.50 (+51.50) | 96.00 (+87.00) | 93.00 (+81.00) |  30.28 (+7.48) | 45.80 (+15.60) | 18.10 (+2.35) | 50.00 (+31.70) | 35.20 (+15.00) | 56.80 (+1.80) |
| MAS + AT-GRPO (per-role policies) | 99.00 (+51.00) | 98.00 (+89.00) | 96.00 (+84.00) | 33.10 (+10.30) | 46.50 (+16.30) | 18.10 (+2.35) | 57.00 (+38.70) | 40.00 (+20.00) | 56.60 (+1.60) |

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
git clone https://github.com/NorahYujieZhao/PettingLLMs.git
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
bash scripts/train/math.sh
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

## 📌 License

Released under the Apache 2.0 license.
See LICENSE
 for details.