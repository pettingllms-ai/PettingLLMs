<div align="center">

<div align="center">
<img src="figs/logo.svg" alt="PettingLLMs Logo" width="800">
</div>

# PETTINGLLMS

<div>
🚀 Reinforcement Learning Framework for Multi LLM Agents🌟
</div>
</div>
<div>
<br>


<div align="center">
<img src="figs/pettingllms.svg" alt="PettingLLMs Logo" width="800">
</div>

</div>
**PettingLLMs** is an open-source framework for **on-policy reinforcement learning (RL) with multi-agent large language models (LLMs)**.  
It implements **AT-GRPO** (Agent- and Turn-wise Group Relative Policy Optimization), a novel algorithm and system design for training collaborative LLM agents across **planning, coding, and mathematical reasoning tasks**.

This repo supports:
- ✅ Single-agent RL training  
- ✅ Multi-agent RL training (role-sharing policy)  
- ✅ Multi-agent RL training (role-specialized policies)  

---

## 🚀 Features

- **AT-GRPO Algorithm**: Extends GRPO with **agent- and turn-wise grouping**, tree-structured sampling, and mixed global-local rewards.  
- **Multi-Agent RL System**: GPU-pinned resource pools with **RolloutWorkers** and **UpdateWorkers** for scalable on-policy updates.  
- **Workflow Support**: Built-in MAS workflows for:
  - 🎮 Games: Sudoku 4×4, Sokoban 6×6  
  - 📐 Planning: Plan-Path (10×10 grid)  
  - 💻 Coding: APPS, CodeContests, LiveCodeBench  
  - 🔢 Math: AIME24/25, OlympiadBench  
- **Flexible Policies**: Train with either **role-sharing** or **role-specialized** policies, depending on task domain.  
- **Reproducible Benchmarks**: Predefined datasets, verifiers, and configs ensure reproducible results.  

---

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
- **Game / Plan (Sudoku, Sokoban, Plan-Path)**
  - **MA:** *Planner* proposes actions; *Executor* calls tools & returns effects/observations. Reward reflects step/terminal goal; terminate on goal or turn budget. :contentReference[oaicite:0]{index=0}
  - **SA:** Single agent outputs a plan under the same termination rule. :contentReference[oaicite:1]{index=1}

- **Code (APPS, CodeContests, LiveCodeBench)**
  - **MA:** *Tester* builds/refines unit tests; *Coder* implements/refines code; tests run every turn; reward is per-case/aggregate pass–fail; terminate on alignment (tests pass) or turn budget. :contentReference[oaicite:2]{index=2}
  - **SA:** One agent emits code; single-turn termination (no verification loop). :contentReference[oaicite:3]{index=3}

- **Math (AIME24/25, OlympiadBench)**
  - **MA:** *Tool agent* issues Python/calculator calls; *Reasoner* produces the final answer; reward is exact-match or verifier-checked; terminate on success or turn budget. :contentReference[oaicite:4]{index=4}
  - **SA:** One agent reasons (with direct tool calls if any) in a single turn; single-turn termination. :contentReference[oaicite:5]{index=5}


## 🏗️ Training System Framework Overview

<div align="center">
<img src="figs/worker.png" alt="PettingLLMs worker" width="400">
</div>
- **Per-policy GPU pools:** Each policy has a pinned **RolloutWorker** (inference) and **UpdateWorker** (optimize).
- **Strict on-policy:** RolloutWorkers refresh weights before sampling; UpdateWorkers train only on fresh, routed data.
- **Multi-policy ready:** Supports **role-sharing** (one policy) and **role-specialized** (per-role policies) training in parallel.
- **Scaled envs:** Thousands of sandboxed **CPU EnvWorkers** run Sudoku/Sokoban/Plan-Path/Code/Math with deterministic seeds.
- **Clean routing:** Observations, tool logs, and rewards stream to a Router that dispatches trajectories to the correct policy.


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