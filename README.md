
## 📦 Installation

```bash
git clone https://github.com/pettingllms-ai/PettingLLMs.git
cd /mnt/afs/zhangyaolun/safe_model/tool/PettingLLMs
bash setup.bash
bash scripts/train/math/math_L1_prompt.sh
```

## 🎯 Quick Start


### 2) Training

Example: train a multi-agent system on math tasks.

```bash
bash scripts/train/math/math_L1_prompt.sh
```

Other training scripts live in `scripts/train/`:
- `code_single_policy.sh`, `code_two_policy.sh` (code)
- `plan_path_single.sh`, `plan_path_two_policy.sh` (planning)
- `sokoban_two_policy.sh`, `sokodu_single.sh` (games)

