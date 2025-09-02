"""
用于 ALFWorld 的动作提取与对齐：
- 从 LLM 自然语言输出中抽取原子动作（如 "open fridge", "take apple from fridge"）
- 与 admissible actions 做鲁棒匹配（精确匹配优先，失败则模糊匹配）
"""

import re
from typing import List, Optional
import os
import yaml
try:
    # 更快&更稳的模糊匹配（比 difflib 更好用）
    from rapidfuzz import process, fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    import difflib
    _HAS_RAPIDFUZZ = False


def load_config_file():
    path = os.path.join(os.path.dirname(__file__), './configs/config_tw.yaml')
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def build_prompt_from_obs(obs: dict) -> str:
    task = obs.get("task_desc", "") or ""
    observation = obs.get("observation", "") or ""
    inventory = obs.get("inventory", "") or ""
    admissible = obs.get("admissible_actions", []) or []

    admissible_block = "\n".join(f"- {a}" for a in admissible)

    # 约定输出格式，提高可解析性
    prompt = (
        "You are an embodied agent in a textual household environment (ALFWorld).\n"
        "Choose EXACTLY ONE action from the admissible list. Respond ONLY with the action string.\n\n"
        f"Task:\n{task}\n\n"
        f"Observation:\n{observation}\n\n"
        f"Inventory:\n{inventory}\n\n"
        "Admissible actions:\n"
        f"{admissible_block}\n\n"
        "Output format:\n"
        "<action>your_action_here</action>\n"
    )
    return prompt


_ACTION_TAG = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)

def extract_action_from_text(text: str) -> str:

    m = _ACTION_TAG.search(text)
    if m:
        return cleanup_action(m.group(1))

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""

    raw = lines[0]
    raw = re.sub(r"^(Action)\s*[:：]\s*", "", raw, flags=re.IGNORECASE)
    return cleanup_action(raw)


def cleanup_action(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s


# 3) 将“意图动作”对齐到可执行动作（admissible actions）
def choose_executable_action(intent: str, admissible: List[str], threshold: int = 85) -> str:
    """
    规则：
      a) 若 intent 精确命中，则直接返回
      b) 否则做模糊匹配（rapidfuzz/token_sort_ratio），分数>=threshold 则返回最佳匹配
      c) 兜底：返回第一个 admissible（否则空串）
    """
    if not admissible:
        return intent or ""

    # 先 exact / 前缀 / 子串 命中
    norm = intent.lower().strip()
    for a in admissible:
        if norm == a.lower().strip():
            return a
    # 子串/前缀
    for a in admissible:
        low = a.lower()
        if norm and (norm in low or low.startswith(norm)):
            return a

    # 模糊匹配
    if _HAS_RAPIDFUZZ:
        best, score, _ = process.extractOne(
            norm,
            admissible,
            scorer=fuzz.token_sort_ratio
        )
        if score >= threshold:
            return best
    else:
        # 简单 difflib 兜底
        import difflib
        candidates = difflib.get_close_matches(norm, admissible, n=1, cutoff=0.6)
        if candidates:
            return candidates[0]

    # 实在不行：返回第一个
    return admissible[0]
