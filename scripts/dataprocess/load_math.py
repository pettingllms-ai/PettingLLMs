# build_testset.py
import os
import re
from pathlib import Path
import datasets

# ---------- Answer extraction covering common math benchmarks ----------
def extract_solution(answer_text: str) -> str:
    if answer_text is None:
        return ""
    s = str(answer_text).strip()

    # GSM8K-style: "#### xxx"
    m = re.search(r"####\s*([^\n\r]+)", s)
    if m:
        return m.group(1).strip().replace(",", "")

    # MATH-style: \boxed{...}
    m = re.search(r"\\boxed\{([^{}]+)\}", s)
    if m:
        return m.group(1).strip()

    # Explicit tail labels: "Final answer: xxx" / "Answer: xxx"
    m = re.search(r"(?:final answer|answer)\s*:\s*([^\n\r]+)", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Short final token on the last non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        tail = lines[-1]
        if re.fullmatch(r"[A-Za-z0-9\-\+\*/\.^,_]+", tail) and len(tail) <= 20:
            return tail.replace(",", "")

    return s


def get_first_key(d: dict, candidates):
    for k in candidates:
        if k in d and d[k] is not None:
            return d[k]
    return ""


DATASETS = {
    # AIME 2024
    # https://huggingface.co/datasets/Maxwell-Jia/AIME_2024
    "AIME24": {
        "path": "Maxwell-Jia/AIME_2024",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["Answer", "answer", "final_answer", "solution", "Solution"],
    },
    # AIME 2025
    # https://huggingface.co/datasets/yentinglin/aime_2025
    "AIME25": {
        "path": "yentinglin/aime_2025",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["Answer", "answer", "final_answer", "solution", "Solution"],
    },
    # OlympiadBench (English math, competition subset)
    # https://huggingface.co/datasets/Hothan/OlympiadBench
    "OlympiadBench": {
        "path": "Hothan/OlympiadBench",
        "subset": "OE_TO_maths_en_COMP",
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["Answer", "answer", "final_answer", "solution", "Solution"],
    },
}


def choose_available_split(ds_dict, prefer_splits):
    for sp in prefer_splits:
        if sp in ds_dict:
            return sp
    if len(ds_dict) > 0:
        return list(ds_dict.keys())[0]
    raise ValueError("No splits available in the loaded dataset.")


def standardize_hf_dataset(ds, q_keys, a_keys):
    def map_fn(example):
        q_raw = get_first_key(example, q_keys)
        a_raw = get_first_key(example, a_keys)
        return {
            "question": str(q_raw).strip(),
            "solution": extract_solution(a_raw),
        }
    cols_to_remove = [c for c in ds.column_names if c not in []]
    return ds.map(map_fn, remove_columns=cols_to_remove)


def process_dapo_math(ds):
    """Process DAPO-Math-17k dataset with deduplication."""
    seen_questions = set()
    processed_data = []

    for example in ds:
        # Extract question from prompt array
        prompt_list = example.get("prompt", [])
        if prompt_list and isinstance(prompt_list, list) and len(prompt_list) > 0:
            question = prompt_list[0].get("content", "").strip()
        else:
            continue

        # Extract answer from reward_model
        reward_model = example.get("reward_model", {})
        if isinstance(reward_model, dict):
            answer = str(reward_model.get("ground_truth", "")).strip()
        else:
            answer = ""

        # Skip empty questions or duplicates
        if not question or question in seen_questions:
            continue

        seen_questions.add(question)
        processed_data.append({
            "question": question,
            "solution": answer,
        })

    return datasets.Dataset.from_list(processed_data)


def main():
    project_root = Path(__file__).resolve().parents[2]
    out_train_dir = project_root / "datasets" / "math" / "train"
    out_test_dir = project_root / "datasets" / "math" / "test"
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    # 1) POLARIS train
    print("Loading POLARIS-Project/Polaris-Dataset-53K ...")
    polaris_all = datasets.load_dataset("POLARIS-Project/Polaris-Dataset-53K")
    polaris_split = choose_available_split(polaris_all, ["train", "test", "validation", "dev"])
    polaris_ds = polaris_all[polaris_split]
    print(f"Using split for POLARIS: {polaris_split} (as train)")
    polaris_std = standardize_hf_dataset(
        polaris_ds,
        ["problem", "question", "prompt", "Problem"],
        ["answer", "final_answer", "solution", "Solution", "Answer"],
    )
    polaris_path = out_train_dir / "polaris.parquet"
    polaris_std.to_parquet(str(polaris_path))
    print(f"Saved POLARIS train to: {polaris_path} ({len(polaris_std)} rows)")

    # 2) DAPO-Math-17k train (with deduplication)
    print("Loading BytedTsinghua-SIA/DAPO-Math-17k ...")
    dapo_all = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")
    dapo_split = choose_available_split(dapo_all, ["train", "test", "validation", "dev"])
    dapo_ds = dapo_all[dapo_split]
    print(f"Using split for DAPO-Math: {dapo_split}")
    print(f"Original size: {len(dapo_ds)} rows, deduplicating...")
    dapo_std = process_dapo_math(dapo_ds)
    dapo_path = out_train_dir / "dapo_math.parquet"
    dapo_std.to_parquet(str(dapo_path))
    print(f"Saved DAPO-Math train to: {dapo_path} ({len(dapo_std)} rows after deduplication)")

    # 2) Test sets: AIME24, AIME25, OlympiadBench
    for benchmark in ["AIME24", "AIME25", "OlympiadBench"]:
        conf = DATASETS[benchmark]
        path = conf["path"]
        subset = conf.get("subset", None)
        print(f"Loading {path}" + (f" (subset={subset})" if subset else "") + " ...")
        ds_dict = datasets.load_dataset(path, subset) if subset else datasets.load_dataset(path)
        split = choose_available_split(ds_dict, conf["prefer_splits"])
        ds = ds_dict[split]
        print(f"Using split for {benchmark}: {split} (as test)")
        ds_std = standardize_hf_dataset(ds, conf["q_keys"], conf["a_keys"])
        out_path = out_test_dir / f"{benchmark}.parquet"
        ds_std.to_parquet(str(out_path))
        print(f"Saved {benchmark} test to: {out_path} ({len(ds_std)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
