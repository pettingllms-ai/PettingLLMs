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
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
    # AIME 2025
    # https://huggingface.co/datasets/yentinglin/aime_2025
    "AIME25": {
        "path": "yentinglin/aime_2025",
        "subset": None,
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
    # OlympiadBench (English math, competition subset)
    # https://huggingface.co/datasets/Hothan/OlympiadBench
    "OlympiadBench": {
        "path": "Hothan/OlympiadBench",
        "subset": "OE_TO_maths_en_COMP",
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["problem", "question", "prompt", "Problem"],
        "a_keys": ["answer", "final_answer", "solution", "Solution", "Answer"],
    },
    # GSM8K
    # https://huggingface.co/datasets/openai/gsm8k
    "GSM8K": {
        "path": "openai/gsm8k",
        "subset": "main",
        "prefer_splits": ["test", "validation", "dev", "train"],
        "q_keys": ["question", "problem", "prompt"],
        "a_keys": ["answer", "solution", "final_answer"],
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


def main():
    project_root = Path(__file__).resolve().parents[2]
    out_train_dir = project_root / "datasets" / "math" / "train"
    out_test_dir = project_root / "datasets" / "math" / "test"
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)

    # 1) POLARIS train
    polaris_path = out_train_dir / "polaris.parquet"
    if polaris_path.exists():
        print(f"POLARIS train already exists at: {polaris_path}, skipping download.")
    else:
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
        polaris_std.to_parquet(str(polaris_path))
        print(f"Saved POLARIS train to: {polaris_path} ({len(polaris_std)} rows)")

    # 2) GSM8K train
    gsm8k_train_path = out_train_dir / "gsm8k.parquet"
    if gsm8k_train_path.exists():
        print(f"GSM8K train already exists at: {gsm8k_train_path}, skipping download.")
    else:
        print("Loading GSM8K train ...")
        conf = DATASETS["GSM8K"]
        path = conf["path"]
        subset = conf.get("subset", None)
        ds_dict = datasets.load_dataset(path, subset) if subset else datasets.load_dataset(path)
        if "train" in ds_dict:
            ds = ds_dict["train"]
            print(f"Using split for GSM8K: train")
            ds_std = standardize_hf_dataset(ds, conf["q_keys"], conf["a_keys"])
            ds_std.to_parquet(str(gsm8k_train_path))
            print(f"Saved GSM8K train to: {gsm8k_train_path} ({len(ds_std)} rows)")
        else:
            print("Warning: GSM8K train split not found")

    # 3) Test sets: AIME24, AIME25, OlympiadBench, GSM8K
    for benchmark in ["AIME24", "AIME25", "OlympiadBench", "GSM8K"]:
        out_path = out_test_dir / f"{benchmark}.parquet"
        if out_path.exists():
            print(f"{benchmark} test already exists at: {out_path}, skipping download.")
            continue
        
        conf = DATASETS[benchmark]
        path = conf["path"]
        subset = conf.get("subset", None)
        print(f"Loading {path}" + (f" (subset={subset})" if subset else "") + " ...")
        ds_dict = datasets.load_dataset(path, subset) if subset else datasets.load_dataset(path)
        
        if benchmark == "GSM8K":
            split = "test" if "test" in ds_dict else choose_available_split(ds_dict, conf["prefer_splits"])
        else:
            split = choose_available_split(ds_dict, conf["prefer_splits"])
        
        ds = ds_dict[split]
        print(f"Using split for {benchmark}: {split} (as test)")
        ds_std = standardize_hf_dataset(ds, conf["q_keys"], conf["a_keys"])
        ds_std.to_parquet(str(out_path))
        print(f"Saved {benchmark} test to: {out_path} ({len(ds_std)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
