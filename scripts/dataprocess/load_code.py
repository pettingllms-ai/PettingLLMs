# pip install datasets pandas pyarrow
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import os
import sys
import re, ast, json, zlib, pickle, base64
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


sys.set_int_max_str_digits(0) 



def _find_matching_bracket(s: str, start: int) -> Optional[int]:
    depth = 0
    for i in range(start, len(s)):
        if s[i] == '[':
            depth += 1
        elif s[i] == ']':
            depth -= 1
            if depth == 0:
                return i
    return None

def transform_tokens(s: str) -> str:
    events = []
    masked = s

    start2 = s.find('[[')
    if start2 != -1:
        end2 = _find_matching_bracket(s, start2)
        if end2 is not None:
            arr_lit = s[start2:end2+1]
            try:
                arr2d = ast.literal_eval(arr_lit)
            except Exception:
                arr2d = []
            events.append((start2, 'array2d', arr2d))
            masked = masked[:start2] + ' '*(end2+1 - start2) + masked[end2+1:]

    token_re = re.compile(r'\[[^\]]*\]|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\S+')
    for m in token_re.finditer(masked):
        tok = m.group(0); pos = m.start()
        if tok.startswith('[') and tok.endswith(']'):
            try:
                raw = ast.literal_eval(tok)
                parts = [str(x) for x in raw]
            except Exception:
                parts = [x.strip() for x in tok.strip('[]').split(',') if x.strip()]
            events.append((pos, 'array1d', parts))
        elif (tok.startswith('"') and tok.endswith('"')) or (tok.startswith("'") and tok.endswith("'")):
            events.append((pos, 'scalar', tok[1:-1]))
        else:
            events.append((pos, 'scalar', tok))

    events.sort(key=lambda e: e[0])

    out = []
    for _, typ, data in events:
        if typ == 'scalar':
            out.append(str(data))
        elif typ == 'array1d':
            out.append(" ".join(data))
        else:
            for row in data:
                out.append(" ".join(map(str, row)))
    return "\n".join(out) + "\n"

def transform_input_block(spec: str) -> str:
    events: List[Tuple[int, str, object]] = []
    token_re = re.compile(
        r"""
        (?P<kv_array>      \b\w+\s*=\s*\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])
      | (?P<kv_scalar>     \b\w+\s*=\s*(?: "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | True | False | -?\d+(?:\.\d+)? | \w+))
      | (?P<array>         \[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])
      | (?P<scalar>        "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' | True | False | -?\d+(?:\.\d+)? | \w+)
        """,
        re.X,
    )

    for m in token_re.finditer(spec):
        if m.group("kv_array"):
            lit = m.group("kv_array").split("=", 1)[1].lstrip()
            try: arr = ast.literal_eval(lit)
            except Exception: arr = []
            events.append((m.start(), "array", arr)); continue
        if m.group("kv_scalar"):
            val = m.group("kv_scalar").split("=", 1)[1].lstrip()
            if val and val[0] in "\"'":
                val = val[1:-1]
            events.append((m.start(), "scalar", val)); continue
        if m.group("array"):
            try: arr = ast.literal_eval(m.group("array"))
            except Exception: arr = []
            events.append((m.start(), "array", arr)); continue
        if m.group("scalar"):
            tok = m.group("scalar")
            if tok and tok[0] in "\"'":
                tok = tok[1:-1]
            events.append((m.start(), "scalar", tok))

    events.sort(key=lambda e: e[0])
    lines: List[str] = []
    for _, kind, val in events:
        if kind == "scalar":
            lines.append(str(val))
        else:
            if isinstance(val, list) and val and all(isinstance(r, list) for r in val):
                lines.extend(" ".join(map(str, r)) for r in val)
            else:
                lines.append(" ".join(map(str, val)))
    return "\n".join(lines) + "\n"

def replace_input_block(text: str) -> str:
    def _repl(m):
        return f"{m.group(1)}\n{transform_input_block(m.group(2))}"
    pattern = re.compile(r'(Input\s*:\s*)(.*?)(?=\s*(?:Output\s*:|$))', flags=re.I|re.S)
    return pattern.sub(_repl, text)

def replace_output_block(text: str) -> str:
    def strip_quotes(tok: str) -> str:
        return tok[1:-1] if len(tok) >= 2 and tok[0] in "\"'" and tok[-1] == tok[0] else tok
    out, last = [], 0
    for m in re.finditer(r'Output\s*:', text):
        out.append(text[last:m.end()]); i = m.end()
        while i < len(text) and text[i].isspace():
            out.append(text[i]); i += 1
        if i >= len(text): break
        if text[i] == '[':
            start, end = i, _find_matching_bracket(text, i)
            literal = text[start:end+1] if end is not None else "[]"
            try: arr = ast.literal_eval(literal)
            except Exception: arr = []
            lines = ([" ".join(map(str, r)) for r in arr]
                     if arr and all(isinstance(r, list) for r in arr)
                     else [" ".join(map(str, arr))])
            out.append("\n" + "\n".join(lines) + "\n")
            last = (end or i) + 1
        else:
            m2 = re.match(r'(-?\d+|True|False|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', text[i:])
            if m2:
                out.append(f"\n{strip_quotes(m2.group(0))}\n")
                last = i + len(m2.group(0))
            else:
                last = i
    out.append(text[last:])
    return "".join(out)

def _normalize_cell(s: str) -> str:
    s = (s or "").strip().replace("\\n", "\n")
    if not s.endswith("\n"):
        s += "\n"
    return s

def _clean_solution(sol: Any) -> str:
    """Remove triple quotes/markdown code fences and ensure string format"""
    if sol is None:
        return ""
    s = str(sol).strip()
    # Remove ```python ... ``` or ``` ... ```
    s = re.sub(r"^```(?:\w+)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _filter_nonempty_io(df: pd.DataFrame) -> pd.DataFrame:
    """Only keep samples where both test_input and test_output are non-empty"""
    def _len_list(x):
        try:
            return len(x)
        except Exception:
            return 0
    mask = (df["test_input"].apply(_len_list) > 0) & (df["test_output"].apply(_len_list) > 0)
    return df.loc[mask].reset_index(drop=True)



_ASSERT_PATTERNS: List[re.Pattern] = [
    re.compile(r'^\s*assert\s+(?P<call>\w+\s*\(.*\))\s*==\s*(?P<exp>.+?)\s*$', re.S),
    re.compile(r'^\s*assert\s+(?P<exp>.+?)\s*==\s*(?P<call>\w+\s*\(.*\))\s*$', re.S),
    re.compile(r'^\s*(?:assert\s+)?(?:check|check_equal|check_solution)\s*\(\s*(?P<call>\w+\s*\(.*\))\s*,\s*(?P<exp>.+?)\s*\)\s*$', re.S),
    re.compile(r'^\s*(?:assert\s+)?(?:check|check_equal|check_solution)\s*\(\s*(?P<exp>.+?)\s*,\s*(?P<call>\w+\s*\(.*\))\s*\)\s*$', re.S),
]

def _strip_trailing_comment(s: str) -> str:
    return re.split(r'#(?![^\'"]*["\'])', s, maxsplit=1)[0].strip()

def _extract_args_from_call(call: str, prefer_fn: Optional[str] = None) -> Optional[str]:
    m = re.match(r'(?P<fn>\w+)\s*\((?P<args>.*)\)\s*$', call.strip(), re.S)
    if not m:
        return None
    # If prefer_fn is provided and does not match, still accept (some wrapper internal forwarding)
    return m.group("args")

def parse_asserts_to_io(lines: List[str], prefer_fn: Optional[str] = None) -> Tuple[List[str], List[str]]:
    ins, outs = [], []
    for raw in lines:
        line = _strip_trailing_comment(raw)
        if not line:
            continue
        matched = None
        for pat in _ASSERT_PATTERNS:
            m = pat.match(line)
            if m:
                matched = m; break
        if not matched:
            continue
        call = matched.group("call")
        exp  = matched.group("exp")
        args = _extract_args_from_call(call, prefer_fn=prefer_fn)
        if args is None:
            continue
        in_text  = _normalize_cell(transform_tokens(args))
        out_text = _normalize_cell(transform_tokens(exp))
        ins.append(in_text); outs.append(out_text)
    return ins, outs

# ============================================================
# CodeContests
# ============================================================

PY_LANG_IDS = {1, 3}  # deepmind/code_contests: 1=PYTHON(2), 3=PYTHON3

def _pick_first_py(solutions: Dict[str, Any], k: int = 1) -> List[str]:
    langs = solutions.get("language") or []
    codes = solutions.get("solution") or []
    out = []
    for lang, code in zip(langs, codes):
        if lang in PY_LANG_IDS:
            out.append(code)
            if len(out) >= k:
                break
    return out

def _has_py(solutions: Dict[str, Any]) -> bool:
    return any((l in PY_LANG_IDS) for l in (solutions.get("language") or []))


import re
import ast

def _node_to_text(n: ast.AST) -> str:
   
    if isinstance(n, ast.Constant) and isinstance(n.value, str):
        return n.value
    # Other literals: int/float/bool/None
    if isinstance(n, ast.Constant):
        return repr(n.value) if not isinstance(n.value, (int, float, bool, type(None))) else str(n.value)
    # Special handling for matrices (2D lists)
    if isinstance(n, ast.List):
        # Check if it is matrix form (all elements are lists)
        if all(isinstance(elem, ast.List) for elem in n.elts):
            # This is a matrix, keep original format
            return ast.unparse(n)
        else:
            # Regular list, also keep original format
            return ast.unparse(n)
    # Lists/dicts/tuples/expressions: reverse generate source code
    return ast.unparse(n)

def convert_asserts_to_io(assert_lines: List[str]) -> Dict[str, List[str]]:
    results = {"test_input": [], "test_output": []}
    for line in assert_lines:
        s = line.strip()
        if not s or not s.startswith("assert"):
            continue
        m = re.match(r"assert\s+(.+?)\s*==\s*(.+)$", s)
        if not m:
            continue
        call_expr, expected_expr = m.groups()

        # Parse function call
        call_tree = ast.parse(call_expr, mode="eval")
        if not isinstance(call_tree.body, ast.Call):
            continue
        call = call_tree.body

        # Input: process parameters, pay special attention to matrix format
        arg_texts = []
        for a in call.args:
            text = _node_to_text(a)
            # If this is a matrix (contains brackets and commas), keep as single line
            if text.startswith('[') and text.endswith(']') and ',' in text:
                arg_texts.append(text)
            else:
                arg_texts.append(text)
        
        # If there is only one parameter and it looks like matrix format, use directly
        if len(arg_texts) == 1 and arg_texts[0].startswith('[') and arg_texts[0].endswith(']'):
            input_str = arg_texts[0] + "\n"
        else:
            input_str = "\n".join(arg_texts) + "\n"

        # Expected output
        expected_node = ast.parse(expected_expr, mode="eval").body
        output_str = _node_to_text(expected_node) + "\n"

        results["test_input"].append(input_str)
        results["test_output"].append(output_str)
    return results



def process_code_contests(split: str) -> pd.DataFrame:
    print(f"Loading deepmind/code_contests split={split} ...")
    ds = load_dataset("deepmind/code_contests", split=split)
    rows = []
    for ex in ds:

        tests = (ex.get("public_tests") or {}) if split == "test" else (ex.get("private_tests") or {})
        test_in = tests.get("input") or []
        test_out = tests.get("output") or []
        if not (isinstance(test_in, list) and test_in):
            continue
        solutions = ex.get("solutions") or {}
        if not _has_py(solutions):
            solution = ""
        else:
            py = _pick_first_py(solutions, k=1)
            solution = _clean_solution(py[0]) if py else ""
        rows.append({
            "question": (ex.get("description") or "").strip(),
            "solution": solution,
            "test_input": test_in,
            "test_output": test_out,
        })
    df = pd.DataFrame(rows, columns=["question","test_input","test_output","solution"])
    df = _filter_nonempty_io(df)
    print(f"Completed code_contests/{split}: {len(df)}")
    return df

def process_apps_test() -> pd.DataFrame:
    print("Loading apps ...")
    ds = load_dataset(
        "json",
        data_files={"test": "hf://datasets/codeparrot/apps/test.jsonl"},
        split="test",
    )
    ds = list(ds)
    rows = []
    
    for ex in ds[:500]:
        # Parse solutions and input_output fields
        try:
            solutions = json.loads(ex.get("solutions", "[]"))
            input_output = json.loads(ex.get("input_output", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
            
        # Get problem description
        question = (ex.get("question") or "").strip()
        if not question:
            continue
            
        # Process solutions
        if not solutions:
            solution = ""
        else:
            # apps dataset solutions is a list of strings, take the first one
            solution = _clean_solution(solutions[0]) if solutions else ""
            
        # Process test input and output
        test_input = []
        test_output = []
        
        if input_output:
            inputs = input_output.get("inputs", [])
            outputs = input_output.get("outputs", [])
            
            if isinstance(inputs, list) and isinstance(outputs, list):
                test_input = [_normalize_cell(str(x)) for x in inputs]
                test_output = [_normalize_cell(str(x)) for x in outputs]
        
        # Only keep samples with test cases
        if not (test_input and test_output):
            continue
            
        rows.append({
            "question": question,
            "solution": solution,
            "test_input": test_input,
            "test_output": test_output,
        })
    
    df = pd.DataFrame(rows, columns=["question", "test_input", "test_output", "solution"])
    df = _filter_nonempty_io(df)
    print(f"Completed apps: {len(df)}")
    return df

def process_apps_train() -> pd.DataFrame:
    print("Loading apps ...")
    ds = load_dataset(
        "json",
        data_files={"test": "hf://datasets/codeparrot/apps/test.jsonl"},
        split="test",
    )
    ds = list(ds)
    rows = []
    
    for ex in ds[500:4500]:
        # Parse solutions and input_output fields
        try:
            solutions = json.loads(ex.get("solutions", "[]"))
            input_output = json.loads(ex.get("input_output", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
            
        # Get problem description
        question = (ex.get("question") or "").strip()
        if not question:
            continue
            
        # Process solutions
        if not solutions:
            solution = ""
        else:
            # apps dataset solutions is a list of strings, take the first one
            solution = _clean_solution(solutions[0]) if solutions else ""
            
        # Process test input and output
        test_input = []
        test_output = []
        
        if input_output:
            inputs = input_output.get("inputs", [])
            outputs = input_output.get("outputs", [])
            
            if isinstance(inputs, list) and isinstance(outputs, list):
                test_input = [_normalize_cell(str(x)) for x in inputs]
                test_output = [_normalize_cell(str(x)) for x in outputs]
        
        # Only keep samples with test cases
        if not (test_input and test_output):
            continue
            
        rows.append({
            "question": question,
            "solution": solution,
            "test_input": test_input,
            "test_output": test_output,
        })
    
    df = pd.DataFrame(rows, columns=["question", "test_input", "test_output", "solution"])
    df = _filter_nonempty_io(df)
    print(f"Completed apps: {len(df)}")
    return df


# ============================================================
# LiveCodeBench (using code_generation_lite v6)
# ============================================================

class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"

@dataclass
class _LCB_Test:
    input: str
    output: str
    testtype: TestType
    def __post_init__(self):
        self.testtype = TestType(self.testtype)

@dataclass
class _LCB_Problem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[_LCB_Test]
    private_test_cases: list[_LCB_Test]
    metadata: dict
    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)
        pts = json.loads(self.public_test_cases)
        self.public_test_cases = [_LCB_Test(**t) for t in pts]
        try:
            pr = json.loads(self.private_test_cases)
        except Exception:
            pr = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))
                    )
                )
            )
        self.private_test_cases = [_LCB_Test(**t) for t in pr]
        self.metadata = json.loads(self.metadata)

def _load_lcb_lite_v6() -> pd.DataFrame:
    HF_PREFIX = "hf://datasets/livecodebench/code_generation_lite/"
    V6_FILES = [f"{HF_PREFIX}test6.jsonl"]
    ds = load_dataset("json", data_files=V6_FILES, split="train")
    rows = []
    for ex in ds:
        title = ex.get("question_title") or ""
        content = ex.get("question_content") or ""
        question = (title + ("\n\n" if title and content else "") + content).strip()

        def _parse_simple_tests(raw):
            if not raw:
                return [], []
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(data, dict) and "input" in data and "output" in data:
                    ins = data.get("input") or []
                    outs = data.get("output") or []
                    ins = ins if isinstance(ins, list) else [ins]
                    outs = outs if isinstance(outs, list) else [outs]
                    return [str(x) for x in ins], [str(x) for x in outs]
                elif isinstance(data, list):
                    ins, outs = [], []
                    for item in data:
                        if isinstance(item, dict):
                            ins.append(str(item.get("input", "")))
                            outs.append(str(item.get("output", "")))
                    return ins, outs
            except Exception:
                pass
            return [str(raw)], [""]

        pub_raw = ex.get("public_test_cases") or ""
        pri_raw = ex.get("private_test_cases") or ""
        pub_in, pub_out = _parse_simple_tests(pub_raw)
        pri_in, pri_out = _parse_simple_tests(pri_raw)
        test_input = pub_in if pub_in else pri_in
        test_output = pub_out if pub_out else pri_out

        rows.append({
            "question": question,
            "solution": "", 
            "test_input": [ _normalize_cell(x) for x in test_input ],
            "test_output": [ _normalize_cell(x) for x in test_output ],
        })

    df = pd.DataFrame(rows, columns=["question", "solution", "test_input", "test_output"])
    df = _filter_nonempty_io(df)
    print(f"LCB v6 loaded: {len(df)}")
    return df

def process_livecodebench() -> pd.DataFrame:
    print(f"Loading LiveCodeBench v6 ...")
    df = _load_lcb_lite_v6()
    print(f"Completed livecodebench v6: {len(df)}")
    return df



def main():
    project_root = Path(__file__).resolve().parents[2]
    
    # Create train and test directories
    train_dir = project_root / "datasets" / "code" / "train"
    test_dir = project_root / "datasets" / "code" / "test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"train_dir: {train_dir}")
    print(f"test_dir: {test_dir}")

    # ============================================================
    # TRAIN dataset
    # ============================================================
    
    # 1. CodeContests(train)
    df_cc_train = process_code_contests(split="train")
    (train_dir / "code_contests_train.parquet").unlink(missing_ok=True)
    df_cc_train.to_parquet(train_dir / "code_contests_train.parquet", index=False)
    print(f"Saved: {train_dir / 'code_contests_train.parquet'}")

    # 2. Apps(train)
    df_apps_train = process_apps_train()
    (train_dir / "apps_train.parquet").unlink(missing_ok=True)
    df_apps_train.to_parquet(train_dir / "apps_train.parquet", index=False)
    print(f"Saved: {train_dir / 'apps_train.parquet'}")
    
    # ============================================================
    # TEST dataset
    # ============================================================

    # 1. CodeContests(test)
    df_cc_test = process_code_contests(split="test")
    (test_dir / "code_contests_test.parquet").unlink(missing_ok=True)
    df_cc_test.to_parquet(test_dir / "code_contests_test.parquet", index=False)
    print(f"Saved: {test_dir / 'code_contests_test.parquet'}")

    # 2. Apps(test)
    df_apps_test = process_apps_test()
    (test_dir / "apps_test.parquet").unlink(missing_ok=True)
    df_apps_test.to_parquet(test_dir / "apps_test.parquet", index=False)
    print(f"Saved: {test_dir / 'apps_test.parquet'}")

    # 3. LiveCodeBench v6
    df_lcb = process_livecodebench()
    (test_dir / "livecodebench.parquet").unlink(missing_ok=True)
    df_lcb.to_parquet(test_dir / "livecodebench.parquet", index=False)
    print(f"Saved: {test_dir / 'livecodebench.parquet'}")

if __name__ == "__main__":
    main()
