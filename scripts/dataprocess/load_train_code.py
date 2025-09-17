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

# 设置更高的整数字符串转换限制，以处理非常大的测试输出
sys.set_int_max_str_digits(0)  # 0 表示无限制

# ============================================================
# 公共：把任意“参数/字面量说明”规约为 stdin 风格的多行文本
# ============================================================

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

    # 先抓第一段 [[...]]，避免内部再被拆
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
    """去掉三引号/markdown 代码围栏，确保字符串"""
    if sol is None:
        return ""
    s = str(sol).strip()
    # 去除 ```python ... ``` 或 ``` ... ```
    s = re.sub(r"^```(?:\w+)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _filter_nonempty_io(df: pd.DataFrame) -> pd.DataFrame:
    """只保留 test_input 和 test_output 都非空的样本"""
    def _len_list(x):
        try:
            return len(x)
        except Exception:
            return 0
    mask = (df["test_input"].apply(_len_list) > 0) & (df["test_output"].apply(_len_list) > 0)
    return df.loc[mask].reset_index(drop=True)

# ============================================================
# 断言 → (input, output) 抽取（MBPP / HumanEval 用）
# ============================================================

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
    # 若 prefer_fn 提供且不匹配，仍接受（一些包装器内部转发）
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

def _parse_cc_difficulty(ex: Dict[str, Any]) -> Optional[int]:
    """尽量从样本中解析出难度为整数。
    返回 None 表示无法确定难度。
    可识别：数值/数字字符串/"easy|medium|hard"。
    """
    val: Any = None
    # 常见直出字段
    for key in ("difficulty", "difficulty_level", "level"):
        if key in ex and ex.get(key) is not None:
            val = ex.get(key)
            break
    # 可能存在于 metadata
    if val is None:
        meta = ex.get("metadata")
        if isinstance(meta, dict):
            for key in ("difficulty", "difficulty_level", "level"):
                if key in meta and meta.get(key) is not None:
                    val = meta.get(key)
                    break

    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s.isdigit():
            return int(s)
        mapping = {"easy": 1, "e": 1, "medium": 2, "m": 2, "hard": 3, "h": 3}
        return mapping.get(s)
    return None


import re
import ast

def _node_to_text(n: ast.AST) -> str:
    # 字符串：直接取值，不带引号
    if isinstance(n, ast.Constant) and isinstance(n.value, str):
        return n.value
    # 其他字面量：int/float/bool/None
    if isinstance(n, ast.Constant):
        return repr(n.value) if not isinstance(n.value, (int, float, bool, type(None))) else str(n.value)
    # 特殊处理矩阵（二维列表）
    if isinstance(n, ast.List):
        # 检查是否是矩阵形式（所有元素都是列表）
        if all(isinstance(elem, ast.List) for elem in n.elts):
            # 这是一个矩阵，保持原格式
            return ast.unparse(n)
        else:
            # 普通列表，也保持原格式
            return ast.unparse(n)
    # 列表/字典/元组/表达式：反向生成源代码
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

        # 解析函数调用
        call_tree = ast.parse(call_expr, mode="eval")
        if not isinstance(call_tree.body, ast.Call):
            continue
        call = call_tree.body

        # 输入：处理参数，特别注意矩阵格式
        arg_texts = []
        for a in call.args:
            text = _node_to_text(a)
            # 如果这是一个矩阵（包含方括号和逗号），保持为单行
            if text.startswith('[') and text.endswith(']') and ',' in text:
                arg_texts.append(text)
            else:
                arg_texts.append(text)
        
        # 如果只有一个参数且看起来是矩阵格式，直接使用
        if len(arg_texts) == 1 and arg_texts[0].startswith('[') and arg_texts[0].endswith(']'):
            input_str = arg_texts[0] + "\n"
        else:
            input_str = "\n".join(arg_texts) + "\n"

        # 期望输出
        expected_node = ast.parse(expected_expr, mode="eval").body
        output_str = _node_to_text(expected_node) + "\n"

        results["test_input"].append(input_str)
        results["test_output"].append(output_str)
    return results



def process_code_contests(split: str) -> pd.DataFrame:
    print(f"🔄 加载 deepmind/code_contests split={split} ...")
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
    print(f"✅ code_contests/{split}: {len(df)}")
    return df

# ============================================================
# MBPP
# ============================================================

def process_mbpp() -> pd.DataFrame:
    print("🔄 加载 MBPP（优先 sanitized/test）...")
    ds = load_dataset("Gen-Verse/MBPP-ReasonFlux", split="test")
    rows = []
    for i,ex in enumerate(ds):
        question = (ex.get("text") or ex.get("prompt") or ex.get("description") or "").strip()
        # 修复：确保 solution 获取并清洗
        solution_raw = ex.get("code", None)
        if not solution_raw:
            solution_raw = ex.get("solution", "")
        solution = _clean_solution(solution_raw)
        results = convert_asserts_to_io(ex.get("test_list"))
        inputs = ex.get("test_input")
        outputs = ex.get("test_output")
        rows.append({
            "question": question,
            "solution": solution,
            "test_input": inputs,
            "test_output": outputs,
        })
        if i<5:
            print(question)
            print(solution)
            print(inputs)
            print(outputs)
    df = pd.DataFrame(rows, columns=["question","test_input","test_output","solution"])
    df = _filter_nonempty_io(df)
    print(f"✅ mbpp: {len(df)}")
    return df

# ============================================================
# HumanEval
# ============================================================
def process_apps_test() -> pd.DataFrame:
    print("🔄 加载 apps ...")
    ds = load_dataset(
        "json",
        data_files={"test": "hf://datasets/codeparrot/apps/test.jsonl"},
        split="test",
    )
    ds = list(ds)
    rows = []
    
    for ex in ds[:500]:
        # 解析 solutions 和 input_output 字段
        try:
            solutions = json.loads(ex.get("solutions", "[]"))
            input_output = json.loads(ex.get("input_output", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
            
        # 获取问题描述
        question = (ex.get("question") or "").strip()
        if not question:
            continue
            
        # 处理解决方案
        if not solutions:
            solution = ""
        else:
            # apps 数据集的 solutions 是字符串列表，直接取第一个
            solution = _clean_solution(solutions[0]) if solutions else ""
            
        # 处理测试输入输出
        test_input = []
        test_output = []
        
        if input_output:
            inputs = input_output.get("inputs", [])
            outputs = input_output.get("outputs", [])
            
            if isinstance(inputs, list) and isinstance(outputs, list):
                test_input = [_normalize_cell(str(x)) for x in inputs]
                test_output = [_normalize_cell(str(x)) for x in outputs]
        
        # 只保留有测试用例的样本
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
    print(f"✅ apps: {len(df)}")
    return df

def process_apps_train() -> pd.DataFrame:
    print("🔄 加载 apps ...")
    ds = load_dataset(
        "json",
        data_files={"train": "hf://datasets/codeparrot/apps/train.jsonl"},
        split="train",
    )
    ds = list(ds)
    rows = []
    
    for ex in ds:
        # 解析 solutions 和 input_output 字段
        try:
            solutions = json.loads(ex.get("solutions", "[]"))
            input_output = json.loads(ex.get("input_output", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
            
        # 获取问题描述
        question = (ex.get("question") or "").strip()
        if not question:
            continue
            
        # 处理解决方案
        if not solutions:
            solution = ""
        else:
            # apps 数据集的 solutions 是字符串列表，直接取第一个
            solution = _clean_solution(solutions[0]) if solutions else ""
            
        # 处理测试输入输出
        test_input = []
        test_output = []
        
        if input_output:
            inputs = input_output.get("inputs", [])
            outputs = input_output.get("outputs", [])
            
            if isinstance(inputs, list) and isinstance(outputs, list):
                test_input = [_normalize_cell(str(x)) for x in inputs]
                test_output = [_normalize_cell(str(x)) for x in outputs]
        
        # 只保留有测试用例的样本
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
    print(f"✅ apps: {len(df)}")
    return df

def process_humaneval() -> pd.DataFrame:
    print("🔄 加载 openai_humaneval/test ...")
    ds = load_dataset("openai_humaneval", split="test")
    rows = []
    for ex in ds:
        question = (ex.get("prompt") or "").strip()
        # 修复：优先 canonical_solution
        sol_raw = ex.get("canonical_solution", None)
        if not sol_raw:
            sol_raw = ex.get("solution", "")
        solution = _clean_solution(sol_raw)

        entry_point = ex.get("entry_point") or None
        test_str = ex.get("test") or ""
        test_lines = [ln for ln in str(test_str).splitlines() if ln.strip()]
        inputs, outputs = parse_asserts_to_io(test_lines, prefer_fn=entry_point)
        rows.append({
            "question": question,
            "solution": solution,
            "test_input": inputs,
            "test_output": outputs,
        })
    df = pd.DataFrame(rows, columns=["question","test_input","test_output","solution"])
    df = _filter_nonempty_io(df)
    print(f"✅ human_eval: {len(df)}")
    return df



# ============================================================
# LiveCodeBench（使用 code_generation_lite v6）
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
            "solution": "",  # LCB 无参考实现
            "test_input": [ _normalize_cell(x) for x in test_input ],
            "test_output": [ _normalize_cell(x) for x in test_output ],
        })

    df = pd.DataFrame(rows, columns=["question", "solution", "test_input", "test_output"])
    df = _filter_nonempty_io(df)
    print(f"LCB v6 loaded: {len(df)}")
    return df

def process_livecodebench() -> pd.DataFrame:
    print(f"🔄 加载 LiveCodeBench v6 ...")
    df = _load_lcb_lite_v6()
    print(f"✅ livecodebench v6: {len(df)}")
    return df

# ============================================================
# 主流程：写出 4 份 parquet（仅 4 列）
# ============================================================

def main():
    
    


    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "code" / "train"
    os.makedirs(out_dir, exist_ok=True)
    print(f"📁 输出目录: {out_dir}")

    df_mbpp = process_mbpp()
    (out_dir / "mbpp.parquet").unlink(missing_ok=True)
    df_mbpp.to_parquet(out_dir / "mbpp.parquet", index=False)
    print(f"💾 保存: {out_dir / 'mbpp.parquet'}")

    df_apps = process_apps_test()
    (out_dir / "apps.parquet").unlink(missing_ok=True)
    df_apps.to_parquet(out_dir / "apps.parquet", index=False)
    print(f"💾 保存: {out_dir / 'apps.parquet'}")

    # 1) CodeContests(train)
    df_train = process_code_contests(split="train")
    (out_dir / "train.parquet").unlink(missing_ok=True)
    df_train.to_parquet(out_dir / "train.parquet", index=False)
    print(f"💾 保存: {out_dir / 'train.parquet'}")

    # 2.1 CodeContests(test)
    df_cc_test = process_code_contests(split="test")
    (out_dir / "code_contests.parquet").unlink(missing_ok=True)
    df_cc_test.to_parquet(out_dir / "code_contests.parquet", index=False)
    print(f"💾 保存: {out_dir / 'code_contests.parquet'}")

    # 2.2 MBPP
    

    # 2.3 HumanEval
    df_he = process_humaneval()
    (out_dir / "human_eval.parquet").unlink(missing_ok=True)
    df_he.to_parquet(out_dir / "human_eval.parquet", index=False)
    print(f"💾 保存: {out_dir / 'human_eval.parquet'}")

    # 2.4 Apps
    

    # 2.4 LiveCodeBench v6
    df_lcb = process_livecodebench()
    (out_dir / "livecodebench.parquet").unlink(missing_ok=True)
    df_lcb.to_parquet(out_dir / "livecodebench.parquet", index=False)
    print(f"💾 保存: {out_dir / 'livecodebench.parquet'}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "datasets" / "code" / "train"
    os.makedirs(out_dir, exist_ok=True)
    df_apps_train = process_apps_train()
    df_apps_train.to_parquet(out_dir / "apps_train.parquet", index=False)
    print(f"💾 保存: {out_dir / 'apps_train.parquet'}")
