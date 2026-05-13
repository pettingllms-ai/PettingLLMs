"""Small MetaAgent-X demo runner.

This module is intentionally independent from the trainer/evaluator loop.  It
serves one demo path: ask a served MetaAgent-X model to design a MAS for a user
question, execute the generated workflow, and write inspectable artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from pettingllms.multi_agent_env.autoevol.gen_agent import MASGenerator


DEFAULT_QUESTION = (
    "Find the value of x if 2x + 3 = 17. Answer with a single number."
)


def _normalise_server_address(server_address: str) -> str:
    address = server_address.rstrip("/")
    if address.endswith("/v1"):
        address = address[:-3]
    if address.startswith("http://") or address.startswith("https://"):
        return address
    return f"http://{address}"


def _openai_base_url(server_address: str) -> str:
    return _normalise_server_address(server_address) + "/v1"


def _request_design(
    server_address: str,
    model_name: str,
    question: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> str:
    client = OpenAI(api_key=api_key or "dummy", base_url=_openai_base_url(server_address))
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer specializing in "
                    "multi-agent workflow systems."
                ),
            },
            {
                "role": "user",
                "content": "Design Multi Agent System for the Question: " + question,
            },
        ],
    )
    return response.choices[0].message.content or ""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _build_executable_mas(
    generator: MASGenerator,
    server_address: str,
    model_name: str,
    tokenizer_path: str,
    max_prompt_length: int,
    max_response_length: int,
    enable_thinking: bool,
    task_type: str,
) -> str:
    repo_root = _repo_root()
    autoevol_root = repo_root / "pettingllms" / "multi_agent_env" / "autoevol"
    api_base = _openai_base_url(server_address)

    setup_code = f'''
import logging
import os
import sys
import warnings

repo_root = {str(repo_root)!r}
autoevol_root = {str(autoevol_root)!r}
sys.path.insert(0, repo_root)
sys.path.insert(0, autoevol_root)

logging.getLogger("autogen").setLevel(logging.ERROR)
logging.getLogger("aiohttp").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=ResourceWarning)

os.environ.setdefault("TASK_TYPE", {task_type!r})
os.environ.setdefault("EVALUATE_MODE", "True")

from pettingllms.multi_agent_env.autoevol.utils.BaseOpenAI import AIClient

ai_client = AIClient(
    api_base={api_base!r},
    api_key="dummy",
    chat_model={model_name!r},
    max_answer_tokens={max_response_length},
    tokenizer_path=None,
    server_address=None,
    max_prompt_length={max_prompt_length},
    max_response_length={max_response_length},
    enable_thinking={enable_thinking!r},
    workflow=None,
)
'''

    patched_code = generator._patch_imports(generator.generated_code)
    patched_code = generator._patch_string_escapes(patched_code)
    patched_code = generator._patch_workflow_init(
        patched_code,
        server_address=server_address,
        enable_thinking=enable_thinking,
    )
    return setup_code + "\n" + patched_code + "\n"


def _parse_node_assignments(code: str) -> Dict[str, Tuple[str, str]]:
    node_pattern = re.compile(
        r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        r"(?P<class>AgentNode|EnsembleNode|ReflectionNode|DebateNode|RouterNode)"
        r"\s*\([^)]*?name\s*=\s*[\"'](?P<name>[^\"']+)[\"']",
        re.DOTALL,
    )
    return {
        match.group("var"): (match.group("name"), match.group("class"))
        for match in node_pattern.finditer(code)
    }


def _parse_workflow_order(code: str) -> List[str]:
    order: List[str] = []
    for match in re.finditer(r"workflow\.add_nodes\s*\(\s*\[(?P<body>[^\]]+)\]", code):
        order.extend(
            item.strip()
            for item in match.group("body").split(",")
            if item.strip()
        )
    for match in re.finditer(r"workflow\.add_node\s*\(\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)", code):
        order.append(match.group("var"))
    return order


def _mermaid_label(name: str, class_name: str) -> str:
    return f"{name}\\n{class_name}"


def _write_visualization(code: str, output_dir: Path) -> Path:
    assignments = _parse_node_assignments(code)
    order = _parse_workflow_order(code)
    if not order:
        order = list(assignments.keys())

    lines = ["flowchart TD", '  input["User question"]']
    previous = "input"
    used: List[str] = []
    for index, var_name in enumerate(order):
        if var_name not in assignments:
            continue
        display_name, class_name = assignments[var_name]
        node_id = f"node_{index}"
        lines.append(f'  {node_id}["{_mermaid_label(display_name, class_name)}"]')
        lines.append(f"  {previous} --> {node_id}")
        previous = node_id
        used.append(var_name)

    if not used and assignments:
        for index, (var_name, (display_name, class_name)) in enumerate(assignments.items()):
            node_id = f"node_{index}"
            lines.append(f'  {node_id}["{_mermaid_label(display_name, class_name)}"]')
            lines.append(f"  input --> {node_id}")
            previous = node_id

    lines.append('  output["Final answer"]')
    lines.append(f"  {previous} --> output")

    path = output_dir / "mas_visualization.mmd"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _extract_final_answer(output_text: str) -> str:
    patterns: Iterable[str] = (
        r"FINAL ANSWER:\s*(.+)",
        r"Final Answer:\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
    )
    for pattern in patterns:
        matches = re.findall(pattern, output_text)
        if matches:
            return matches[-1].strip()
    tail = output_text.strip().splitlines()
    return tail[-1].strip() if tail else ""


def run_demo(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    question = args.question or DEFAULT_QUESTION
    generator = MASGenerator(task_type=args.task_type)

    designer_response = _request_design(
        server_address=args.server_address,
        model_name=args.model_name,
        question=question,
        temperature=args.temperature,
        max_tokens=args.design_max_tokens,
        api_key=args.api_key,
    )
    generator.update_from_model(designer_response)

    (output_dir / "question.txt").write_text(question + "\n", encoding="utf-8")
    (output_dir / "designer_response.txt").write_text(
        designer_response,
        encoding="utf-8",
    )
    (output_dir / "mas_design.py").write_text(
        generator.generated_code,
        encoding="utf-8",
    )
    visualization_path = _write_visualization(generator.generated_code, output_dir)

    result: Dict[str, object] = {
        "question": question,
        "output_dir": str(output_dir),
        "designer_response": str(output_dir / "designer_response.txt"),
        "mas_design": str(output_dir / "mas_design.py"),
        "visualization": str(visualization_path),
        "execution_success": False,
    }

    if args.design_only:
        result["execution_skipped"] = True
        return result

    if not generator.generated_code.strip():
        result["error"] = "The model response did not contain executable MAS code."
        return result

    executable_code = _build_executable_mas(
        generator=generator,
        server_address=args.server_address,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path or args.model_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        enable_thinking=args.enable_thinking,
        task_type=args.task_type,
    )
    executable_path = output_dir / "mas.py"
    executable_path.write_text(executable_code, encoding="utf-8")

    completed = subprocess.run(
        [args.python_bin, str(executable_path)],
        cwd=str(_repo_root()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.execution_timeout,
        check=False,
    )
    output_text = completed.stdout or ""
    (output_dir / "execution.log").write_text(output_text, encoding="utf-8")

    result.update(
        {
            "mas_executable": str(executable_path),
            "execution_log": str(output_dir / "execution.log"),
            "execution_returncode": completed.returncode,
            "execution_success": completed.returncode == 0,
            "final_answer": _extract_final_answer(output_text),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one MetaAgent-X MAS demo.")
    parser.add_argument("--server-address", default="127.0.0.1:8300")
    parser.add_argument("--model-name", default="shared_model")
    parser.add_argument("--model-path", default="Mercury7353/MetaAgent-X")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--task-type", choices=["math", "code"], default="math")
    parser.add_argument("--output-dir", default="outputs/autoeval_demo")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--design-max-tokens", type=int, default=8192)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--execution-timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_demo(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
