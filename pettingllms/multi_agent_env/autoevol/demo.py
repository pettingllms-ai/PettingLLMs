"""Small MetaAgent-X demo runner.

This module is intentionally independent from the trainer/evaluator loop.  It
serves one demo path: ask a served MetaAgent-X model to design a MAS for a user
question, execute the generated workflow, and write inspectable artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from pettingllms.multi_agent_env.autoevol.gen_agent import MASGenerator


DEFAULT_QUESTION = (
    "Question: Every morning Aya goes for a $9$-kilometer-long walk and stops "
    "at a coffee shop afterwards. When she walks at a constant speed of $s$ "
    "kilometers per hour, the walk takes her 4 hours, including $t$ minutes "
    "spent in the coffee shop. When she walks $s+2$ kilometers per hour, the "
    "walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the "
    "coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. "
    "Find the number of minutes the walk takes her, including the $t$ minutes "
    "spent in the coffee shop."
)
DEFAULT_CODE_QUESTION = (
    "Question: LeetCode Hard - Trapping Rain Water. Given n non-negative "
    "integers representing an elevation map where the width of each bar is 1, "
    "compute how much water it can trap after raining. Write a Python function "
    "trap(height: List[int]) -> int."
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


def _format_demo_question(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return DEFAULT_QUESTION
    if text.lower().startswith("question:"):
        return text
    return "Question: " + text


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
    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", output_text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    solution_matches = re.findall(
        r"<solution>\s*(.*?)\s*</solution>",
        output_text,
        flags=re.DOTALL,
    )
    if solution_matches:
        return solution_matches[-1].strip()

    patterns: Iterable[str] = (
        r"FINAL ANSWER:\s*(.+)",
        r"Final Answer:\s*(.+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, output_text)
        if matches:
            candidate = matches[-1].strip()
            if candidate and not candidate.startswith("="):
                return candidate
    tail = output_text.strip().splitlines()
    return tail[-1].strip() if tail else ""


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_agent_trace(output_text: str) -> List[Dict[str, str]]:
    pattern = re.compile(
        r"=+ AGENT NODE: (?P<name>.*?) =+\n"
        r"(?P<body>.*?)(?=\n=+ AGENT NODE: |\Z)",
        re.DOTALL,
    )
    trace = []
    for match in pattern.finditer(output_text):
        body = match.group("body").strip()
        response_match = re.search(
            r"\[AGENT RESPONSE\]:\s*(?P<response>.*?)(?:\n\[TOKENS\]:|\Z)",
            body,
            re.DOTALL,
        )
        trace.append(
            {
                "name": match.group("name").strip(),
                "body": body,
                "response": response_match.group("response").strip()
                if response_match
                else "",
            }
        )
    return trace


def _workflow_steps(code: str) -> List[Tuple[str, str]]:
    assignments = _parse_node_assignments(code)
    order = _parse_workflow_order(code)
    steps: List[Tuple[str, str]] = []
    for var_name in order:
        if var_name in assignments:
            steps.append(assignments[var_name])
    if not steps:
        steps = list(assignments.values())
    return steps


def _render_cards(items: List[str]) -> str:
    return "\n".join(f"<div class=\"flow-card\">{html.escape(item)}</div>" for item in items)


def _write_demo_ui(
    output_dir: Path,
    result: Dict[str, object],
    task_type: str,
    question: str,
    designer_response: str,
    mas_design: str,
    execution_log: str,
    visualization_source: str,
) -> Path:
    examples = {
        "math": DEFAULT_QUESTION,
        "code": DEFAULT_CODE_QUESTION,
    }
    attempts = sorted(output_dir.glob("designer_response_attempt_*.txt"))
    attempt_items = "\n".join(
        f"<button class=\"attempt-tab\" data-attempt=\"attempt-{idx}\">Attempt {idx}</button>"
        for idx, _ in enumerate(attempts, start=1)
    )
    attempt_panels = "\n".join(
        (
            f"<pre class=\"attempt-panel\" id=\"attempt-{idx}\">"
            f"{html.escape(_read_text(path))}</pre>"
        )
        for idx, path in enumerate(attempts, start=1)
    )
    if not attempt_items:
        attempt_items = "<span class=\"muted\">No retry attempts recorded.</span>"
        attempt_panels = ""

    trace = _extract_agent_trace(execution_log)
    trace_cards = "\n".join(
        (
            "<article class=\"trace-card\">"
            f"<h4>{html.escape(item['name'])}</h4>"
            f"<pre>{html.escape(item['response'] or item['body'])}</pre>"
            "</article>"
        )
        for item in trace
    )
    if not trace_cards:
        trace_cards = "<div class=\"empty\">No AgentNode execution trace captured.</div>"

    steps = _workflow_steps(mas_design)
    flow_items = ["User query"] + [
        f"{name} ({class_name})" for name, class_name in steps
    ] + ["Final answer"]
    flow_cards = _render_cards(flow_items)

    final_answer = str(result.get("final_answer", ""))
    status = "success" if result.get("execution_success") else "not complete"

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MetaAgent-X Demo</title>
  <style>
    :root {{
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #657282;
      --line: #d7dde5;
      --accent: #0f766e;
      --accent-soft: #e6f4f1;
      --code-bg: #111827;
      --code-ink: #e5e7eb;
      --warn: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 20px 28px;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    .subhead {{ color: var(--muted); margin-top: 4px; }}
    main {{
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
    }}
    aside, section, article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    aside {{ padding: 16px; align-self: start; position: sticky; top: 92px; }}
    .example {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 12px;
      margin-top: 10px;
      cursor: pointer;
      color: var(--ink);
    }}
    .example.active {{ border-color: var(--accent); background: var(--accent-soft); }}
    .example span {{ display: block; color: var(--muted); margin-top: 4px; font-size: 13px; }}
    .grid {{ display: grid; gap: 18px; }}
    section {{ padding: 16px; min-width: 0; }}
    h2 {{ font-size: 16px; margin: 0 0 12px; }}
    h3 {{ font-size: 14px; margin: 18px 0 8px; }}
    h4 {{ font-size: 13px; margin: 0 0 8px; color: var(--accent); }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .metric {{
      background: #f9fafb;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .metric strong {{ display: block; font-size: 13px; }}
    .metric span {{ color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }}
    pre {{
      margin: 0;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border-radius: 8px;
      padding: 12px;
      background: #f9fafb;
      border: 1px solid var(--line);
      max-height: 520px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .code {{
      background: var(--code-bg);
      color: var(--code-ink);
      border-color: #1f2937;
    }}
    .flow {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }}
    .flow-card {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 13px;
      max-width: 220px;
    }}
    .flow-card:not(:last-child)::after {{
      content: ">";
      color: var(--muted);
      margin-left: 8px;
    }}
    .trace-list {{ display: grid; gap: 10px; }}
    .trace-card {{ padding: 12px; }}
    .attempts {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }}
    .attempt-tab {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
    }}
    .attempt-tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
    .attempt-panel {{ display: none; }}
    .attempt-panel.active {{ display: block; }}
    .empty {{ color: var(--warn); background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 10px; }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
      .meta {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MetaAgent-X Demo</h1>
    <div class="subhead">MAS design, workflow execution, and final output for a single query.</div>
  </header>
  <main>
    <aside>
      <h2>Example Inputs</h2>
      <button class="example {'active' if task_type == 'math' else ''}" data-kind="math">
        Math
        <span>{html.escape(examples['math'])}</span>
      </button>
      <button class="example {'active' if task_type == 'code' else ''}" data-kind="code">
        Code
        <span>{html.escape(examples['code'])}</span>
      </button>
      <h3>Current Query</h3>
      <pre>{html.escape(question)}</pre>
    </aside>
    <div class="grid">
      <section>
        <h2>Run Summary</h2>
        <div class="meta">
          <div class="metric"><strong>Task</strong><span>{html.escape(task_type)}</span></div>
          <div class="metric"><strong>Status</strong><span>{html.escape(status)}</span></div>
          <div class="metric"><strong>Output</strong><span>{html.escape(str(output_dir))}</span></div>
        </div>
        <h3>Final Result</h3>
        <pre>{html.escape(final_answer or "No final answer extracted.")}</pre>
      </section>

      <section>
        <h2>MAS Workflow</h2>
        <div class="flow">{flow_cards}</div>
        <h3>Mermaid Source</h3>
        <pre>{html.escape(visualization_source)}</pre>
      </section>

      <section>
        <h2>MAS Design Code</h2>
        <pre class="code">{html.escape(mas_design or "No MAS design code was extracted.")}</pre>
      </section>

      <section>
        <h2>Designer Output Attempts</h2>
        <div class="attempts">{attempt_items}</div>
        {attempt_panels}
        <h3>Selected Designer Response</h3>
        <pre>{html.escape(designer_response)}</pre>
      </section>

      <section>
        <h2>Execution Trace</h2>
        <div class="trace-list">{trace_cards}</div>
      </section>

      <section>
        <h2>Full Execution Log</h2>
        <pre>{html.escape(execution_log or "No execution log was produced.")}</pre>
      </section>
    </div>
  </main>
  <script>
    const tabs = Array.from(document.querySelectorAll(".attempt-tab"));
    const panels = Array.from(document.querySelectorAll(".attempt-panel"));
    function activateAttempt(id) {{
      tabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.attempt === id));
      panels.forEach((panel) => panel.classList.toggle("active", panel.id === id));
    }}
    if (tabs.length) activateAttempt(tabs[0].dataset.attempt);
    tabs.forEach((tab) => tab.addEventListener("click", () => activateAttempt(tab.dataset.attempt)));
  </script>
</body>
</html>
"""
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def run_demo(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    question = _format_demo_question(args.question or DEFAULT_QUESTION)
    generator = MASGenerator(task_type=args.task_type)

    designer_response = ""
    for attempt in range(1, args.design_retries + 1):
        designer_response = _request_design(
            server_address=args.server_address,
            model_name=args.model_name,
            question=question,
            temperature=args.temperature,
            max_tokens=args.design_max_tokens,
            api_key=args.api_key,
        )
        (output_dir / f"designer_response_attempt_{attempt}.txt").write_text(
            designer_response,
            encoding="utf-8",
        )
        generator.update_from_model(designer_response)
        if generator.generated_code.strip():
            break

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
        "task_type": args.task_type,
        "output_dir": str(output_dir),
        "designer_response": str(output_dir / "designer_response.txt"),
        "mas_design": str(output_dir / "mas_design.py"),
        "visualization": str(visualization_path),
        "execution_success": False,
    }

    if args.design_only:
        result["execution_skipped"] = True
        ui_path = _write_demo_ui(
            output_dir=output_dir,
            result=result,
            task_type=args.task_type,
            question=question,
            designer_response=designer_response,
            mas_design=generator.generated_code,
            execution_log="",
            visualization_source=_read_text(visualization_path),
        )
        result["ui"] = str(ui_path)
        return result

    if not generator.generated_code.strip():
        result["error"] = "The model response did not contain executable MAS code."
        ui_path = _write_demo_ui(
            output_dir=output_dir,
            result=result,
            task_type=args.task_type,
            question=question,
            designer_response=designer_response,
            mas_design=generator.generated_code,
            execution_log="",
            visualization_source=_read_text(visualization_path),
        )
        result["ui"] = str(ui_path)
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
    ui_path = _write_demo_ui(
        output_dir=output_dir,
        result=result,
        task_type=args.task_type,
        question=question,
        designer_response=designer_response,
        mas_design=generator.generated_code,
        execution_log=output_text,
        visualization_source=_read_text(visualization_path),
    )
    result["ui"] = str(ui_path)
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
    parser.add_argument("--design-retries", type=int, default=3)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--execution-timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_demo(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
