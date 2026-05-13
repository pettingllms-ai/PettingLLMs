"""Interactive MetaAgent-X demo server."""

from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from urllib.parse import urlparse

from pettingllms.multi_agent_env.autoevol.demo import (
    DEFAULT_CODE_QUESTION,
    DEFAULT_QUESTION,
    _extract_agent_trace,
    _read_text,
    run_demo,
)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MetaAgent-X Interactive Demo</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #0f766e;
      --accent-soft: #e8f5f2;
      --code-bg: #111827;
      --code-ink: #e5e7eb;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
    }
    h1 { margin: 0; font-size: 22px; }
    .status { color: var(--muted); font-size: 13px; }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      padding: 16px;
    }
    aside, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    aside { align-self: start; position: sticky; top: 16px; }
    label { display: block; margin: 12px 0 6px; font-size: 13px; font-weight: 650; }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 150px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button {
      border: 1px solid var(--accent);
      border-radius: 7px;
      background: var(--accent);
      color: white;
      padding: 10px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    button.secondary {
      background: #fff;
      color: var(--accent);
    }
    button:disabled { opacity: .6; cursor: not-allowed; }
    .examples { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
    .tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
      margin: -16px -16px 16px;
      padding: 10px 16px 0;
    }
    .tab {
      color: var(--ink);
      background: transparent;
      border: 1px solid transparent;
      border-bottom: 0;
      border-radius: 7px 7px 0 0;
      padding: 8px 10px;
    }
    .tab.active { background: var(--accent-soft); border-color: var(--line); color: var(--accent); }
    .panel { display: none; }
    .panel.active { display: block; }
    pre {
      margin: 0;
      max-height: 650px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f9fafb;
      padding: 12px;
      font-size: 12px;
      line-height: 1.45;
    }
    pre.code { background: var(--code-bg); color: var(--code-ink); border-color: #1f2937; }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 8px;
      padding: 10px;
      min-width: 0;
    }
    .metric strong { display: block; font-size: 12px; color: var(--muted); }
    .metric span { display: block; overflow-wrap: anywhere; margin-top: 3px; }
    .trace { display: grid; gap: 10px; }
    .trace-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
    .trace-card h3 { margin: 0 0 8px; font-size: 14px; color: var(--accent); }
    .empty { color: var(--muted); padding: 12px; border: 1px dashed var(--line); border-radius: 8px; }
    .error { color: var(--danger); background: #fff5f5; border: 1px solid #fecdca; padding: 10px; border-radius: 8px; margin-bottom: 12px; display: none; }
    .flow {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .flow-item {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 13px;
    }
    .flow-item:not(:last-child)::after { content: ">"; margin-left: 8px; color: var(--muted); }
    @media (max-width: 960px) {
      main { grid-template-columns: 1fr; }
      aside { position: static; }
      .summary { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>MetaAgent-X Interactive Demo</h1>
      <div class="status" id="serverStatus">Ready</div>
    </div>
    <div class="status" id="endpointInfo"></div>
  </header>
  <main>
    <aside>
      <label for="taskType">Task type</label>
      <select id="taskType">
        <option value="math">math</option>
        <option value="code">code</option>
      </select>
      <div class="examples">
        <button class="secondary" id="mathExample">Math example</button>
        <button class="secondary" id="codeExample">Code example</button>
      </div>
      <label for="question">Query</label>
      <textarea id="question"></textarea>
      <div class="row">
        <div>
          <label for="designRetries">Retries</label>
          <input id="designRetries" type="number" min="1" max="10" value="3">
        </div>
        <div>
          <label for="maxResponseLength">MAS max tokens</label>
          <input id="maxResponseLength" type="number" min="512" step="512" value="8192">
        </div>
      </div>
      <label for="outputDir">Output dir</label>
      <input id="outputDir" value="outputs/autoeval_interactive">
      <div style="display:flex; gap:8px; margin-top:14px;">
        <button id="runButton">Run MAS Demo</button>
        <button class="secondary" id="clearButton">Clear</button>
      </div>
    </aside>
    <section>
      <div class="error" id="errorBox"></div>
      <div class="summary">
        <div class="metric"><strong>Status</strong><span id="runStatus">Idle</span></div>
        <div class="metric"><strong>Task</strong><span id="summaryTask">-</span></div>
        <div class="metric"><strong>Final result</strong><span id="summaryAnswer">-</span></div>
        <div class="metric"><strong>Artifacts</strong><span id="summaryOutput">-</span></div>
      </div>
      <div class="tabs">
        <button class="tab active" data-panel="designPanel">Design</button>
        <button class="tab" data-panel="executionPanel">Execution</button>
        <button class="tab" data-panel="tracePanel">Agent Trace</button>
        <button class="tab" data-panel="flowPanel">Workflow</button>
        <button class="tab" data-panel="attemptsPanel">Attempts</button>
      </div>
      <div class="panel active" id="designPanel"><pre class="code" id="designCode">Run a query to see MAS design code.</pre></div>
      <div class="panel" id="executionPanel"><pre id="executionLog">Run a query to see execution output.</pre></div>
      <div class="panel" id="tracePanel"><div class="trace" id="traceList"><div class="empty">No trace yet.</div></div></div>
      <div class="panel" id="flowPanel"><div class="flow" id="flowView"><div class="flow-item">No workflow yet.</div></div><h3>Mermaid</h3><pre id="mermaidSource"></pre></div>
      <div class="panel" id="attemptsPanel"><pre id="attemptsText">No attempts yet.</pre></div>
    </section>
  </main>
  <script>
    const defaults = {
      math: __MATH_EXAMPLE__,
      code: __CODE_EXAMPLE__,
      endpoint: __ENDPOINT__,
      modelName: __MODEL_NAME__
    };
    const $ = (id) => document.getElementById(id);
    $("question").value = defaults.math;
    $("endpointInfo").textContent = `${defaults.modelName} @ ${defaults.endpoint}`;
    $("mathExample").onclick = () => { $("taskType").value = "math"; $("question").value = defaults.math; };
    $("codeExample").onclick = () => { $("taskType").value = "code"; $("question").value = defaults.code; };
    $("clearButton").onclick = () => {
      $("designCode").textContent = "";
      $("executionLog").textContent = "";
      $("traceList").innerHTML = '<div class="empty">No trace yet.</div>';
      $("flowView").innerHTML = '<div class="flow-item">No workflow yet.</div>';
      $("mermaidSource").textContent = "";
      $("attemptsText").textContent = "";
      $("summaryAnswer").textContent = "-";
      $("summaryOutput").textContent = "-";
      $("runStatus").textContent = "Idle";
      $("errorBox").style.display = "none";
    };
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.onclick = () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        $(tab.dataset.panel).classList.add("active");
      };
    });
    function renderFlow(steps) {
      const items = ["User query", ...(steps || []), "Final answer"];
      $("flowView").innerHTML = items.map((item) => `<div class="flow-item">${escapeHtml(item)}</div>`).join("");
    }
    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    async function runDemo() {
      $("runButton").disabled = true;
      $("runStatus").textContent = "Running";
      $("serverStatus").textContent = "Running MAS design and execution...";
      $("errorBox").style.display = "none";
      const payload = {
        task_type: $("taskType").value,
        question: $("question").value,
        output_dir: $("outputDir").value,
        design_retries: Number($("designRetries").value || 3),
        max_response_length: Number($("maxResponseLength").value || 8192)
      };
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || data.error) throw new Error(data.error || "Demo failed");
        $("runStatus").textContent = data.execution_success ? "Success" : "Completed";
        $("summaryTask").textContent = data.task_type || payload.task_type;
        $("summaryAnswer").textContent = data.final_answer || "No final answer extracted";
        $("summaryOutput").textContent = data.output_dir || "-";
        $("designCode").textContent = data.mas_design_text || "";
        $("executionLog").textContent = data.execution_log_text || "";
        $("mermaidSource").textContent = data.visualization_text || "";
        $("attemptsText").textContent = (data.attempts || []).map((item, idx) => `===== Attempt ${idx + 1} =====\\n${item}`).join("\\n\\n");
        renderFlow(data.workflow_steps || []);
        if (data.trace && data.trace.length) {
          $("traceList").innerHTML = data.trace.map((item) => `
            <article class="trace-card">
              <h3>${escapeHtml(item.name)}</h3>
              <pre>${escapeHtml(item.response || item.body || "")}</pre>
            </article>
          `).join("");
        } else {
          $("traceList").innerHTML = '<div class="empty">No AgentNode execution trace captured.</div>';
        }
      } catch (err) {
        $("runStatus").textContent = "Error";
        $("errorBox").textContent = err.message || String(err);
        $("errorBox").style.display = "block";
      } finally {
        $("runButton").disabled = false;
        $("serverStatus").textContent = "Ready";
      }
    }
    $("runButton").onclick = runDemo;
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler: BaseHTTPRequestHandler, status: int, text: str, content_type: str) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _workflow_steps_from_mermaid(source: str) -> list[str]:
    steps = []
    for line in source.splitlines():
        if '["' in line and '"]' in line and "input" not in line and "output" not in line:
            label = line.split('["', 1)[1].split('"]', 1)[0].replace("\\n", " ")
            steps.append(label)
    return steps


def _result_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(str(result.get("output_dir", "")))
    execution_log = _read_text(output_dir / "execution.log")
    visualization = _read_text(output_dir / "mas_visualization.mmd")
    attempts = [
        _read_text(path)
        for path in sorted(output_dir.glob("designer_response_attempt_*.txt"))
    ]
    payload = dict(result)
    payload.update(
        {
            "designer_response_text": _read_text(output_dir / "designer_response.txt"),
            "mas_design_text": _read_text(output_dir / "mas_design.py"),
            "execution_log_text": execution_log,
            "visualization_text": visualization,
            "html_report": str(output_dir / "index.html"),
            "attempts": attempts,
            "trace": _extract_agent_trace(execution_log),
            "workflow_steps": _workflow_steps_from_mermaid(visualization),
        }
    )
    return payload


def _make_handler(config: argparse.Namespace):
    class DemoHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[demo-ui] {self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                page = (
                    INDEX_HTML.replace("__MATH_EXAMPLE__", json.dumps(DEFAULT_QUESTION))
                    .replace("__CODE_EXAMPLE__", json.dumps(DEFAULT_CODE_QUESTION))
                    .replace("__ENDPOINT__", json.dumps(config.server_address))
                    .replace("__MODEL_NAME__", json.dumps(config.model_name))
                )
                _text_response(self, 200, page, "text/html; charset=utf-8")
                return
            if path == "/api/health":
                _json_response(self, 200, {"ok": True})
                return
            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/run":
                _json_response(self, 404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body or "{}")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_root = Path(payload.get("output_dir") or config.output_dir)
                output_dir = output_root / f"run_{timestamp}"
                demo_args = SimpleNamespace(
                    server_address=config.server_address,
                    model_name=config.model_name,
                    model_path=config.model_path,
                    tokenizer_path=config.tokenizer_path,
                    api_key=config.api_key,
                    question=payload.get("question") or DEFAULT_QUESTION,
                    task_type=payload.get("task_type") or "math",
                    output_dir=str(output_dir),
                    python_bin=config.python_bin,
                    temperature=float(payload.get("temperature", config.temperature)),
                    design_max_tokens=int(payload.get("design_max_tokens", config.design_max_tokens)),
                    design_retries=int(payload.get("design_retries", config.design_retries)),
                    max_prompt_length=int(payload.get("max_prompt_length", config.max_prompt_length)),
                    max_response_length=int(payload.get("max_response_length", config.max_response_length)),
                    execution_timeout=float(payload.get("execution_timeout", config.execution_timeout)),
                    enable_thinking=bool(payload.get("enable_thinking", config.enable_thinking)),
                    design_only=bool(payload.get("design_only", False)),
                )
                result = run_demo(demo_args)
                _json_response(self, 200, _result_payload(result))
            except Exception as exc:
                _json_response(self, 500, {"error": f"{type(exc).__name__}: {exc}"})

    return DemoHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the interactive MetaAgent-X demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--server-address", default="127.0.0.1:8300")
    parser.add_argument("--model-name", default="shared_model")
    parser.add_argument("--model-path", default="Mercury7353/MetaAgent-X")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--output-dir", default="outputs/autoeval_interactive")
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", "python3"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--design-max-tokens", type=int, default=8192)
    parser.add_argument("--design-retries", type=int, default=3)
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-response-length", type=int, default=8192)
    parser.add_argument("--execution-timeout", type=float, default=600.0)
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(args))
    print(f"MetaAgent-X interactive demo UI: http://{args.host}:{args.port}")
    print(f"Using model endpoint: {args.server_address} ({args.model_name})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
