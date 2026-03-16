# Bad Case Analysis — AIME24 Validation Run

**Run:** `autoeval_code_only_4design_4execution_5e_6_trainall`
**Date:** 2026-03-16
**Model:** Mercury7353/masrl_0228_mix_coldstart

---

## Overview

This directory contains analysis of failure modes observed in the AIME24 validation rollouts. Failures fall into two broad categories:

---

## Category 1: Pipeline Failures (`pipeline_failure/`)

These are rollouts where `[SUCCESSSAVED]` is absent from `output.txt`, meaning the MAS pipeline crashed before producing any usable output. The executor subprocess itself terminated with a Python exception.

### Subcategories observed

| Failure Type | Count | Example Rollouts |
|---|---|---|
| `OSError: Bad file descriptor` | 3 | rollout_104, rollout_146, rollout_150 |
| `SyntaxError` (malformed mas.py) | 1 | rollout_106 |
| `TypeError` in orchestration logic | 1 | rollout_127 |
| `AttributeError: NoneType has no .run()` | 1 | rollout_149 |

### Root cause summary

**Bad file descriptor (OSError 9):** The Designer generated a `solve()` function that calls `sys.stdin.read()` directly. When the MAS executor runs the script, stdin is already closed or redirected to a non-readable fd, causing `OSError: [Errno 9] Bad file descriptor`. The generated solution treats the problem as a competitive-programming script that reads from stdin at runtime rather than embedding the data in code or reading from a subprocess.

**SyntaxError in mas.py:** The Designer emitted an unclosed parenthesis in a multi-line string literal inside a `system_prompt=` argument (rollout_106), making the entire script unparseable.

**TypeError in orchestration:** The Designer wrote custom orchestration logic that concatenated a string and a list (`str + list`) during ensemble/brute-force testing (rollout_127).

**AttributeError NoneType:** The Designer defined `consensus_agent=None` in the ensemble but then called `.run()` on it. The conditional logic never populated the field before use (rollout_149).

---

## Category 2: Wrong Answer / No Valid Solution (`wrong_answer/`)

These rollouts completed with `[SUCCESSSAVED]` but produced no correct or extractable solution. The output reveals one or more of these failure modes:

| Failure Type | Count | Example Rollouts |
|---|---|---|
| Problem description truncated / agents asked for clarification | 1 | rollout_7 |
| Model admits inability and outputs apology instead of code | 1 | rollout_5 |
| UnitTestAgent produces garbage critique (massive repeated test scaffold) | 1 | rollout_3 |
| Simulation logic broken (wrong grid traversal algorithm) | 1 | rollout_9 |
| Tree path-sum formula incorrect (subtree-sum instead of path-sum) | 1 | rollout_8 |

### Root cause summary

**Truncated problem / clarification loop (rollout_7):** The Designer passed only the first line of the problem (`"Winter holiday decoration problem."`) as the question string into the workflow. All three ensemble solvers and the judge independently asked for more details — none produced code.

**Model admits failure (rollout_5):** The executor agent exhausted its token budget on a hard probability/Markov-chain problem, then explicitly emitted `"FINAL ANSWER: Unfortunately, I'm unable to provide..."`. A solution was wrapped in `<solution>` tags but contains placeholder `...` code.

**Critique agent runaway (rollout_3):** The UnitTestAgent generated an extreme stress-test scaffold (100+ test cases with strings up to length 50+) instead of critiquing correctness. The agent's own output includes a typo (`count_possible_intengers` on test 81) indicating the critique itself is AI-hallucinated and unvalidated.

**Wrong simulation (rollout_9):** The ball-dropping grid simulation does not correctly implement the problem's direction-flip rule (the cell the ball just left should flip, not the cell it enters). Tests were passed using hardcoded expected values, masking the bug.

**Wrong path-sum formula (rollout_8):** The tree query uses subtree prefix sums to compute path distances, but the formula `sum_u + sum_v - 2*sum_lca` is valid for subtree sums only when the subtree orientation matches the path — it fails for arbitrary tree structures. The correct approach requires path-based prefix sums using the Euler tour, not subtree sums.

---

## File Index

### `pipeline_failure/`
- `rollout_104.md` — OSError: Bad file descriptor (stdin closed)
- `rollout_106.md` — SyntaxError: unclosed parenthesis in system_prompt
- `rollout_127.md` — TypeError: str + list concatenation in ensemble logic
- `rollout_146.md` — OSError: Bad file descriptor (stdin closed)
- `rollout_149.md` — AttributeError: consensus_agent is None
- `rollout_150.md` — OSError: Bad file descriptor (stdin closed)

### `wrong_answer/`
- `rollout_3.md`  — Runaway UnitTestAgent critique, no valid solution extracted
- `rollout_5.md`  — Model admits inability; placeholder solution emitted
- `rollout_7.md`  — Truncated problem description; all agents ask for clarification
- `rollout_8.md`  — Wrong tree path-sum formula (subtree sum misused)
- `rollout_9.md`  — Ball simulation direction-flip bug masked by test scaffold
