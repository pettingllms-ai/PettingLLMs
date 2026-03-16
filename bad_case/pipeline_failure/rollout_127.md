# Pipeline Failure: rollout_127

**Failure type:** `TypeError: can only concatenate str (not "list") to str`
**Status:** No `[SUCCESSSAVED]` in output.txt

---

## Problem Description

Given a string S (with only characters from T) and a 3-character string T, rearrange S so that T is not a subsequence of S. Multiple test cases.

Sample:
```
7
abacaba  -> abc  => aaaacbb
cccba    -> acb  => abccc
...
```

---

## Designer's MAS Script — Key Issue (mas.py lines 193–214)

The Designer defined a `solver_bruteforce()` function that contains embedded Python code as a multi-line string (`code = '''...'''`), then attempts to prepend a test input to it:

```python
def solver_bruteforce() -> str:
    code = '''
from functools import lru_cache
import sys
...
'''
    test_input = '''7
abacaba
abc
...'''

    full_test = f'''{test_input}'''+code.split('\n')[5:]   # <-- crash line 211
    print("Testing BRUTEFORCE strategy:")
    res = execute_code(full_test)
```

`code.split('\n')[5:]` returns a **list** of strings (all lines after line 5). The f-string produces a `str`, so `str + list` raises `TypeError`.

The intent was to combine the test input with the solution code for testing, but the Designer incorrectly used string splitting to skip header lines instead of constructing a proper combined script.

Additionally, the embedded `code` string itself is internally broken — it defines variables like `S`, `T`, `n`, `m` without binding them, uses `lru_cache` with a recursive function that references unbound outer variables, and iterates over `product(range(n), repeat=n)` (exponential) as a placeholder fallback.

---

## Executor Output (output.txt — relevant portion)

```
error: Traceback (most recent call last):
  File ".../script.py", line 608, in <module>
    result = run_ensemble_judge()
  File ".../script.py", line 523, in run_ensemble_judge
    bf_result = solver_bruteforce()
  File ".../script.py", line 211, in solver_bruteforce
    full_test = f'''{test_input}'''+code.split('\n')[5:]
                ~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~
TypeError: can only concatenate str (not "list") to str

STDOUT:
[AICLIENT SETUP] AIClient created successfully
==================================================
ENSEMBLE COMPETITION RESULTS:
==================================================
Running Ensemble Solver Competition
--------------------------------------------------

=== BRUTE FORCE SOLVER ===
```

The pipeline did start and the ensemble framework began executing before the crash.

---

## Failure Analysis

**Root cause:** The Designer built its own custom ensemble and testing framework inline in mas.py (reimplementing `ToolRegistry`, `AgentNode`, `EnsembleNode`, etc. as stub classes). Within this custom framework, the `solver_bruteforce()` function attempted to concatenate a string with a list slice — a Python type error.

**Secondary issues:**
1. The embedded bruteforce code string itself is logically broken (undefined variables, exponential complexity placeholder).
2. The Designer reimplemented framework classes (`ToolRegistry`, `AgentNode`) as stubs with hardcoded return values instead of calling the actual LLM — the agents would never solve the real problem.
3. The script is 608+ lines of mixed orchestration logic and broken solver stubs.

**Pattern:** The Designer attempted to write a self-contained testing harness and embedded problem-solving logic simultaneously, producing a sprawling script with logic errors at the glue code level.
