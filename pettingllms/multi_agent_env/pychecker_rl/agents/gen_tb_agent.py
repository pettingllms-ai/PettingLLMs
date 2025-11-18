import json
import logging
import subprocess
import sys
import os
from typing import Dict, List, Any

# Import from consolidated utils
from pettingllms.multi_agent_env.pychecker_rl.pychecker_utils import (
    extract_code_from_response as pro_extract_code,
    create_task_folder
)
import shutil

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


# ============================================================================
# CMB (Combinational) Circuit Configuration
# ============================================================================

CMB_SYSTEM_PROMPT = """
You are an expert in RTL design. 
"""

CMB_GENERATION_PROMPT = """
Your task is to generate a Python function named "stimulus_gen" that produces test vectors for a combinational circuit.

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

**CRITICAL**: Use the EXACT **INPUT** signal names from the module_header (excluding clk). DO NOT include output signals. DO NOT use generic placeholder names.

## Output Format

**CRITICAL**: Your function MUST return a **list of dictionaries** (NOT tuples, NOT lists).

Each dictionary represents one test vector:
- Dictionary keys = **INPUT** signal names ONLY from module_header (DO NOT include outputs)
- Dictionary values = binary strings (e.g., "0", "1", "1010")

**CORRECT format** (dictionary with INPUT signals only):
```python
# If module has: input a, input b, output out
# CORRECT: Only include inputs
[
  {{"a": "0", "b": "1"}},
  {{"a": "1", "b": "0"}},
  ...
]
```

**WRONG formats** (DO NOT USE):
```python
# Wrong: includes output signal 'out'
[{{"a": "0", "b": "1", "out": "1"}}]  # ❌ NO! Don't include outputs!

# Wrong: tuple
[("0", "101"), ("1", "010")]  # ❌ NO!

# Wrong: list
[["0", "101"], ["1", "010"]]  # ❌ NO!

# Wrong: no signal names
[{{"0": "0", "1": "101"}}]  # ❌ NO! Use actual signal names
```

## Requirements

1. **INPUT signals ONLY**: Include ONLY input signals. NEVER include output signals in test vectors.
2. **Binary strings only**: Use "0", "1", "101", etc. No 'X' or 'Z' values.
3. **All inputs included**: Every test vector must include all INPUT signals from the module header.
4. **Variable names**: Must exactly match the module header (excluding clock signals and outputs).
5. **Dictionary format**: Each test vector MUST be a dictionary with signal names as keys.
6. **IMPORTANT**: DO NOT hard-code stimulus as JSON arrays directly. Use Python programming techniques to generate them dynamically.

## Programming Techniques Required

You can use Python programming constructs to generate test vectors:
- Use `random` module for random value generation
- Use `numpy` for array operations and batch generation
- Use list comprehensions and loops for dynamic generation
- Use format strings to convert numbers to binary
- Use range() and iterators for exhaustive testing




## Key Techniques

**Exhaustive testing with bit manipulation:**
```python
for i in range(2**width):
    signal = format(i, f'0{{width}}b')
```

**Random values with specific bit width:**
```python
val = random.getrandbits(8)  # 8-bit random value
test_vectors.append({{"data": format(val, '08b')}})
```

**Batch generation with numpy:**
```python
values = np.random.randint(0, 256, 100)  # 100 random 8-bit values
signals = [format(v, '08b') for v in values]
```

**List comprehension for signal generation:**
```python
random_signals = [format(random.getrandbits(8), '08b') for _ in range(50)]
```

## Complete Example

For a module with header:
```verilog
module example(input [3:0] data, input enable, output out);
```


Now implement the stimulus_gen() function using Python programming techniques.

**IMPORTANT**: You MUST return your code in this exact format:
```python
def stimulus_gen():
    test_vectors = []
    # your implementation here
    # Each test vector MUST be a dictionary with signal names as keys
    test_vectors.append({{"signal_name": "binary_value"}})
    return test_vectors  # MUST return list of dictionaries
```

**REMINDER**:
- Use `{{"key": "value"}}` (dictionary) NOT `("value1", "value2")` (tuple)
- Use actual signal names from module_header as dictionary keys
"""

CMB_INSTRUCTIONS = """
Instructions for stimulus_gen():
1. Return a list of dictionaries
2. Each dictionary has input signal names as keys, binary strings as values
3. Include all input signals (except clk) from the module header
4. Generate comprehensive test cases covering corners, edges, and random cases
"""

CMB_PYTHON_HEADER = """
import json
import random
import numpy as np
import subprocess
import re
"""

CMB_TAIL = """
def extract_module_ports(verilog_file="top_module.v"):
    \"\"\"
    Extract module input/output port names (excluding clk/rst) from Verilog file.
    Returns dict with 'inputs' and 'outputs' sets.
    \"\"\"
    try:
        with open(verilog_file, 'r') as f:
            content = f.read()

        # Extract module declaration
        module_match = re.search(r'module\\s+\\w+\\s*\\((.*?)\\);', content, re.DOTALL)
        if not module_match:
            return {"inputs": set(), "outputs": set()}

        port_list = module_match.group(1)

        # Extract input signals (excluding clk/rst)
        input_pattern = r'input\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        inputs = set()
        for match in re.findall(input_pattern, port_list):
            if match.lower() not in ['clk', 'clock', 'rst', 'reset', 'rstn', 'rst_n', 'areset']:
                inputs.add(match)

        # Extract output signals
        output_pattern = r'output\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        outputs = set(re.findall(output_pattern, port_list))

        return {"inputs": inputs, "outputs": outputs}
    except Exception as e:
        print(f"Error extracting ports: {e}")
        return {"inputs": set(), "outputs": set()}

def fuzzy_match_signal(test_name, actual_signals):
    \"\"\"Fuzzy match test signal name to actual signal name.\"\"\"
    if test_name in actual_signals:
        return test_name
    for actual in actual_signals:
        if test_name.lower() == actual.lower():
            print(f"  Fuzzy match: '{test_name}' -> '{actual}' (case-insensitive)")
            return actual
    test_variations = [test_name.replace('_', ''), test_name + '_in', test_name + '_out',
                      test_name.rstrip('_in'), test_name.rstrip('_out')]
    for variation in test_variations:
        for actual in actual_signals:
            if variation.lower() == actual.lower():
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (via '{variation}')")
                return actual
    abbreviation_map = {
        'data_in': ['data', 'din', 'd', 'in'], 'data_out': ['data', 'dout', 'q', 'out'],
        'din': ['data_in', 'data', 'd', 'in'], 'dout': ['data_out', 'data', 'q', 'out'],
        'load': ['load', 'ld', 'en', 'enable'], 'ena': ['enable', 'en', 'ena'],
        'enable': ['ena', 'en', 'enable'], 'data': ['data_in', 'din', 'data_out', 'dout'],
        'q': ['data_out', 'dout', 'out'], 'd': ['data_in', 'din', 'data'],
        'in': ['data_in', 'din'], 'out': ['data_out', 'dout'],
    }
    if test_name.lower() in abbreviation_map:
        for candidate in abbreviation_map[test_name.lower()]:
            for actual in actual_signals:
                if candidate.lower() == actual.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (abbreviation)")
                    return actual
    for actual in actual_signals:
        if actual.lower() in abbreviation_map:
            for candidate in abbreviation_map[actual.lower()]:
                if candidate.lower() == test_name.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (reverse abbreviation)")
                    return actual
    for actual in actual_signals:
        test_lower, actual_lower = test_name.lower(), actual.lower()
        if test_lower in actual_lower or actual_lower in test_lower:
            min_len, max_len = min(len(test_lower), len(actual_lower)), max(len(test_lower), len(actual_lower))
            if min_len / max_len >= 0.6:
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (substring)")
                return actual
    return None

def fix_stimulus(stimulus_data, verilog_file="top_module.v"):
    \"\"\"Fix stimulus by fuzzy matching, filtering outputs, and adding missing signals.\"\"\"
    if not stimulus_data or not isinstance(stimulus_data, list):
        return stimulus_data, ["Stimulus is empty or invalid"]
    ports = extract_module_ports(verilog_file)
    expected_inputs = ports["inputs"]
    expected_outputs = ports["outputs"]
    if not expected_inputs:
        return stimulus_data, ["Could not extract input ports"]
    corrected_data, warnings = [], []
    for idx, test_vector in enumerate(stimulus_data):
        if not isinstance(test_vector, dict):
            warnings.append(f"Vector {idx} not a dict, skipping")
            continue
        corrected_vector = {}
        for test_signal, value in test_vector.items():
            # Check if it's an output signal (should be filtered out)
            if test_signal in expected_outputs or fuzzy_match_signal(test_signal, expected_outputs):
                warnings.append(f"Signal '{test_signal}' is an OUTPUT, filtering out")
                continue
            # Try to match input signals
            matched = fuzzy_match_signal(test_signal, expected_inputs)
            if matched:
                corrected_vector[matched] = value
            else:
                warnings.append(f"Signal '{test_signal}' not matched to any input, skipping")
        for expected_signal in expected_inputs:
            if expected_signal not in corrected_vector:
                random_val = format(random.randint(0, 255), '08b')
                corrected_vector[expected_signal] = random_val
                warnings.append(f"Added missing '{expected_signal}': {random_val}")
        corrected_data.append(corrected_vector)
    return corrected_data, warnings

if __name__ == "__main__":
    import os
    import sys
    result = stimulus_gen()
    print("\\n=== Fixing and verifying stimulus ===")
    fixed_result, warnings = fix_stimulus(result, verilog_file="top_module.v")
    if warnings:
        print("⚠ Warnings:")
        for w in warnings: print(f"  - {w}")
    with open("stimulus.json", "w") as f:
        json.dump(fixed_result, f, indent=2)
    print("✓ Saved corrected stimulus.json")
    print("==============================================\\n")
"""



# ============================================================================
# SEQ (Sequential) Circuit Configuration
# ============================================================================

SEQ_SYSTEM_PROMPT = """
You are an expert in RTL design. You can always write correct testbenches for RTL designs.
"""

SEQ_GENERATION_PROMPT = """
Your task is to generate a Python function named "stimulus_gen" that produces test scenarios for a sequential circuit.

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

## Output Format

Return a list of test scenarios. Each scenario specifies clock cycles and **INPUT** signal sequences.

**CRITICAL**: Use the EXACT **INPUT** signal names from the module_header (excluding clk). DO NOT include output signals. DO NOT use generic placeholder names.


## Requirements

1. **INPUT signals ONLY**: Include ONLY input signals. NEVER include output signals in scenarios.
2. **Clock cycles**: Each scenario must have a "clock_cycles" field (integer)
3. **Signal sequences**: Each INPUT signal is a list of binary strings with length = clock_cycles
4. **Binary strings only**: Use "0", "1", etc. No 'X' or 'Z' values.
5. **CRITICAL - Signal Names**: You MUST use EXACTLY the same INPUT signal names as in the module_header above (excluding clk and outputs). DO NOT use generic names like "reset", "enable", etc. if the actual signal names are different (e.g., "areset", "en").
6. **All inputs included**: Include all INPUT signals from module header (except clk, exclude outputs)
7. **Comprehensive testing**: Include:
   - Reset sequences
   - Normal operation
   - Edge cases (state transitions, wraparounds)
   - Random scenarios (at least 10-20)
8. **IMPORTANT**: DO NOT hard-code stimulus sequences as literal arrays. Use Python programming techniques to generate them dynamically.

## Programming Techniques Required

You MUST use Python programming constructs to generate test scenarios:
- Use `random` module for random value generation and patterns
- Use `numpy` for array operations and sequence generation
- Use list comprehensions to generate signal sequences
- Use loops and conditionals for intelligent pattern generation
- Use format strings to convert numbers to binary

## Complete Example

For a shift register:
```verilog
module shift_reg(input clk, input rst, input [7:0] data_in, input load, output [7:0] data_out);
```

**IMPORTANT: Signal names in this example match the module_header above. Use the EXACT signal names from YOUR module_header.**

```python
import random
import numpy as np

def stimulus_gen():
    scenarios = []
    
    # Reset scenario - NOTE: using "rst" because that's the signal name in module_header
    cycles = 5
    scenarios.append({{
        "clock_cycles": cycles,
        "rst": ["1"] + ["0"] * (cycles - 1),  # Use actual signal name from module_header
        "load": ["0"] * cycles,
        "data_in": ["00000000"] * cycles
    }})
  
    # Random scenarios
    for _ in range(20):
        cycles = random.randint(15, 30)
        scenarios.append({{
            "clock_cycles": cycles,
            "rst": ["0"] * cycles,
            "load": [random.choice(["0", "1"]) for _ in range(cycles)],
            "data_in": [format(random.getrandbits(8), '08b') for _ in range(cycles)]
        }})
    
    return scenarios
```

## Key Techniques

**Reset pattern generation (use actual reset signal name from module_header):**
```python
# If module has "areset": use areset
areset_signal = ["1"] + ["0"] * (cycles - 1)  # Assert reset on first cycle
# If module has "rst": use rst
# If module has "reset": use reset
```

**Random single-bit sequences (use actual signal names from module_header):**
```python
# Example: if module has "enable" signal
enable = [random.choice(["0", "1"]) for _ in range(cycles)]
```

**Random multi-bit sequences:**
```python
data = [format(random.getrandbits(8), '08b') for _ in range(cycles)]
```

**Numpy for probability-based patterns:**
```python
pattern = (np.random.random(cycles) > 0.3).astype(int)  # 70% ones, 30% zeros
signal = [str(v) for v in pattern]
```

**Burst operation patterns:**
```python
signal = ["0"] * cycles
for i in range(start_pos, end_pos):
    signal[i] = "1"
```

**State transition sequences:**
```python
test_patterns = ["0011", "1010", "1111"]
for pattern in test_patterns:
    signal_seq = list(pattern) + ["0"] * extra_cycles
```

Now implement the stimulus_gen() function using Python programming techniques.

**IMPORTANT**: You MUST return your code in this exact format:
```python
def stimulus_gen():
    # your implementation here
```
"""

SEQ_INSTRUCTIONS = """
Instructions for stimulus_gen():
1. Return a list of dictionaries (scenarios)
2. Each scenario must have "clock_cycles" (integer) and input signal lists
3. All signal lists must have length equal to clock_cycles
4. Include all input signals (except clk) from the module header
5. Generate comprehensive scenarios: reset, normal operation, edge cases, and random tests
"""

SEQ_PYTHON_HEADER = """
import json
import random
import numpy as np
import subprocess
import re
"""

SEQ_TAIL = """
def extract_module_ports(verilog_file="top_module.v"):
    \"\"\"
    Extract module input/output port names (excluding clk/rst) from Verilog file.
    Returns dict with 'inputs' and 'outputs' sets.
    \"\"\"
    try:
        with open(verilog_file, 'r') as f:
            content = f.read()

        # Extract module declaration
        module_match = re.search(r'module\\s+\\w+\\s*\\((.*?)\\);', content, re.DOTALL)
        if not module_match:
            return {"inputs": set(), "outputs": set()}

        port_list = module_match.group(1)

        # Extract input signals (excluding clk/rst)
        input_pattern = r'input\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        inputs = set()
        for match in re.findall(input_pattern, port_list):
            if match.lower() not in ['clk', 'clock', 'rst', 'reset', 'rstn', 'rst_n', 'areset']:
                inputs.add(match)

        # Extract output signals
        output_pattern = r'output\\s+(?:(?:wire|reg|logic)\\s+)?(?:\\[\\s*\\d+\\s*:\\s*\\d+\\s*\\])?\\s*(\\w+)'
        outputs = set(re.findall(output_pattern, port_list))

        return {"inputs": inputs, "outputs": outputs}
    except Exception as e:
        print(f"Error extracting ports: {e}")
        return {"inputs": set(), "outputs": set()}

def fuzzy_match_signal(test_name, actual_signals):
    \"\"\"Fuzzy match test signal name to actual signal name.\"\"\"
    if test_name in actual_signals:
        return test_name
    for actual in actual_signals:
        if test_name.lower() == actual.lower():
            print(f"  Fuzzy match: '{test_name}' -> '{actual}' (case-insensitive)")
            return actual
    test_variations = [test_name.replace('_', ''), test_name + '_in', test_name + '_out',
                      test_name.rstrip('_in'), test_name.rstrip('_out')]
    for variation in test_variations:
        for actual in actual_signals:
            if variation.lower() == actual.lower():
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (via '{variation}')")
                return actual
    abbreviation_map = {
        'data_in': ['data', 'din', 'd', 'in'], 'data_out': ['data', 'dout', 'q', 'out'],
        'din': ['data_in', 'data', 'd', 'in'], 'dout': ['data_out', 'data', 'q', 'out'],
        'load': ['load', 'ld', 'en', 'enable'], 'ena': ['enable', 'en', 'ena'],
        'enable': ['ena', 'en', 'enable'], 'data': ['data_in', 'din', 'data_out', 'dout'],
        'q': ['data_out', 'dout', 'out'], 'd': ['data_in', 'din', 'data'],
        'in': ['data_in', 'din'], 'out': ['data_out', 'dout'],
    }
    if test_name.lower() in abbreviation_map:
        for candidate in abbreviation_map[test_name.lower()]:
            for actual in actual_signals:
                if candidate.lower() == actual.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (abbreviation)")
                    return actual
    for actual in actual_signals:
        if actual.lower() in abbreviation_map:
            for candidate in abbreviation_map[actual.lower()]:
                if candidate.lower() == test_name.lower():
                    print(f"  Fuzzy match: '{test_name}' -> '{actual}' (reverse abbreviation)")
                    return actual
    for actual in actual_signals:
        test_lower, actual_lower = test_name.lower(), actual.lower()
        if test_lower in actual_lower or actual_lower in test_lower:
            min_len, max_len = min(len(test_lower), len(actual_lower)), max(len(test_lower), len(actual_lower))
            if min_len / max_len >= 0.6:
                print(f"  Fuzzy match: '{test_name}' -> '{actual}' (substring)")
                return actual
    return None

def fix_stimulus_seq(stimulus_data, verilog_file="top_module.v"):
    \"\"\"Fix sequential stimulus: filter outputs, fuzzy match signals, fix cycle counts, pad/truncate sequences.\"\"\"
    if not stimulus_data or not isinstance(stimulus_data, list):
        return stimulus_data, ["Stimulus is empty or invalid"]
    ports = extract_module_ports(verilog_file)
    expected_inputs = ports["inputs"]
    expected_outputs = ports["outputs"]
    if not expected_inputs:
        return stimulus_data, ["Could not extract input ports"]
    corrected_data, warnings = [], []
    for idx, scenario in enumerate(stimulus_data):
        if not isinstance(scenario, dict):
            warnings.append(f"Scenario {idx} not a dict, skipping")
            continue
        if "clock_cycles" not in scenario:
            warnings.append(f"Scenario {idx} missing 'clock_cycles', skipping")
            continue
        clock_cycles = scenario["clock_cycles"]
        corrected_scenario = {"clock_cycles": clock_cycles}
        # Fuzzy match and fix signal sequences
        for test_signal, values in scenario.items():
            if test_signal == "clock_cycles":
                continue
            # Check if it's an output signal (should be filtered out)
            if test_signal in expected_outputs or fuzzy_match_signal(test_signal, expected_outputs):
                warnings.append(f"Scenario {idx}: Signal '{test_signal}' is an OUTPUT, filtering out")
                continue
            # Try to match input signals
            matched = fuzzy_match_signal(test_signal, expected_inputs)
            if matched:
                if isinstance(values, list):
                    if len(values) < clock_cycles:
                        # Pad with zeros
                        padded = values + ["0"] * (clock_cycles - len(values))
                        corrected_scenario[matched] = padded
                        warnings.append(f"Scenario {idx}: Padded '{matched}' from {len(values)} to {clock_cycles} cycles")
                    elif len(values) > clock_cycles:
                        # Truncate
                        truncated = values[:clock_cycles]
                        corrected_scenario[matched] = truncated
                        warnings.append(f"Scenario {idx}: Truncated '{matched}' from {len(values)} to {clock_cycles} cycles")
                    else:
                        corrected_scenario[matched] = values
                else:
                    # Single value, convert to list
                    corrected_scenario[matched] = [str(values)] * clock_cycles
                    warnings.append(f"Scenario {idx}: Converted single value '{matched}' to sequence")
            else:
                warnings.append(f"Scenario {idx}: Signal '{test_signal}' not matched to any input, skipping")
        # Add missing signals with random sequences
        for expected_signal in expected_inputs:
            if expected_signal not in corrected_scenario:
                random_seq = [format(random.randint(0, 1), 'b') for _ in range(clock_cycles)]
                corrected_scenario[expected_signal] = random_seq
                warnings.append(f"Scenario {idx}: Added missing '{expected_signal}' with random sequence")
        corrected_data.append(corrected_scenario)
    return corrected_data, warnings

if __name__ == "__main__":
    import os
    import sys
    result = stimulus_gen()
    print("\\n=== Fixing and verifying stimulus ===")
    fixed_result, warnings = fix_stimulus_seq(result, verilog_file="top_module.v")
    if warnings:
        print("⚠ Warnings:")
        for w in warnings: print(f"  - {w}")
    with open("stimulus.json", "w") as f:
        json.dump(fixed_result, f, indent=2)
    print("✓ Saved corrected stimulus.json")
    print("==============================================\\n")
"""

logger = logging.getLogger(__name__)

# ============================================================================
# TB_Generator Class
# ============================================================================



# ============================================================================
# GenTBAgent Class (Agent Interface for Multi-Agent RL)
# ============================================================================

class GenTBAgent:
    """
    Agent for generating stimulus test benches for hardware verification
    Integrates with the multi-agent RL framework
    """

    def __init__(self, rollout_idx: int | None = None, **kwargs):
        """
        Initialize the GenTB Agent
        
        Args:
            rollout_idx: Rollout index for tracking
            **kwargs: Additional arguments (sample_num, etc.)
        """
        self.rollout_idx = rollout_idx
        self.sample_num = kwargs.get('sample_num', 1)
        
        # Agent state
        self.current_action = None  # Generated stimulus code
        self.current_prompt = None
        self.current_response = None
        self.agent_reward = 0.0
        self.success = False
        
        # History tracking
        self.action_history = []
        self.answer_history = []
        self.reward_history = []
        self.env_data = None
        
        # Stimulus generation tracking
        self.stimulus_json_path = None
        self.stimulus_generated = False
        self.task_folder = None
        
        # Accept other keyword arguments for compatibility
        for key, value in (kwargs or {}).items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def update_from_env(self, turn_idx: int, env_data):
        """
        Update agent state from environment and generate prompt for stimulus generation

        Args:
            turn_idx: Current turn index (should be 0 for GenTB agent)
            env_data: Environment data object
        """
        self.env_data = env_data
        state = getattr(env_data, "state", None)

        problem_input = getattr(state, "problem_input", None)
        spec = getattr(state, "spec", "")
        circuit_type = getattr(state, "circuit_type", "CMB")

        # Create task folder path with GPU group and worker_id for isolation
        env_idx = getattr(env_data, "env_idx", 0)
        rollout_idx = getattr(env_data, "rollout_idx", self.rollout_idx)

        # Get assigned worker_id and GPU group from env state (set by execution engine)
        worker_id = getattr(getattr(env_data, 'state', None), 'assigned_worker_id', None)
        gpu_group_id = getattr(getattr(env_data, 'state', None), 'gpu_group_id', None)

        if worker_id is None:
            # Fallback: compute worker_id deterministically
            worker_id = rollout_idx % 512  # default num_workers

        if gpu_group_id is None:
            # Fallback: detect GPU group from environment
            cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if cuda_visible:
                gpu_ids = sorted([g.strip() for g in cuda_visible.split(",") if g.strip()])
                gpu_group_id = f"gpu_{'_'.join(gpu_ids)}"
            else:
                gpu_group_id = "gpu_default"

        # Include GPU group and worker_id in path to ensure isolation between concurrent tasks
        # Different GPU groups (CUDA_VISIBLE_DEVICES) will have completely separate storage
        base_dir = os.path.join(os.getcwd(), "tmp", "pychecker_tasks")
        self.task_folder = os.path.join(
            base_dir,
            gpu_group_id,  # GPU group isolation (e.g., "gpu_0_1" or "gpu_3_4")
            f"worker_{worker_id}",  # Worker isolation within GPU group
            f"env_{env_idx}",
            f"rollout_{rollout_idx}",
            f"turn_{turn_idx}"
        )

        # Delete task_folder if it exists and rebuild it
        if os.path.exists(self.task_folder):
            shutil.rmtree(self.task_folder)

        # Create fresh task folder
        os.makedirs(self.task_folder, exist_ok=True)
        self.stimulus_json_path = os.path.join(self.task_folder, "stimulus.json")

        # Store task_folder in env state for next agent
        if state is not None:
            state.task_folder = self.task_folder

        # Generate prompt based on circuit type
        formatted_prompt = self._generate_stimulus_prompt(problem_input, spec, circuit_type)
        self.current_prompt = {"text": formatted_prompt, "image": None}

    def _generate_stimulus_prompt(self, problem_input: str, spec: str, circuit_type: str) -> str:
        """Generate prompt for stimulus generation"""

        if circuit_type == "CMB":
            prompt = CMB_SYSTEM_PROMPT + "\n\n" + CMB_GENERATION_PROMPT.format(
                description=problem_input,
                module_header=spec
            )
        else:  # SEQ
            prompt = SEQ_SYSTEM_PROMPT + "\n\n" + SEQ_GENERATION_PROMPT.format(
                description=problem_input,
                module_header=spec
            )

        return prompt

    def update_from_model(self, response: str):
        """
        Parse the LLM response and extract stimulus generation code
        
        Args:
            response: LLM response string
            
        Returns:
            Extracted Python code
        """
        self.current_response = response
        self.current_action = pro_extract_code(response, code_type="python")
        return self.current_action

    async def step(self, env_data, env_worker: Any = None):
        """
        Execute stimulus generation code to produce stimulus.json

        Args:
            env_data: Environment data object
            env_worker: Optional worker for async execution
        """
        generated_code = self.current_action
        env_data.state.task_folder = self.task_folder

        # Initialize info dict
        info = {
            'code_extracted': False,
            'code_runs': False,
            'stimulus_created': False,
            'error_message': ''
        }

        reward = 0.0

        # Check if code was extracted
        if not generated_code or len(generated_code.strip()) < 10:
            info['error_message'] = 'No stimulus code extracted from LLM response'
            logger.warning(f"Stimulus code extraction failed for rollout {self.rollout_idx}")
            self.agent_reward = 0.0

            # Update env state with agent-specific fields
            env_data.state.tb_extracted = info['code_extracted']
            env_data.state.tb_runs = info['code_runs']
            env_data.state.tb_stimulus_created = info['stimulus_created']
            env_data.state.tb_error_message = info['error_message']

            logger.info(f"GenTB Rollout {self.rollout_idx}: reward={reward:.2f}, "
                       f"extracted={info['code_extracted']}, runs={info['code_runs']}, "
                       f"created={info['stimulus_created']}")
            return

        info['code_extracted'] = True

        try:
            # Add Python header and tail based on circuit type
            circuit_type = env_data.state.circuit_type
            if circuit_type == "SEQ":
                python_header = SEQ_PYTHON_HEADER
                tail = SEQ_TAIL
            else:
                python_header = CMB_PYTHON_HEADER
                tail = CMB_TAIL

            complete_code = python_header + "\n" + generated_code + "\n" + tail

            # Save to file
            stimulus_py_path = os.path.join(self.task_folder, "stimulus_gen.py")
            with open(stimulus_py_path, 'w') as f:
                f.write(complete_code)

            # Execute the code to generate stimulus.json with Ray worker support
            # Get Ray worker from env_worker if available
            ray_actor = getattr(env_worker, 'ray_actor', None) if env_worker else None

            if ray_actor is not None:
                # Use Ray worker for parallel execution
                try:
                    from pettingllms.multi_agent_env.pychecker_rl.pychecker_worker import get_stimulus_generation_result

                    success, error_msg = await get_stimulus_generation_result(
                        stimulus_py_path=stimulus_py_path,
                        task_folder=self.task_folder,
                        timeout=60,
                        ray_actor=ray_actor
                    )

                    if not success:
                        info['error_message'] = f"Stimulus execution failed: {error_msg}"
                        logger.warning(f"Stimulus execution failed: {error_msg}")
                    else:
                        info['code_runs'] = True

                        # Check if stimulus.json was created
                        if os.path.exists(self.stimulus_json_path):
                            with open(self.stimulus_json_path, 'r') as f:
                                stimulus_data = json.load(f)

                            info['stimulus_created'] = True
                            reward = 0.0  # No reward for generating testbench.json (will be updated to 1.0 if simulation succeeds)
                            self.stimulus_generated = True

                            # Store stimulus path in env state for next agent
                            env_data.state.stimulus_json_path = self.stimulus_json_path

                            logger.info(f"Generated {len(stimulus_data)} test vectors/scenarios")
                        else:
                            info['error_message'] = "stimulus.json not created"
                            logger.warning("stimulus.json not created by the generated code")

                except Exception as e:
                    logger.error(f"Ray worker stimulus execution failed, falling back to local: {e}")
                    # Fallback to local execution
                    ray_actor = None

            if ray_actor is None:
                # Fallback to synchronous execution
                result = subprocess.run(
                    ["python", stimulus_py_path],
                    cwd=self.task_folder,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    info['error_message'] = f"Stimulus execution failed: {result.stderr}"
                    logger.warning(f"Stimulus execution failed: {result.stderr}")
                else:
                    info['code_runs'] = True

                    # Check if stimulus.json was created
                    if os.path.exists(self.stimulus_json_path):
                        with open(self.stimulus_json_path, 'r') as f:
                            stimulus_data = json.load(f)

                        info['stimulus_created'] = True
                        reward = 0.0  # No reward for generating testbench.json (will be updated to 1.0 if simulation succeeds)
                        self.stimulus_generated = True

                        # Store stimulus path in env state for next agent
                        env_data.state.stimulus_json_path = self.stimulus_json_path

                        logger.info(f"Generated {len(stimulus_data)} test vectors/scenarios")
                    else:
                        info['error_message'] = "stimulus.json not created"
                        logger.warning("stimulus.json not created by the generated code")

        except subprocess.TimeoutExpired:
            info['error_message'] = "Stimulus execution timeout"
            logger.error(f"Stimulus execution timeout for rollout {self.rollout_idx}")
        except Exception as e:
            info['error_message'] = f"Stimulus generation error: {str(e)}"
            logger.error(f"Stimulus generation error: {e}", exc_info=True)
        
        self.agent_reward = reward
        
        # Update env state with agent-specific fields
        env_data.state.tb_extracted = info['code_extracted']
        env_data.state.tb_runs = info['code_runs']
        env_data.state.tb_stimulus_created = info['stimulus_created']
        env_data.state.tb_error_message = info['error_message']
        
        logger.info(f"GenTB Rollout {self.rollout_idx}: reward={reward:.2f}, "
                   f"extracted={info['code_extracted']}, runs={info['code_runs']}, "
                   f"created={info['stimulus_created']}")

    def calculate_reward(self, env_data):
        """
        Calculate reward based on simulation results
        - If simulation successfully ran: reward = 1.0
        - Otherwise: reward = 0.0
        """
        # Check if simulation ran successfully (from PyCheckerAgent results)
        if hasattr(env_data, 'state') and hasattr(env_data.state, 'pychecker_sim_success'):
            if env_data.state.pychecker_sim_success:
                # Simulation ran successfully (may or may not pass all tests)
                self.agent_reward = 1.0
            # else: keep the reward from step() (0.0)

        self.success = (self.agent_reward >= 1.0)

        # Update history
        self.action_history.append(self.current_action)
        self.reward_history.append(self.agent_reward)

    def reset(self):
        """
        Reset the agent's internal state for a new episode.
        """
        self.current_action = None
        self.current_prompt = None
        self.current_response = None
        self.agent_reward = 0.0
        self.success = False
        self.action_history = []
        self.answer_history = []
        self.reward_history = []
        
        # Reset stimulus tracking
        self.task_folder = None
        self.stimulus_json_path = None
        self.stimulus_generated = False
