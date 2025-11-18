CMB_SYSTEM_PROMPT = """You are an expert in RTL design and Python programming. You can always write correct Python code to verify RTL functionality."""





# ============================================================================
# SEQ (Sequential) Circuit Configuration
# ============================================================================

SEQ_SYSTEM_PROMPT = """You are an expert in RTL design and Python programming. You can always write correct Python code to verify RTL functionality."""


# ============================================================================
# CMB (Combinational) Circuit Prompts
# ============================================================================

CMB_SYSTEM_PROMPT = """You are an expert in RTL design and Python programming. You can always write correct Python code to verify RTL functionality."""

CMB_GENERATION_PROMPT =  r"""
You are implementing a Python class "GoldenDUT" for combinational logic.

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

## Requirements

1. Use EXACT signal names from module_header
2. Bit widths: `[m:n]` → width = m-n+1, no range → 1 bit
3. Output ONLY '0' and '1' binary strings - NEVER 'X', 'Z', 'd'
4. Always mask outputs: `result & ((1 << width) - 1)`
5. Multi-bit format: `format(result, f'0{{{{width}}}}b')`

## Implementation

```python
class GoldenDUT:
    def __init__(self):
        pass
    
    def load(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Parse inputs: data = int(inputs["data"], 2)
        # Return: {{"out": str(result)}} or {{"out": format(result, f'0{{width}}b')}}
        pass
```

**REMEMBER**: Output ONLY binary strings with '0' and '1'!
"""

CMB_PythonHeader = """
import json
import re
import random
import subprocess
import os
from typing import Dict, List, Union, Any

def extract_module_ports_with_yosys(verilog_file="top_module.v"):
    \"\"\"
    Use yosys to extract module port information (inputs/outputs with widths).
    Returns a dict with 'inputs' and 'outputs', each containing {port_name: width}.
    \"\"\"
    try:
        # Create a temporary yosys script
        yosys_script = f\"\"\"
read_verilog {verilog_file}
hierarchy -check
proc
opt_clean
write_json ports.json
\"\"\"
        with open("extract_ports.ys", "w") as f:
            f.write(yosys_script)

        # Run yosys
        result = subprocess.run(
            ["yosys", "-s", "extract_ports.ys"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Yosys extraction failed: {result.stderr}")
            return None

        # Parse the JSON output
        with open("ports.json", "r") as f:
            yosys_data = json.load(f)

        # Extract port information from the first module
        module_name = list(yosys_data["modules"].keys())[0]
        module_data = yosys_data["modules"][module_name]

        ports = {"inputs": {}, "outputs": {}}

        for port_name, port_info in module_data["ports"].items():
            direction = port_info["direction"]
            bits = port_info["bits"]
            width = len(bits)

            port_name_lower = port_name.lower()
            # Skip clock and reset signals
            if any(keyword in port_name_lower for keyword in ["clk", "clock", "rst", "reset", "areset"]):
                continue

            if direction == "input":
                ports["inputs"][port_name] = width
            elif direction == "output":
                ports["outputs"][port_name] = width

        return ports
    except Exception as e:
        print(f"Error extracting ports with yosys: {e}")
        return None

def generate_random_stimulus(ports_info, num_vectors=10):
    \"\"\"
    Generate random stimulus based on port information.
    Returns a list of test vectors (for combinational) or scenarios (for sequential).
    \"\"\"
    test_vectors = []

    for _ in range(num_vectors):
        vector = {}
        for port_name, width in ports_info["inputs"].items():
            # Generate random binary string of specified width
            random_value = random.randint(0, (1 << width) - 1)
            vector[port_name] = format(random_value, f'0{width}b')
        test_vectors.append(vector)

    return test_vectors

def verify_and_fix_stimulus(stimulus_data, verilog_file="top_module.v"):
    \"\"\"
    Verify stimulus.json matches module ports. If not, generate new stimulus.
    Returns the verified/corrected stimulus data.
    \"\"\"
    # Extract port information using yosys
    ports_info = extract_module_ports_with_yosys(verilog_file)

    if ports_info is None:
        print("Warning: Could not extract port information with yosys, using original stimulus")
        return stimulus_data

    # Get expected input port names (excluding clock/reset)
    expected_inputs = set(ports_info["inputs"].keys())

    if len(stimulus_data) == 0:
        print("Warning: stimulus.json is empty, generating random stimulus")
        return generate_random_stimulus(ports_info)

    # Check if stimulus keys match expected inputs
    actual_inputs = set(stimulus_data[0].keys()) - {"clock_cycles"}  # For sequential circuits

    if expected_inputs != actual_inputs:
        print(f"Mismatch detected!")
        print(f"Expected inputs from module: {sorted(expected_inputs)}")
        print(f"Actual inputs from stimulus.json: {sorted(actual_inputs)}")
        print(f"Generating new random stimulus based on module ports...")

        return generate_random_stimulus(ports_info, num_vectors=len(stimulus_data))
    else:
        print(f"Stimulus verification passed. All ports match: {sorted(expected_inputs)}")
        return stimulus_data
"""





CMB_CHECKER_TAIL = """
if __name__ == "__main__":
    import os
    import sys

    # Check if stimulus.json exists
    if not os.path.exists("stimulus.json"):
        print("Error: stimulus.json not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        sys.exit(1)

    # Load stimulus.json
    with open("stimulus.json", "r") as f:
        test_vectors = json.load(f)

    # Verify and fix stimulus if needed (using yosys)
    print("\\n=== Verifying stimulus against module ports ===")
    test_vectors = verify_and_fix_stimulus(test_vectors, verilog_file="top_module.v")
    print("==============================================\\n")

    dut = GoldenDUT()
    testbench = []

    for test_vector in test_vectors:
        try:
            output = dut.load(test_vector)
            testbench.append({
                "inputs": test_vector,
                "expected_outputs": output
            })
        except Exception as e:
            print(f"Error in test vector: {e}")
            testbench.append({
                "inputs": test_vector,
                "expected_outputs": {}
            })

    with open("testbench.json", "w") as f:
        json.dump(testbench, f, indent=2)

    print("Testbench generation successful")
"""
SEQ_CHECKER_TAIL = """
if __name__ == "__main__":
    import os
    import sys

    # Check if stimulus.json exists
    if not os.path.exists("stimulus.json"):
        print("Error: stimulus.json not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        sys.exit(1)

    # Load stimulus.json
    with open("stimulus.json", "r") as f:
        test_scenarios = json.load(f)

    # Verify and fix stimulus if needed (using yosys)
    print("\\n=== Verifying stimulus against module ports ===")
    test_scenarios = verify_and_fix_stimulus_seq(test_scenarios, verilog_file="top_module.v")
    print("==============================================\\n")

    testbench = []

    for scenario in test_scenarios:
        dut = GoldenDUT()

        clock_cycles = scenario.get("clock_cycles", 0)
        input_signals = {k: v for k, v in scenario.items() if k != "clock_cycles"}

        scenario_outputs = []
        for cycle in range(clock_cycles):
            cycle_inputs = {sig: vals[cycle] if cycle < len(vals) else "0" for sig, vals in input_signals.items()}

            cycle_output = {}
            try:
                rising_output = dut.load(1, cycle_inputs)
                cycle_output["rising_edge"] = rising_output
            except Exception as e:
                print(f"Error in rising edge cycle {cycle}: {e}")
                cycle_output["rising_edge"] = {}

            try:
                falling_output = dut.load(0, cycle_inputs)
                cycle_output["falling_edge"] = falling_output
            except Exception as e:
                print(f"Error in falling edge cycle {cycle}: {e}")
                cycle_output["falling_edge"] = {}

            scenario_outputs.append(cycle_output)

        test_case = {"clock_cycles": clock_cycles}
        for sig_name, sig_values in input_signals.items():
            test_case[sig_name] = sig_values
        test_case["expected_outputs"] = scenario_outputs
        testbench.append(test_case)

    with open("testbench.json", "w") as f:
        json.dump(testbench, f, indent=2)

    print("Testbench generation successful")
"""


# ============================================================================
# SEQ (Sequential) Circuit Configuration
# ============================================================================

SEQ_SYSTEM_PROMPT = """You are an expert in RTL design and Python programming. You can always write correct Python code to verify RTL functionality."""

SEQ_GENERATION_PROMPT = r"""

You are implementing a Python class "GoldenDUT" for sequential logic.

<description>
{description}
</description>

<module_header>
{module_header}
</module_header>

## Requirements

1. Use EXACT signal names from module_header (exclude 'clk')
2. Bit widths: `[m:n]` → width = m-n+1, no range → 1 bit
3. Output ONLY '0' and '1' binary strings - NEVER 'X', 'Z', 'd'
4. Always mask outputs: `result & ((1 << width) - 1)`
5. Multi-bit format: `format(result, f'0{{{{width}}}}b')`
6. Update state ONLY when `clk == 1` (rising edge)

## Implementation

```python
class GoldenDUT:
    def __init__(self):
        # Initialize state variables to 0
        pass
    
    def load(self, clk: int, inputs: Dict[str, str]) -> Dict[str, str]:
        # Parse inputs: data = int(inputs["data"], 2)
        # Update state on clk == 1
        # Return: {{"out": str(result)}} or {{"out": format(result, f'0{{width}}b')}}
        pass
```

**REMEMBER**: Output ONLY binary strings with '0' and '1'!
"""

SEQ_PythonHeader = """
import json
import re
import random
import subprocess
import os
from typing import Dict, List, Union

def extract_module_ports_with_yosys(verilog_file="top_module.v"):
    \"\"\"
    Use yosys to extract module port information (inputs/outputs with widths).
    Returns a dict with 'inputs' and 'outputs', each containing {port_name: width}.
    \"\"\"
    try:
        # Create a temporary yosys script
        yosys_script = f\"\"\"
read_verilog {verilog_file}
hierarchy -check
proc
opt_clean
write_json ports.json
\"\"\"
        with open("extract_ports.ys", "w") as f:
            f.write(yosys_script)

        # Run yosys
        result = subprocess.run(
            ["yosys", "-s", "extract_ports.ys"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"Yosys extraction failed: {result.stderr}")
            return None

        # Parse the JSON output
        with open("ports.json", "r") as f:
            yosys_data = json.load(f)

        # Extract port information from the first module
        module_name = list(yosys_data["modules"].keys())[0]
        module_data = yosys_data["modules"][module_name]

        ports = {"inputs": {}, "outputs": {}}

        for port_name, port_info in module_data["ports"].items():
            direction = port_info["direction"]
            bits = port_info["bits"]
            width = len(bits)

            port_name_lower = port_name.lower()
            # Skip clock and reset signals
            if any(keyword in port_name_lower for keyword in ["clk", "clock", "rst", "reset", "areset"]):
                continue

            if direction == "input":
                ports["inputs"][port_name] = width
            elif direction == "output":
                ports["outputs"][port_name] = width

        return ports
    except Exception as e:
        print(f"Error extracting ports with yosys: {e}")
        return None

def generate_random_stimulus_seq(ports_info, num_scenarios=5, cycles_per_scenario=10):
    \"\"\"
    Generate random stimulus scenarios for sequential circuits.
    Returns a list of scenarios with clock_cycles and input sequences.
    \"\"\"
    scenarios = []

    for _ in range(num_scenarios):
        scenario = {"clock_cycles": cycles_per_scenario}

        for port_name, width in ports_info["inputs"].items():
            # Generate random sequence for this input
            sequence = []
            for _ in range(cycles_per_scenario):
                random_value = random.randint(0, (1 << width) - 1)
                sequence.append(format(random_value, f'0{width}b'))
            scenario[port_name] = sequence

        scenarios.append(scenario)

    return scenarios

def verify_and_fix_stimulus_seq(stimulus_data, verilog_file="top_module.v"):
    \"\"\"
    Verify stimulus.json matches module ports for sequential circuits.
    If not, generate new stimulus.
    Returns the verified/corrected stimulus data.
    \"\"\"
    # Extract port information using yosys
    ports_info = extract_module_ports_with_yosys(verilog_file)

    if ports_info is None:
        print("Warning: Could not extract port information with yosys, using original stimulus")
        return stimulus_data

    # Get expected input port names (excluding clock/reset)
    expected_inputs = set(ports_info["inputs"].keys())

    if len(stimulus_data) == 0:
        print("Warning: stimulus.json is empty, generating random stimulus")
        return generate_random_stimulus_seq(ports_info)

    # Check if stimulus keys match expected inputs
    actual_inputs = set(stimulus_data[0].keys()) - {"clock_cycles"}

    if expected_inputs != actual_inputs:
        print(f"Mismatch detected!")
        print(f"Expected inputs from module: {sorted(expected_inputs)}")
        print(f"Actual inputs from stimulus.json: {sorted(actual_inputs)}")
        print(f"Generating new random stimulus based on module ports...")

        # Determine average cycles from original stimulus
        avg_cycles = sum(s.get("clock_cycles", 10) for s in stimulus_data) // len(stimulus_data)
        return generate_random_stimulus_seq(ports_info, num_scenarios=len(stimulus_data), cycles_per_scenario=avg_cycles)
    else:
        print(f"Stimulus verification passed. All ports match: {sorted(expected_inputs)}")
        return stimulus_data

"""
# =============================================================================
