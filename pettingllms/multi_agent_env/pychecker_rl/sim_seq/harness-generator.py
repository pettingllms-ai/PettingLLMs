# Analyse the DUT verilog files
# 1> generate rfuzz harness to fuzzing method
# 2> generate explicit signal prompt to LLM method

from __future__ import absolute_import, print_function

import json
import os
import sys
import logging

# the next line can be removed after installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# import pyverilog
# from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer

import re
import random


def extract_module_signals(verilog_file):
    """
    Extract input and output signal names from Verilog module.
    Returns dict with 'inputs', 'outputs', and 'clock_name' (actual clock signal name).
    """
    try:
        with open(verilog_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        logger.warning(f"Verilog file '{verilog_file}' not found, cannot extract signals")
        return {"inputs": {}, "outputs": {}, "clock_name": "clk"}

    signals = {"inputs": {}, "outputs": {}, "clock_name": "clk"}  # default to 'clk'

    # Extract module declaration
    module_match = re.search(r'module\s+\w+\s*\((.*?)\);', content, re.DOTALL)
    if not module_match:
        logger.warning("Could not find module declaration")
        return signals

    port_list = module_match.group(1)

    # Extract input signals with widths: input [width:0] name or input name
    # Pattern matches: input [MSB:LSB] signal1, signal2, signal3
    # Also matches: input wire/reg/logic [MSB:LSB] name
    # The pattern now properly handles wire/reg/logic keywords and only captures the signal name
    input_pattern = r'input\s+(?:(?:wire|reg|logic)\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)'
    input_matches = re.findall(input_pattern, port_list)

    for match in input_matches:
        msb, lsb, name = match

        # Detect and save clock signal name (but don't add to regular inputs)
        if name.lower() in ['clk', 'clock', 'rst', 'reset', 'rstn', 'rst_n']:
            if name.lower() in ['clk', 'clock']:
                signals["clock_name"] = name  # Save actual clock signal name
                logger.info(f"Detected clock signal name: '{name}'")
            continue
        if msb and lsb:
            width = int(msb) - int(lsb) + 1
        else:
            width = 1
        signals["inputs"][name] = width

    # Extract output signals with widths
    # Pattern matches: output [MSB:LSB] signal1, signal2, signal3
    # Also matches: output wire/reg/logic [MSB:LSB] name
    # The pattern now properly handles wire/reg/logic keywords and only captures the signal name
    output_pattern = r'output\s+(?:(?:wire|reg|logic)\s+)?(?:\[\s*(\d+)\s*:\s*(\d+)\s*\])?\s*(\w+)'
    output_matches = re.findall(output_pattern, port_list)

    for match in output_matches:
        msb, lsb, name = match

        if msb and lsb:
            width = int(msb) - int(lsb) + 1
        else:
            width = 1
        signals["outputs"][name] = width

    logger.info(f"Extracted signals - inputs: {signals['inputs']}, outputs: {signals['outputs']}, clock: {signals['clock_name']}")
    return signals


def fuzzy_match_signal(test_name, actual_signals):
    """
    Fuzzy match test signal name to actual signal name.
    Returns the matched actual signal name or None.

    Matching priority:
    1. Exact match (case-sensitive)
    2. Case-insensitive exact match
    3. Common name transformations (underscores, suffixes)
    4. Abbreviation/alias mapping
    5. Conservative substring match (only if very similar)
    """
    # Priority 1: Exact match
    if test_name in actual_signals:
        return test_name

    # Priority 2: Case-insensitive match
    for actual in actual_signals:
        if test_name.lower() == actual.lower():
            logger.info(f"Case-insensitive match: '{test_name}' -> '{actual}'")
            return actual

    # Priority 3: Common name transformations
    # Try with/without underscores, common suffixes
    test_variations = [
        test_name.replace('_', ''),
        test_name + '_in',
        test_name + '_out',
        test_name.rstrip('_in'),
        test_name.rstrip('_out'),
    ]
    for variation in test_variations:
        for actual in actual_signals:
            if variation.lower() == actual.lower():
                logger.info(f"Name variation match: '{test_name}' -> '{actual}' (via '{variation}')")
                return actual

    # Priority 4: Abbreviation/alias mapping (BIDIRECTIONAL)
    # Map common test names to actual signal names and vice versa
    abbreviation_map = {
        # testbench name -> possible actual names
        'data_in': ['data', 'din', 'd', 'in'],
        'data_out': ['data', 'dout', 'q', 'out'],
        'din': ['data_in', 'data', 'd', 'in'],
        'dout': ['data_out', 'data', 'q', 'out'],
        'load': ['load', 'ld', 'en', 'enable'],
        'ena': ['enable', 'en', 'ena'],
        'enable': ['ena', 'en', 'enable'],
        # actual name -> possible testbench names (reverse mapping)
        'data': ['data_in', 'din', 'data_out', 'dout'],
        'q': ['data_out', 'dout', 'out'],
        'd': ['data_in', 'din', 'data'],
        'in': ['data_in', 'din'],
        'out': ['data_out', 'dout'],
    }

    # Try test_name -> actual
    if test_name.lower() in abbreviation_map:
        for candidate in abbreviation_map[test_name.lower()]:
            for actual in actual_signals:
                if candidate.lower() == actual.lower():
                    logger.info(f"Abbreviation match (test->actual): '{test_name}' -> '{actual}'")
                    return actual

    # Try actual -> test_name (reverse lookup)
    for actual in actual_signals:
        if actual.lower() in abbreviation_map:
            for candidate in abbreviation_map[actual.lower()]:
                if candidate.lower() == test_name.lower():
                    logger.info(f"Abbreviation match (actual->test): '{test_name}' -> '{actual}'")
                    return actual

    # Priority 5: Conservative substring match
    # Only match if one is a clear suffix/prefix of the other AND length difference is small
    for actual in actual_signals:
        test_lower = test_name.lower()
        actual_lower = actual.lower()

        # Check if one is a substring of the other
        if test_lower in actual_lower or actual_lower in test_lower:
            # Only accept if length ratio is reasonable (avoid matching 'a' to 'data')
            min_len = min(len(test_lower), len(actual_lower))
            max_len = max(len(test_lower), len(actual_lower))

            # Require at least 60% length similarity
            if min_len / max_len >= 0.6:
                logger.info(f"Conservative substring match: '{test_name}' -> '{actual}' (ratio: {min_len/max_len:.2f})")
                return actual

    return None


def generate_random_value(width):
    """Generate a random binary string of specified width."""
    if width <= 0:
        return "0"
    value = random.getrandbits(width)
    return format(value, f'0{width}b')


def sanitize_value(value):
    """
    Sanitize input/output values to ensure they are valid binary strings.
    Handles cases where values might have hex prefixes or malformed formats.
    """
    temp = str(value).strip()
    
    # Remove any whitespace
    temp = temp.replace(" ", "")
    
    # Check if it looks like a hex value (starts with 0x or 0X)
    if temp.lower().startswith("0x"):
        # Remove the 0x prefix
        hex_str = temp[2:]
        # Remove any duplicate 'x' characters that might have been added
        hex_str = hex_str.lstrip('xX')

        if not hex_str:
            hex_str = "0"

        try:
            # Convert hex to int, then to binary string (without '0b' prefix)
            int_val = int(hex_str, 16)
            binary_str = bin(int_val)[2:]
            logger.warning(f"Detected hex value '{temp}', converted to binary: '{binary_str}'")
            return binary_str
        except ValueError as e:
            logger.error(f"Failed to convert hex value '{temp}': {e}")
            # Try to clean up malformed hex values like "0x0x1" or "0xxff5"
            # Remove all 'x' and '0x' patterns more aggressively
            cleaned = temp.lower().replace('0x', '')
           

            if cleaned:
                try:
                    int_val = int(cleaned, 16)
                    binary_str = bin(int_val)[2:]
                    logger.warning(f"Cleaned malformed hex '{temp}' to '{cleaned}', binary: '{binary_str}'")
                    return binary_str
                except ValueError:
                    pass
    
    # Check if it's already a valid binary string (only 0s and 1s)
    if all(c in '01' for c in temp):
        return temp
    
    # If it contains only hex characters, try to interpret as hex
    if all(c in '0123456789abcdefABCDEF' for c in temp):
        try:
            int_val = int(temp, 16)
            binary_str = bin(int_val)[2:]
            logger.warning(f"Interpreted '{temp}' as hex, converted to binary: '{binary_str}'")
            return binary_str
        except ValueError:
            pass
    
    # Last resort: return as-is and let it fail with a clear error
    logger.error(f"Could not sanitize value: '{value}' -> '{temp}'")
    return temp


def main():

    test_file = "testbench.json"
    datas = []
    try:
        with open(test_file, "r") as f:
            datas = json.load(f)
    except json.JSONDecodeError:
        try:
            with open(test_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        datas.append(data)
        except Exception as e:
            logger.error(f"Error reading JSON file: {e}")
            return

    # Extract actual signal names from Verilog module
    actual_signals = extract_module_signals("top_module.v")
    clock_signal_name = actual_signals.get("clock_name", "clk")

    # Verify clock signal name exists in Verilog (case-sensitive check)
    # Read Verilog file to find the actual clock signal declaration
    try:
        with open("top_module.v", 'r') as f:
            verilog_content = f.read()

        # Search for clock signal in module ports (case-sensitive)
        # Look for patterns like: input clk, input Clk, input clock, etc.
        clock_patterns = [
            r'\binput\s+(?:wire\s+)?(\w*[Cc][Ll][Kk]\w*)\b',
            r'\binput\s+(?:wire\s+)?(\w*[Cc][Ll][Oo][Cc][Kk]\w*)\b',
        ]

        found_clock = None
        for pattern in clock_patterns:
            matches = re.findall(pattern, verilog_content)
            if matches:
                found_clock = matches[0]
                break

        if found_clock:
            clock_signal_name = found_clock
            logger.info(f"Detected clock signal name from Verilog: '{clock_signal_name}'")
        else:
            logger.warning(f"Could not find clock signal in Verilog, using default: '{clock_signal_name}'")
    except Exception as e:
        logger.warning(f"Error reading Verilog file for clock detection: {e}, using: '{clock_signal_name}'")

    # Map testbench signals to actual signals and fix clock_cycles
    if len(datas) > 0 and len(actual_signals["inputs"]) > 0:
        corrected_datas = []
        for data_idx, data in enumerate(datas):
            if "clock_cycles" not in data:
                logger.warning(f"Scenario {data_idx} missing 'clock_cycles', skipping")
                continue

            original_clock_cycles = data["clock_cycles"]
            corrected_data = {"clock_cycles": original_clock_cycles, "expected_outputs": []}

            # Get input signals (excluding clock_cycles and expected_outputs)
            # IMPORTANT FIX: Filter out signals that are actually outputs in Verilog
            input_signals = {}
            for k, v in data.items():
                if k in ["clock_cycles", "expected_outputs"]:
                    continue
                # Check if this signal is actually an output in Verilog (common testbench bug)
                if k in actual_signals["outputs"]:
                    logger.warning(f"Signal '{k}' is in testbench inputs but is actually a Verilog output, skipping from inputs")
                    continue
                input_signals[k] = v

            # Find minimum signal length to fix IndexError
            min_signal_length = original_clock_cycles
            for test_signal, values in input_signals.items():
                if isinstance(values, list):
                    if len(values) < min_signal_length:
                        logger.warning(f"Signal '{test_signal}' has {len(values)} values but clock_cycles is {original_clock_cycles}")
                        min_signal_length = len(values)

            # Adjust clock_cycles to minimum signal length
            if min_signal_length < original_clock_cycles:
                logger.warning(f"Adjusting clock_cycles from {original_clock_cycles} to {min_signal_length}")
                corrected_data["clock_cycles"] = min_signal_length

            # Process input signals with fuzzy matching
            for test_signal, values in input_signals.items():
                matched_signal = fuzzy_match_signal(test_signal, actual_signals["inputs"])
                if matched_signal:
                    # Truncate to corrected clock_cycles length
                    if isinstance(values, list):
                        corrected_data[matched_signal] = values[:corrected_data["clock_cycles"]]
                    else:
                        corrected_data[matched_signal] = values
                else:
                    # Try fuzzy matching against outputs (in case testbench has wrong classification)
                    matched_output = fuzzy_match_signal(test_signal, actual_signals["outputs"])
                    if matched_output:
                        logger.warning(f"Signal '{test_signal}' matched to output '{matched_output}', not using as input")
                    else:
                        # Check if it's a clock/reset signal that was filtered out
                        if test_signal.lower() in ['clk', 'clock', 'rst', 'reset', 'rstn', 'rst_n']:
                            logger.info(f"Signal '{test_signal}' is a clock/reset signal (handled by simulation framework)")
                        else:
                            logger.warning(f"Input signal '{test_signal}' not matched to any Verilog signal, will use random values")

            # Add random values for unmatched actual input signals
            for actual_input, width in actual_signals["inputs"].items():
                if actual_input not in corrected_data:
                    random_vals = [generate_random_value(width) for _ in range(corrected_data["clock_cycles"])]
                    corrected_data[actual_input] = random_vals
                    logger.info(f"Added random values for unmatched input '{actual_input}'")

            # Process expected outputs with fuzzy matching
            if "expected_outputs" in data and isinstance(data["expected_outputs"], list):
                for cycle_idx, cycle_outputs in enumerate(data["expected_outputs"][:corrected_data["clock_cycles"]]):
                    if not isinstance(cycle_outputs, dict):
                        logger.warning(f"Expected output for cycle {cycle_idx} is not a dict, skipping (type={type(cycle_outputs).__name__})")
                        continue

                    corrected_cycle = {}

                    # Handle rising_edge outputs
                    rising_outputs = cycle_outputs.get("rising_edge")
                    if isinstance(rising_outputs, dict):
                        corrected_cycle["rising_edge"] = {}
                        for test_signal, value in rising_outputs.items():
                            matched_signal = fuzzy_match_signal(test_signal, actual_signals["outputs"])
                            if matched_signal:
                                corrected_cycle["rising_edge"][matched_signal] = value
                            else:
                                logger.warning(f"Output signal '{test_signal}' not matched at cycle {cycle_idx}, skipping")
                    elif rising_outputs not in (None, {}):
                        logger.warning(f"Unexpected rising_edge format at cycle {cycle_idx}, skipping (type={type(rising_outputs).__name__})")

                    # Handle falling_edge outputs
                    falling_outputs = cycle_outputs.get("falling_edge")
                    if isinstance(falling_outputs, dict):
                        corrected_cycle["falling_edge"] = {}
                        for test_signal, value in falling_outputs.items():
                            matched_signal = fuzzy_match_signal(test_signal, actual_signals["outputs"])
                            if matched_signal:
                                corrected_cycle["falling_edge"][matched_signal] = value
                            else:
                                logger.warning(f"Output signal '{test_signal}' not matched at cycle {cycle_idx}, skipping")
                    elif falling_outputs not in (None, {}):
                        logger.warning(f"Unexpected falling_edge format at cycle {cycle_idx}, skipping (type={type(falling_outputs).__name__})")

                    if corrected_cycle:
                        corrected_data["expected_outputs"].append(corrected_cycle)

            # Only add scenario if we have some expected outputs
            if corrected_data["expected_outputs"]:
                corrected_datas.append(corrected_data)

        datas = corrected_datas
        logger.info(f"Signal mapping complete: {len(datas)} scenarios after correction")


    ###############################################
    # Generate Harness with JSON testbench (Sequential Logic)
    ###############################################
    cpp_code = """
#include "rfuzz-harness.h"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <verilated.h>
#include "Vtop_module.h"

int fuzz_poke() {
    int unpass_total = 0;
    int unpass = 0;
    
"""

    # Collect all wide signals (input and output) from first data
    # Determine wide signals based on ACTUAL VERILOG WIDTHS, not testbench values
    # This fixes the VlWide type mismatch errors (same fix as sim_cmb)
    wide_input_signals = {}
    wide_output_signals = {}

    # Check input signals from Verilog definition
    for name, width in actual_signals["inputs"].items():
        if width > 64:  # Signals > 64 bits need VlWide
            n_words = (width + 31) // 32
            wide_input_signals[name] = n_words
            cpp_code += f"""    VlWide<{n_words}> {name}_wide;\n"""

    # Check output signals from Verilog definition
    for name, width in actual_signals["outputs"].items():
        if width > 64:  # Signals > 64 bits need VlWide
            n_words = (width + 31) // 32
            wide_output_signals[name] = n_words
            cpp_code += f"""    VlWide<{n_words}> {name}_wide;\n"""
    
    cpp_code += "\n"
    # Generate test logic for each scenario
    scenario_idx = 0
    for data in datas:
        clock_cycles = data["clock_cycles"]
        expected_outputs = data["expected_outputs"]
        scenario_name = f"scenario_{scenario_idx}"
        
        # Get input signals (excluding clock_cycles and expected_outputs)
        input_signals = {k: v for k, v in data.items() if k not in ["clock_cycles", "expected_outputs"]}
        
        cpp_code += f"""    ///////////////////////////////////////////////////////////\n"""
        cpp_code += f"""    // Scenario: {scenario_name}\n"""
        cpp_code += f"""    ///////////////////////////////////////////////////////////\n"""
        cpp_code += f"""    printf("\\n========== Testing Scenario: {scenario_name} ==========\\n");\n"""
        cpp_code += f"""    unpass = 0;\n"""
        cpp_code += f"""    \n"""
        
        # Create new instance for this scenario (to reset state)
        cpp_code += f"""    // Create new instance for scenario {scenario_name}\n"""
        cpp_code += f"""    const std::unique_ptr<VerilatedContext> contextp_{scenario_idx}{{new VerilatedContext}};\n"""
        cpp_code += f"""    const std::unique_ptr<Vtop_module> top_{scenario_idx}{{new Vtop_module}};\n"""
        cpp_code += f"""    auto* contextp_{scenario_idx}_ptr = contextp_{scenario_idx}.get();\n"""
        cpp_code += f"""    auto* top_{scenario_idx}_ptr = top_{scenario_idx}.get();\n"""
        cpp_code += f"""    \n"""
        
        # Initialize all signals to 0
        cpp_code += f"""    // Initialize clock to 0\n"""
        cpp_code += f"""    top_{scenario_idx}_ptr->{clock_signal_name} = 0;\n"""
        cpp_code += f"""    \n"""
        
        # Initialize all input signals to 0 (or first cycle values if available)
        cpp_code += f"""    // Initialize all input signals to 0\n"""
        for name in input_signals.keys():
            actual_width = actual_signals["inputs"].get(name, 1)
            if actual_width <= 64:
                cpp_code += f"""    top_{scenario_idx}_ptr->{name} = 0;\n"""
            else:
                n_words = wide_input_signals.get(name, (actual_width + 31) // 32)
                for j in range(n_words):
                    cpp_code += f"""    top_{scenario_idx}_ptr->{name}[{j}] = 0;\n"""
        
        cpp_code += f"""    top_{scenario_idx}_ptr->eval();\n"""
        cpp_code += f"""    \n"""
        
        # Process each clock cycle
        for cycle in range(clock_cycles):
            cpp_code += f"""    \n"""
            cpp_code += f"""    // Clock cycle {cycle}\n"""
            cpp_code += f"""    printf("--- Cycle {cycle} ---\\n");\n"""
            
            # Set input signals at clock low (before rising edge)
            cpp_code += f"""    // Set inputs while clock is low\n"""
            for name, values in input_signals.items():
                temp = sanitize_value(values[cycle])
                
                # Skip inputs with 'x', 'z', or other non-binary values
                if any(c not in '01' for c in temp.lower()):
                    cpp_code += f"""    printf("  Input {name} = 0b{temp} (skipped - contains x/z/unknown)\\n");\n"""
                    continue

                # Get actual width from Verilog definition
                actual_width = actual_signals["inputs"].get(name, len(temp))

                # Pad or truncate temp to match actual width
                if len(temp) < actual_width:
                    temp = temp.zfill(actual_width)
                elif len(temp) > actual_width:
                    temp = temp[-actual_width:]  # Take rightmost bits

                hex_len = (len(temp) + 3) // 4
                hex_value = hex(int(temp, 2))[2:].zfill(hex_len)

                cpp_code += f"""    printf("  Input {name} = 0b{temp}\\n");\n"""

                # Decide based on ACTUAL VERILOG WIDTH, not testbench value length
                if actual_width <= 64:
                    if actual_width <= 8:
                        cpp_code += f"""    top_{scenario_idx}_ptr->{name} = (unsigned char)(0x{hex_value});\n"""
                    elif actual_width <= 16:
                        cpp_code += f"""    top_{scenario_idx}_ptr->{name} = (unsigned short)(0x{hex_value});\n"""
                    elif actual_width <= 32:
                        cpp_code += f"""    top_{scenario_idx}_ptr->{name} = (unsigned int)(0x{hex_value});\n"""
                    else:  # 33-64 bits
                        cpp_code += f"""    top_{scenario_idx}_ptr->{name} = (unsigned long long)(0x{hex_value});\n"""
                else:
                    # > 64 bits: Must use VlWide
                    n_words = wide_input_signals[name]  # Already computed based on actual width

                    padded = temp.zfill(n_words * 32)
                    chunks = [
                        int(padded[-32 * (j + 1): -32 * j or None], 2)
                        for j in range(n_words)
                    ]
                    for j, c in enumerate(chunks):
                        cpp_code += f"""    {name}_wide[{j}] = 0x{c:08X}u;\n"""
                        cpp_code += f"""    top_{scenario_idx}_ptr->{name}[{j}] = {name}_wide[{j}];\n"""
            
            # Evaluate with inputs set and clock low
            cpp_code += f"""    \n"""
            cpp_code += f"""    // Evaluate with inputs set (clock still low)\n"""
            cpp_code += f"""    top_{scenario_idx}_ptr->eval();\n"""
            cpp_code += f"""    \n"""
            
            # Rising edge: clk 0->1
            cpp_code += f"""    // Rising edge: 0->1\n"""
            cpp_code += f"""    top_{scenario_idx}_ptr->{clock_signal_name} = 1;\n"""
            cpp_code += f"""    top_{scenario_idx}_ptr->eval();\n"""
            cpp_code += f"""    contextp_{scenario_idx}_ptr->timeInc(1);\n"""
            cpp_code += f"""    \n"""
            
            # Check outputs after rising edge
            if "rising_edge" in expected_outputs[cycle]:
                cpp_code += f"""    // Check outputs after rising edge\n"""
                for name, value in expected_outputs[cycle]["rising_edge"].items():
                    temp = sanitize_value(value)
                    
                    # Skip outputs with 'x', 'z', or other non-binary values
                    if any(c not in '01' for c in temp.lower()):
                        cpp_code += f"""    printf("  Rising edge output {name}: expected=0b{temp} (skipped - contains x/z/unknown)\\n");\n"""
                        continue

                    # Get actual width from Verilog definition
                    actual_width = actual_signals["outputs"].get(name, len(temp))

                    # Pad or truncate temp to match actual width
                    if len(temp) < actual_width:
                        temp = temp.zfill(actual_width)
                    elif len(temp) > actual_width:
                        temp = temp[-actual_width:]  # Take rightmost bits

                    hex_len = (len(temp) + 3) // 4
                    hex_value = hex(int(temp, 2))[2:].zfill(hex_len)

                    # Decide based on ACTUAL VERILOG WIDTH, not testbench value length
                    if actual_width <= 64:
                        # Regular output signal
                        # Expected: from JSON testbench data, Actual: from simulation
                        cpp_code += f"""    printf("  Rising edge output {name}: expected(from JSON)=0x{hex_value}, actual(from sim)=0x%llx\\n", (unsigned long long)top_{scenario_idx}_ptr->{name});\n"""
                        cpp_code += f"""    if (top_{scenario_idx}_ptr->{name} != 0x{hex_value}) {{\n"""
                        cpp_code += f"""        unpass++;\n"""
                        cpp_code += f"""        printf("  [FAIL] Mismatch at {name} (rising edge) in cycle {cycle}\\n");\n"""
                        cpp_code += f"""    }} else {{\n"""
                        cpp_code += f"""        printf("  [PASS] {name} matched (rising edge)\\n");\n"""
                        cpp_code += f"""    }}\n"""
                    else:
                        # > 64 bits: Must use VlWide
                        n_words = wide_output_signals[name]  # Already computed based on actual width

                        padded = temp.zfill(n_words * 32)
                        chunks = [
                            int(padded[-32 * (k + 1): -32 * k or None], 2)
                            for k in range(n_words)
                        ]

                        # Set expected values from JSON testbench data
                        for k, c in enumerate(chunks):
                            cpp_code += f"""    {name}_wide[{k}] = 0x{c:08X}u;\n"""

                        # Compare: expected (from JSON) vs actual (from simulation)
                        cpp_code += f"""    printf("  Rising edge output {name} (wide):\\n");\n"""
                        cpp_code += f"""    bool {name}_rising_match_s{scenario_idx}_c{cycle} = true;\n"""
                        for k in range(n_words):
                            cpp_code += f"""    printf("    [{k}] expected(from JSON)=0x%08X, actual(from sim)=0x%08X\\n", {name}_wide[{k}], top_{scenario_idx}_ptr->{name}[{k}]);\n"""
                            cpp_code += f"""    if (top_{scenario_idx}_ptr->{name}[{k}] != {name}_wide[{k}]) {name}_rising_match_s{scenario_idx}_c{cycle} = false;\n"""

                        cpp_code += f"""    if (!{name}_rising_match_s{scenario_idx}_c{cycle}) {{\n"""
                        cpp_code += f"""        unpass++;\n"""
                        cpp_code += f"""        printf("  [FAIL] Mismatch at {name} (rising edge) in cycle {cycle}\\n");\n"""
                        cpp_code += f"""    }} else {{\n"""
                        cpp_code += f"""        printf("  [PASS] {name} matched (rising edge)\\n");\n"""
                        cpp_code += f"""    }}\n"""
            
            # Falling edge: clk 1->0
            cpp_code += f"""    \n"""
            cpp_code += f"""    // Falling edge\n"""
            cpp_code += f"""    top_{scenario_idx}_ptr->{clock_signal_name} = 0;\n"""
            cpp_code += f"""    top_{scenario_idx}_ptr->eval();\n"""
            cpp_code += f"""    contextp_{scenario_idx}_ptr->timeInc(1);\n"""
            cpp_code += f"""    \n"""
            
            # Check outputs after falling edge
            if "falling_edge" in expected_outputs[cycle]:
                cpp_code += f"""    // Check outputs after falling edge\n"""
                for name, value in expected_outputs[cycle]["falling_edge"].items():
                    temp = sanitize_value(value)
                    
                    # Skip outputs with 'x', 'z', or other non-binary values
                    if any(c not in '01' for c in temp.lower()):
                        cpp_code += f"""    printf("  Falling edge output {name}: expected=0b{temp} (skipped - contains x/z/unknown)\\n");\n"""
                        continue

                    # Get actual width from Verilog definition
                    actual_width = actual_signals["outputs"].get(name, len(temp))

                    # Pad or truncate temp to match actual width
                    if len(temp) < actual_width:
                        temp = temp.zfill(actual_width)
                    elif len(temp) > actual_width:
                        temp = temp[-actual_width:]  # Take rightmost bits

                    hex_len = (len(temp) + 3) // 4
                    hex_value = hex(int(temp, 2))[2:].zfill(hex_len)

                    # Decide based on ACTUAL VERILOG WIDTH, not testbench value length
                    if actual_width <= 64:
                        # Regular output signal
                        # Expected: from JSON testbench data, Actual: from simulation
                        cpp_code += f"""    printf("  Falling edge output {name}: expected(from JSON)=0x{hex_value}, actual(from sim)=0x%llx\\n", (unsigned long long)top_{scenario_idx}_ptr->{name});\n"""
                        cpp_code += f"""    if (top_{scenario_idx}_ptr->{name} != 0x{hex_value}) {{\n"""
                        cpp_code += f"""        unpass++;\n"""
                        cpp_code += f"""        printf("  [FAIL] Mismatch at {name} (falling edge) in cycle {cycle}\\n");\n"""
                        cpp_code += f"""    }} else {{\n"""
                        cpp_code += f"""        printf("  [PASS] {name} matched (falling edge)\\n");\n"""
                        cpp_code += f"""    }}\n"""
                    else:
                        # > 64 bits: Must use VlWide
                        n_words = wide_output_signals[name]  # Already computed based on actual width

                        padded = temp.zfill(n_words * 32)
                        chunks = [
                            int(padded[-32 * (k + 1): -32 * k or None], 2)
                            for k in range(n_words)
                        ]

                        # Set expected values from JSON testbench data
                        for k, c in enumerate(chunks):
                            cpp_code += f"""    {name}_wide[{k}] = 0x{c:08X}u;\n"""

                        # Compare: expected (from JSON) vs actual (from simulation)
                        cpp_code += f"""    printf("  Falling edge output {name} (wide):\\n");\n"""
                        cpp_code += f"""    bool {name}_falling_match_s{scenario_idx}_c{cycle} = true;\n"""
                        for k in range(n_words):
                            cpp_code += f"""    printf("    [{k}] expected(from JSON)=0x%08X, actual(from sim)=0x%08X\\n", {name}_wide[{k}], top_{scenario_idx}_ptr->{name}[{k}]);\n"""
                            cpp_code += f"""    if (top_{scenario_idx}_ptr->{name}[{k}] != {name}_wide[{k}]) {name}_falling_match_s{scenario_idx}_c{cycle} = false;\n"""

                        cpp_code += f"""    if (!{name}_falling_match_s{scenario_idx}_c{cycle}) {{\n"""
                        cpp_code += f"""        unpass++;\n"""
                        cpp_code += f"""        printf("  [FAIL] Mismatch at {name} (falling edge) in cycle {cycle}\\n");\n"""
                        cpp_code += f"""    }} else {{\n"""
                        cpp_code += f"""        printf("  [PASS] {name} matched (falling edge)\\n");\n"""
                        cpp_code += f"""    }}\n"""
        
        # Summary for this scenario
        cpp_code += f"""
    // Scenario summary
    if (unpass == 0) {{
        std::cout << "✓ All tests passed for scenario {scenario_name}" << std::endl;
    }} else {{
        std::cout << "✗ Test failed with " << unpass << " error(s) for scenario {scenario_name}" << std::endl;
        unpass_total += unpass;
    }}
    std::cout << std::endl;

"""
        scenario_idx += 1

    # Final summary
    cpp_code += """
    // Final test summary
    if (unpass_total == 0) {
        std::cout << "========================================" << std::endl;
        std::cout << "✓ All scenarios passed!" << std::endl;
        std::cout << "========================================" << std::endl;
    } else {
        std::cout << "========================================" << std::endl;
        std::cout << "✗ Total failures: " << unpass_total << std::endl;
        std::cout << "========================================" << std::endl;
    }
    
    return unpass_total;
}
"""

    with open("rfuzz-harness.cpp", "w") as file:
        file.write(cpp_code)
    
    logger.info("Generated rfuzz-harness.cpp successfully!")


if __name__ == "__main__":
    main()
