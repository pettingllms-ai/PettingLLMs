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
    Returns dict with 'inputs' and 'outputs' lists.
    """
    try:
        with open(verilog_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        logger.warning(f"Verilog file '{verilog_file}' not found, cannot extract signals")
        return {"inputs": [], "outputs": []}

    signals = {"inputs": {}, "outputs": {}}

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

        # Skip clock and reset signals
        if name.lower() in ['clk', 'clock', 'rst', 'reset', 'rstn', 'rst_n']:
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

    logger.info(f"Extracted signals - inputs: {signals['inputs']}, outputs: {signals['outputs']}")
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
            # Remove all occurrences of '0x'

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

    # Map testbench signals to actual signals
    # Process even if inputs are empty (e.g., for modules with only outputs)
    if len(datas) > 0 and (len(actual_signals["inputs"]) > 0 or len(actual_signals["outputs"]) > 0):
        corrected_datas = []
        for data in datas:
            corrected_data = {"inputs": {}, "expected_outputs": {}}

            # Process input signals (if any)
            for test_signal, value in data.get("inputs", {}).items():
                if len(actual_signals["inputs"]) > 0:
                    # Check if this signal is actually an output in Verilog (common testbench bug)
                    if test_signal in actual_signals["outputs"]:
                        logger.warning(f"Signal '{test_signal}' is in testbench inputs but is actually a Verilog output, skipping")
                        continue

                    matched_signal = fuzzy_match_signal(test_signal, actual_signals["inputs"])
                    if matched_signal:
                        corrected_data["inputs"][matched_signal] = value
                    else:
                        # Try fuzzy matching against outputs (in case testbench has wrong classification)
                        matched_output = fuzzy_match_signal(test_signal, actual_signals["outputs"])
                        if matched_output:
                            logger.warning(f"Signal '{test_signal}' matched to output '{matched_output}', not using as input")
                        # Check if it's a clock/reset signal that was filtered out
                        elif test_signal.lower() in ['clk', 'clock', 'rst', 'reset', 'rstn', 'rst_n']:
                            logger.info(f"Signal '{test_signal}' is a clock/reset signal (handled by simulation framework)")
                        else:
                            logger.warning(f"Input signal '{test_signal}' not matched to any Verilog signal, skipping")
                else:
                    logger.warning(f"No input signals in Verilog, skipping testbench input '{test_signal}'")

            # If we have actual input signals not covered, add random values
            for actual_input, width in actual_signals["inputs"].items():
                if actual_input not in corrected_data["inputs"]:
                    random_val = generate_random_value(width)
                    corrected_data["inputs"][actual_input] = random_val
                    logger.info(f"Added random value for unmatched input '{actual_input}': {random_val}")

            # Process output signals with fuzzy matching
            for test_signal, value in data.get("expected_outputs", {}).items():
                if len(actual_signals["outputs"]) > 0:
                    matched_signal = fuzzy_match_signal(test_signal, actual_signals["outputs"])
                    if matched_signal:
                        corrected_data["expected_outputs"][matched_signal] = value
                    else:
                        logger.warning(f"Output signal '{test_signal}' not matched to any actual output, skipping")
                else:
                    logger.warning(f"No output signals in Verilog, skipping testbench output '{test_signal}'")

            # Only add test case if we have some expected outputs
            if corrected_data["expected_outputs"]:
                corrected_datas.append(corrected_data)
            else:
                logger.warning(f"Skipping test case with no matched expected outputs")

        datas = corrected_datas
        logger.info(f"Signal mapping complete: {len(datas)} test vectors after correction")

    ###############################################
    # Generate Harness with JSON testbench (Combinational Logic)
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

    # Determine wide signals based on ACTUAL VERILOG WIDTHS, not testbench values
    # This fixes the VlWide type mismatch errors
    wide_input_signals = {}
    wide_output_signals = {}
    declared_wide_signals = set()

    # Check input signals from Verilog definition
    for name, width in actual_signals["inputs"].items():
        if width > 64:  # Signals > 64 bits need VlWide
            n_words = (width + 31) // 32
            wide_input_signals[name] = n_words
            if name not in declared_wide_signals:
                cpp_code += f"""    VlWide<{n_words}> {name}_wide;\n"""
                declared_wide_signals.add(name)

    # Check output signals from Verilog definition
    for name, width in actual_signals["outputs"].items():
        if width > 64:  # Signals > 64 bits need VlWide
            n_words = (width + 31) // 32
            wide_output_signals[name] = n_words
            if name not in declared_wide_signals:
                cpp_code += f"""    VlWide<{n_words}> {name}_wide;\n"""
                declared_wide_signals.add(name)
    
    cpp_code += "\n"
    
    # Create DUT instance
    cpp_code += """    // Create DUT instance
    const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};
    const std::unique_ptr<Vtop_module> top{new Vtop_module};
    
"""
    
    # Generate test logic for each test vector
    test_idx = 0
    for data in datas:
        inputs = data["inputs"]
        expected_outputs = data["expected_outputs"]
        test_name = f"test_{test_idx}"
        
        cpp_code += f"""    ///////////////////////////////////////////////////////////\n"""
        cpp_code += f"""    // Test vector {test_idx}\n"""
        cpp_code += f"""    ///////////////////////////////////////////////////////////\n"""
        cpp_code += f"""    printf("\\n========== Test Vector {test_idx} ==========\\n");\n"""
        cpp_code += f"""    \n"""
        
        # Set input signals
        cpp_code += f"""    // Set inputs\n"""
        for name, value in inputs.items():
            temp = sanitize_value(value)

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
                    cpp_code += f"""    top->{name} = (unsigned char)(0x{hex_value});\n"""
                elif actual_width <= 16:
                    cpp_code += f"""    top->{name} = (unsigned short)(0x{hex_value});\n"""
                elif actual_width <= 32:
                    cpp_code += f"""    top->{name} = (unsigned int)(0x{hex_value});\n"""
                else:  # 33-64 bits
                    cpp_code += f"""    top->{name} = (unsigned long long)(0x{hex_value});\n"""
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
                    cpp_code += f"""    top->{name}[{j}] = {name}_wide[{j}];\n"""
        
        # Evaluate
        cpp_code += f"""    \n"""
        cpp_code += f"""    // Evaluate\n"""
        cpp_code += f"""    top->eval();\n"""
        cpp_code += f"""    \n"""
        
        # Check outputs
        cpp_code += f"""    // Check outputs\n"""
        for name, value in expected_outputs.items():
            temp = sanitize_value(value)

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
                # Expected: from JSON testbench data, Actual: from simulation
                cpp_code += f"""    printf("  Output {name}: expected(from JSON)=0x{hex_value}, actual(from sim)=0x%llx\\n", (unsigned long long)top->{name});\n"""
                cpp_code += f"""    if (top->{name} != 0x{hex_value}) {{\n"""
                cpp_code += f"""        unpass++;\n"""
                cpp_code += f"""        unpass_total++;\n"""
                cpp_code += f"""        printf("  [FAIL] Mismatch at {name}\\n");\n"""
                cpp_code += f"""    }} else {{\n"""
                cpp_code += f"""        printf("  [PASS] {name} matched\\n");\n"""
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
                cpp_code += f"""    printf("  Output {name} (wide):\\n");\n"""
                cpp_code += f"""    bool {name}_match_{test_idx} = true;\n"""
                for k in range(n_words):
                    cpp_code += f"""    printf("    [{k}] expected(from JSON)=0x%08X, actual(from sim)=0x%08X\\n", {name}_wide[{k}], top->{name}[{k}]);\n"""
                    cpp_code += f"""    if (top->{name}[{k}] != {name}_wide[{k}]) {name}_match_{test_idx} = false;\n"""

                cpp_code += f"""    if (!{name}_match_{test_idx}) {{\n"""
                cpp_code += f"""        unpass++;\n"""
                cpp_code += f"""        unpass_total++;\n"""
                cpp_code += f"""        printf("  [FAIL] Mismatch at {name}\\n");\n"""
                cpp_code += f"""    }} else {{\n"""
                cpp_code += f"""        printf("  [PASS] {name} matched\\n");\n"""
                cpp_code += f"""    }}\n"""
        
        cpp_code += f"""    \n"""
        test_idx += 1

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
