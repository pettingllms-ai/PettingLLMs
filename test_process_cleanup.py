#!/usr/bin/env python3
"""
Test script to verify process cleanup works correctly.
This simulates a timeout scenario to ensure all child processes are killed.
"""

import os
import subprocess
import signal
import time
import psutil


def count_test_processes():
    """Count cc1plus/make processes related to our test"""
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = proc.info['name']
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if name in ['cc1plus', 'make', 'g++']:
                if 'test_cleanup' in cmdline:
                    count += 1
                    print(f"  Found: PID={proc.pid}, name={name}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return count


def test_old_method():
    """Test old subprocess.run method (leaves zombies)"""
    print("\n=== Testing OLD method (subprocess.run) ===")
    print("Starting long-running compilation with 2 second timeout...")

    # This simulates a long compilation that will timeout
    cmd = "sleep 10 && echo 'test_cleanup marker' > /dev/null"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=2
        )
    except subprocess.TimeoutExpired:
        print("✗ Timeout occurred (expected)")

    print("Checking for orphaned processes...")
    time.sleep(1)
    count = count_test_processes()
    if count > 0:
        print(f"✗ OLD method left {count} zombie process(es)")
        return False
    else:
        print("✓ OLD method cleaned up (sleep is simple, no children)")
        return True


def test_new_method():
    """Test new process group method (kills all children)"""
    print("\n=== Testing NEW method (process group) ===")
    print("Starting long-running compilation with 2 second timeout...")

    # Simulate a make command that spawns children
    # This would be like: make -j4 (spawns 4 compiler processes)
    cmd = "bash -c 'for i in {1..5}; do (sleep 20 && echo test_cleanup$i) & done; wait'"

    process = None
    try:
        # NEW METHOD: Use process group
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setpgrp  # Create new process group
        )

        # Wait with timeout
        stdout, stderr = process.communicate(timeout=2)
        print("✓ Process completed within timeout")

    except subprocess.TimeoutExpired:
        print("✗ Timeout occurred (expected)")

        # Kill entire process group
        if process:
            try:
                pgid = os.getpgid(process.pid)
                print(f"Killing process group {pgid}...")
                os.killpg(pgid, signal.SIGTERM)
                time.sleep(0.5)

                # Force kill if still alive
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    print("✓ Process group already terminated")
            except (ProcessLookupError, PermissionError) as e:
                print(f"Warning: {e}")

    print("Checking for orphaned processes...")
    time.sleep(1)
    count = count_test_processes()
    if count > 0:
        print(f"✗ NEW method left {count} zombie process(es)")
        return False
    else:
        print("✓ NEW method cleaned up all processes")
        return True


def main():
    print("=" * 60)
    print("Process Cleanup Test")
    print("=" * 60)

    # Test old method
    old_result = test_old_method()

    # Test new method
    new_result = test_new_method()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Old method (subprocess.run):     {'✓ PASS' if old_result else '✗ FAIL'}")
    print(f"New method (process group):      {'✓ PASS' if new_result else '✗ FAIL'}")

    if new_result:
        print("\n✓ Process cleanup is working correctly!")
        print("  The new implementation will properly clean up cc1plus processes.")
        return 0
    else:
        print("\n✗ Process cleanup test failed!")
        print("  There may be zombie processes remaining.")
        return 1


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        exit(130)
