"""
Process cleanup improvements for pychecker_worker.py

This patch adds proper process cleanup mechanisms to prevent zombie processes
from consuming system memory during Verilator compilation.

Key improvements:
1. Use process groups to kill all child processes
2. Force kill remaining processes after timeout
3. Add cleanup handlers for abnormal termination
"""

import subprocess
import signal
import os
import time
import psutil


def run_with_cleanup(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=None):
    """
    Run subprocess with proper cleanup of child processes on timeout.

    Args:
        cmd: Command to execute
        shell: Whether to use shell
        capture_output: Whether to capture output
        text: Whether to use text mode
        timeout: Timeout in seconds
        cwd: Working directory

    Returns:
        subprocess.CompletedProcess object
    """
    process = None
    try:
        # Create new process group to track all children
        process = subprocess.Popen(
            cmd,
            shell=shell,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=text,
            cwd=cwd,
            preexec_fn=os.setpgrp  # Create new process group
        )

        # Wait for completion with timeout
        stdout, stderr = process.communicate(timeout=timeout)

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr
        )

    except subprocess.TimeoutExpired:
        # Kill the entire process group
        if process:
            try:
                # Get process group ID
                pgid = os.getpgid(process.pid)
                # Kill entire process group (including cc1plus children)
                os.killpg(pgid, signal.SIGTERM)

                # Wait briefly for graceful termination
                time.sleep(1)

                # Force kill if still alive
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already terminated

            except (ProcessLookupError, PermissionError) as e:
                print(f"Warning: Failed to kill process group: {e}")

        # Try to get partial output
        try:
            stdout, stderr = process.communicate(timeout=1)
        except:
            stdout, stderr = "", ""

        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout=stdout if stdout else "",
            stderr=f"Process timeout after {timeout} seconds" + (f"\n{stderr}" if stderr else "")
        )

    except Exception as e:
        # Clean up on any other exception
        if process:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except:
                pass
        raise


def cleanup_simulation_processes():
    """
    Clean up any lingering simulation processes (verilator, cc1plus).

    This should be called periodically or at cleanup time.
    """
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            name = proc.info['name']
            cmdline = ' '.join(proc.info['cmdline'] or [])

            # Kill verilator compilation processes
            if name in ['cc1plus', 'cc1', 'verilator', 'g++']:
                # Only kill if it's related to our simulation (check for rfuzz-harness)
                if 'rfuzz-harness' in cmdline or 'top_module' in cmdline:
                    print(f"Killing lingering process: PID={proc.pid}, name={name}")
                    proc.kill()
                    killed_count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if killed_count > 0:
        print(f"Cleaned up {killed_count} lingering compilation processes")

    return killed_count


# Example usage in simulate_dut_seq_ray and simulate_dut_cmb_ray:
"""
Replace this:
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=SEQ_SIM_TIMEOUT
    )

With this:
    result = run_with_cleanup(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=SEQ_SIM_TIMEOUT,
        cwd=work_dir
    )
"""
