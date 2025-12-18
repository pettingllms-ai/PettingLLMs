import atexit
import signal
import sys
import shutil
import subprocess
from pathlib import Path
import ray


_CLEANED = False
_TEMP_DIRS = []


def register_temp_dirs(*dirs):
    """Register temporary directories to be cleaned up"""
    _TEMP_DIRS.extend(dirs)


def kill_ray_processes():
    """Kill all Ray-related processes"""
    patterns = [
        "ray::",
        "raylet",
        "gcs_server",
        "plasma_store",
        "default_worker.py",
        "worker.py",
    ]
    for pattern in patterns:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass


def cleanup_ray():
    """Clean up Ray resources and temporary directories for current session"""
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True
    
    print("\nCleaning up Ray session...")
    
    if ray.is_initialized():
        ray.shutdown()
    
    for temp_dir in _TEMP_DIRS:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"Removed temporary directory: {temp_dir}")
    
    print("Cleanup completed\n")


def cleanup_ray_runtime():
    """Complete Ray cleanup including process termination"""
    cleanup_ray()
    kill_ray_processes()


def install_cleanup_hooks():
    """Install cleanup hooks for normal exit and signals"""
    atexit.register(cleanup_ray)

    def signal_handler(signum, frame):
        cleanup_ray()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def cleanup_old_image_folders(base_dir: str = "tmp_image", max_subfolders: int = 20, verbose: bool = True):
    """
    Clean up old image folders when the number of subfolders exceeds the limit.

    The function checks each experiment folder (e.g., tmp_image/20241217/experiment_name/)
    and if it contains more than max_subfolders step folders, it deletes the oldest ones.

    Args:
        base_dir: Base directory for images (default: "tmp_image")
        max_subfolders: Maximum number of step subfolders to keep (default: 20)
        verbose: Whether to print cleanup information (default: True)

    Example directory structure:
        tmp_image/
        ├── 20241217/
        │   ├── experiment_1/
        │   │   ├── step_0/    <- Will be deleted if total > 20
        │   │   ├── step_1/    <- Will be deleted if total > 20
        │   │   ├── ...
        │   │   ├── step_19/   <- Kept
        │   │   └── step_20/   <- Kept (newest)
        │   └── experiment_2/
        │       └── ...
        └── 20241218/
            └── ...

    Returns:
        Number of folders deleted
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        return 0

    total_deleted = 0

    # Iterate through date folders (e.g., 20241217)
    for date_folder in base_path.iterdir():
        if not date_folder.is_dir():
            continue

        # Iterate through experiment folders (e.g., experiment_name)
        for experiment_folder in date_folder.iterdir():
            if not experiment_folder.is_dir():
                continue

            # Get all step folders (e.g., step_0, step_1, ...)
            step_folders = []
            for item in experiment_folder.iterdir():
                if item.is_dir() and item.name.startswith("step_"):
                    try:
                        # Extract step number for sorting
                        step_num = int(item.name.split("_")[1])
                        step_folders.append((step_num, item))
                    except (IndexError, ValueError):
                        # Skip folders that don't match the pattern
                        continue

            # Check if we need to clean up
            num_steps = len(step_folders)
            if num_steps > max_subfolders:
                # Sort by step number (oldest first)
                step_folders.sort(key=lambda x: x[0])

                # Calculate how many to delete
                num_to_delete = num_steps - max_subfolders
                folders_to_delete = step_folders[:num_to_delete]

                if verbose:
                    print(f"[Cleanup] Found {num_steps} step folders in {experiment_folder.relative_to(base_path)}")
                    print(f"[Cleanup] Deleting {num_to_delete} oldest folders...")

                # Delete old folders
                for step_num, folder in folders_to_delete:
                    try:
                        shutil.rmtree(folder, ignore_errors=True)
                        total_deleted += 1
                        if verbose:
                            print(f"[Cleanup] Deleted: {folder.relative_to(base_path)}")
                    except Exception as e:
                        if verbose:
                            print(f"[Cleanup] Failed to delete {folder}: {e}")

    if verbose and total_deleted > 0:
        print(f"[Cleanup] Total folders deleted: {total_deleted}")

    return total_deleted

