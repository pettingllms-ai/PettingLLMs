#!/usr/bin/env python3
"""
Test script for cleanup_old_image_folders function

This script creates a test directory structure and tests the cleanup functionality.
"""

from pathlib import Path
import shutil
from pettingllms.utils.clean_up import cleanup_old_image_folders


def create_test_structure(base_dir: str = "test_tmp_image"):
    """
    Create a test directory structure with multiple step folders

    Structure:
        test_tmp_image/
        ├── 20241217/
        │   ├── experiment_1/
        │   │   ├── step_0/
        │   │   ├── step_1/
        │   │   ├── ...
        │   │   └── step_25/  (26 folders total)
        │   └── experiment_2/
        │       ├── step_0/
        │       ├── ...
        │       └── step_15/  (16 folders total)
        └── 20241218/
            └── experiment_3/
                ├── step_0/
                └── step_5/  (6 folders total)
    """
    print(f"\n[Test] Creating test directory structure at '{base_dir}'...")

    base_path = Path(base_dir)

    # Clean up if exists
    if base_path.exists():
        shutil.rmtree(base_path)

    # Create experiment 1: 26 step folders (should trigger cleanup)
    exp1_path = base_path / "20241217" / "experiment_1"
    for i in range(26):
        step_dir = exp1_path / f"step_{i}" / "rollout_0"
        step_dir.mkdir(parents=True, exist_ok=True)
        # Create a dummy file
        (step_dir / "dummy.txt").write_text(f"Step {i} data")

    # Create experiment 2: 16 step folders (should not trigger cleanup with default max=20)
    exp2_path = base_path / "20241217" / "experiment_2"
    for i in range(16):
        step_dir = exp2_path / f"step_{i}" / "rollout_0"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "dummy.txt").write_text(f"Step {i} data")

    # Create experiment 3: 6 step folders (should not trigger cleanup)
    exp3_path = base_path / "20241218" / "experiment_3"
    for i in range(6):
        step_dir = exp3_path / f"step_{i}" / "rollout_0"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "dummy.txt").write_text(f"Step {i} data")

    print("[Test] Test directory structure created successfully!")
    return base_path


def count_step_folders(base_dir: Path):
    """Count step folders in each experiment"""
    results = {}

    for date_folder in base_dir.iterdir():
        if not date_folder.is_dir():
            continue

        for experiment_folder in date_folder.iterdir():
            if not experiment_folder.is_dir():
                continue

            step_count = 0
            step_folders = []
            for item in experiment_folder.iterdir():
                if item.is_dir() and item.name.startswith("step_"):
                    step_count += 1
                    step_folders.append(item.name)

            exp_name = f"{date_folder.name}/{experiment_folder.name}"
            results[exp_name] = {
                "count": step_count,
                "folders": sorted(step_folders)
            }

    return results


def test_cleanup():
    """Test the cleanup_old_image_folders function"""
    print("\n" + "="*70)
    print("Testing cleanup_old_image_folders function")
    print("="*70)

    # Create test structure
    test_base_dir = "test_tmp_image"
    base_path = create_test_structure(test_base_dir)

    # Count folders before cleanup
    print("\n[Test] Folder counts BEFORE cleanup:")
    before_counts = count_step_folders(base_path)
    for exp_name, data in before_counts.items():
        print(f"  {exp_name}: {data['count']} step folders")
        print(f"    Range: {data['folders'][0]} to {data['folders'][-1]}")

    # Test 1: Default cleanup (max_subfolders=20)
    print("\n" + "-"*70)
    print("[Test 1] Testing with default max_subfolders=20")
    print("-"*70)
    deleted_count = cleanup_old_image_folders(
        base_dir=test_base_dir,
        max_subfolders=20,
        verbose=True
    )

    # Count folders after cleanup
    print("\n[Test] Folder counts AFTER cleanup (max=20):")
    after_counts = count_step_folders(base_path)
    for exp_name, data in after_counts.items():
        before_count = before_counts[exp_name]['count']
        after_count = data['count']
        deleted = before_count - after_count
        print(f"  {exp_name}: {after_count} step folders (deleted {deleted})")
        if data['folders']:
            print(f"    Range: {data['folders'][0]} to {data['folders'][-1]}")

    # Verify results
    print("\n" + "-"*70)
    print("[Test] Verification:")
    print("-"*70)

    # Experiment 1: Should have 20 folders (deleted 6)
    exp1_count = after_counts["20241217/experiment_1"]["count"]
    exp1_folders = after_counts["20241217/experiment_1"]["folders"]
    print(f"✓ Experiment 1: {exp1_count} folders (expected 20)")
    assert exp1_count == 20, f"Expected 20 folders, got {exp1_count}"
    assert exp1_folders[0] == "step_6", f"Expected oldest kept folder to be step_6, got {exp1_folders[0]}"
    assert exp1_folders[-1] == "step_25", f"Expected newest folder to be step_25, got {exp1_folders[-1]}"
    print(f"  ✓ Kept: {exp1_folders[0]} to {exp1_folders[-1]}")

    # Experiment 2: Should have 16 folders (no deletion)
    exp2_count = after_counts["20241217/experiment_2"]["count"]
    print(f"✓ Experiment 2: {exp2_count} folders (expected 16, no deletion)")
    assert exp2_count == 16, f"Expected 16 folders, got {exp2_count}"

    # Experiment 3: Should have 6 folders (no deletion)
    exp3_count = after_counts["20241218/experiment_3"]["count"]
    print(f"✓ Experiment 3: {exp3_count} folders (expected 6, no deletion)")
    assert exp3_count == 6, f"Expected 6 folders, got {exp3_count}"

    print("\n[Test] All verification checks passed! ✓")

    # Test 2: Custom max_subfolders
    print("\n" + "-"*70)
    print("[Test 2] Testing with custom max_subfolders=10")
    print("-"*70)

    # Re-create test structure
    shutil.rmtree(base_path)
    create_test_structure(test_base_dir)

    deleted_count = cleanup_old_image_folders(
        base_dir=test_base_dir,
        max_subfolders=10,
        verbose=True
    )

    after_counts = count_step_folders(base_path)
    exp1_count = after_counts["20241217/experiment_1"]["count"]
    exp2_count = after_counts["20241217/experiment_2"]["count"]

    print(f"\n[Test] Results with max=10:")
    print(f"  Experiment 1: {exp1_count} folders (expected 10, deleted 16)")
    print(f"  Experiment 2: {exp2_count} folders (expected 10, deleted 6)")

    assert exp1_count == 10, f"Expected 10 folders, got {exp1_count}"
    assert exp2_count == 10, f"Expected 10 folders, got {exp2_count}"

    print("\n[Test] Custom max_subfolders test passed! ✓")

    # Cleanup test directory
    print("\n" + "-"*70)
    print("[Test] Cleaning up test directory...")
    print("-"*70)
    shutil.rmtree(base_path)
    print("[Test] Test directory removed successfully!")

    print("\n" + "="*70)
    print("All tests passed! ✓✓✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_cleanup()
