#!/usr/bin/env python3
"""
Test script to verify automatic CPU allocation for PyChecker workers
"""

import os
import sys

# Add project path
sys.path.insert(0, '/home/lah003/workspace/verl_efficient')

def test_cpu_allocation():
    """Test the automatic CPU allocation logic"""

    print("=" * 80)
    print("Testing PyChecker Worker CPU Auto-Allocation")
    print("=" * 80)

    # Import after path is set
    from pettingllms.multi_agent_env.pychecker_rl.pychecker_worker import get_ray_pychecker_worker_cls
    import ray

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Get available resources
    resources = ray.available_resources()
    total_cpus = resources.get("CPU", 0)

    print(f"\n📊 Ray Cluster Resources:")
    print(f"   Total CPUs available: {total_cpus}")
    print(f"   Total GPUs available: {resources.get('GPU', 0)}")

    # Test different num_workers values
    test_cases = [
        ("Small", 100),
        ("Medium", 512),
        ("Large", 1000),
        ("Very Large", 1800),
    ]

    print(f"\n🧪 Testing CPU allocation for different worker counts:")
    print(f"{'Case':<15} {'Workers':<10} {'CPU/Worker':<15} {'Total CPU Used':<20} {'Efficiency':<15}")
    print("-" * 80)

    for case_name, num_workers in test_cases:
        # Clear cached class to test fresh calculation
        if hasattr(get_ray_pychecker_worker_cls, "_cls"):
            delattr(get_ray_pychecker_worker_cls, "_cls")

        # Test the allocation
        try:
            worker_cls = get_ray_pychecker_worker_cls(num_workers=num_workers)

            # Get the num_cpus from the remote decorator
            if hasattr(worker_cls, "_ray_metadata"):
                cpu_per_worker = worker_cls._ray_metadata.get("num_cpus", "unknown")
            else:
                cpu_per_worker = "unknown"

            # Calculate efficiency
            if isinstance(cpu_per_worker, (int, float)):
                total_cpu_used = cpu_per_worker * num_workers
                efficiency = (min(total_cpu_used, total_cpus) / total_cpus) * 100
                can_create = min(int(total_cpus / cpu_per_worker), num_workers)
            else:
                total_cpu_used = "unknown"
                efficiency = "unknown"
                can_create = "unknown"

            print(f"{case_name:<15} {num_workers:<10} {cpu_per_worker:<15.3f} {total_cpu_used:<20.2f} {efficiency:<15.1f}%")

            if isinstance(can_create, int):
                if can_create < num_workers:
                    print(f"   ⚠️  Warning: Can only create {can_create}/{num_workers} workers with available CPUs")
                else:
                    print(f"   ✅ Can create all {num_workers} workers")

        except Exception as e:
            print(f"{case_name:<15} {num_workers:<10} ERROR: {e}")

    print("\n" + "=" * 80)
    print("✅ CPU Allocation Test Complete")
    print("=" * 80)

    # Show calculation formula
    print("\n📐 Auto-calculation formula:")
    print(f"   num_cpus_per_worker = (total_cpus * 0.9) / num_workers")
    print(f"   Clamped to range: [0.1, 2.0]")
    print(f"\n💡 Example for {total_cpus} CPUs and 512 workers:")
    print(f"   num_cpus_per_worker = ({total_cpus} * 0.9) / 512 = {(total_cpus * 0.9) / 512:.3f}")
    print(f"   Maximum concurrent tasks = {int(total_cpus / ((total_cpus * 0.9) / 512))}")

    # Cleanup
    ray.shutdown()

if __name__ == "__main__":
    test_cpu_allocation()
