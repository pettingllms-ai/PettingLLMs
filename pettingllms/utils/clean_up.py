import atexit
import signal
import sys
import shutil
from pathlib import Path
import ray


_CLEANED = False
_TEMP_DIRS = []
_RAY_ACTORS = []


def register_temp_dirs(*dirs):
    """Register temporary directories to be cleaned up"""
    _TEMP_DIRS.extend(dirs)


def register_ray_actors(*actors):
    """Register Ray actors to be cleaned up"""
    _RAY_ACTORS.extend(actors)


def cleanup_ray():
    """Clean up Ray resources and temporary directories for current session"""
    global _CLEANED
    if _CLEANED:
        return
    _CLEANED = True

    print("\nCleaning up Ray session...")

    # Kill all registered Ray actors first
    if _RAY_ACTORS:
        print(f"Killing {len(_RAY_ACTORS)} registered Ray actors...")
        for actor in _RAY_ACTORS:
            try:
                if actor is not None:
                    ray.kill(actor)
                    print(f"  Killed actor: {actor}")
            except Exception as e:
                print(f"  Warning: Failed to kill actor {actor}: {e}")
        _RAY_ACTORS.clear()

    # Try to kill all running Ray actors
    if ray.is_initialized():
        try:
            print("Attempting to kill all Ray actors...")
            # Get all actor info
            actors = ray.state.actors()
            if actors:
                for actor_id, actor_info in actors.items():
                    try:
                        if actor_info.get('State') in ['ALIVE', 'PENDING']:
                            actor_handle = ray.get_actor(actor_info['Name']) if 'Name' in actor_info else None
                            if actor_handle:
                                ray.kill(actor_handle)
                                print(f"  Killed actor: {actor_info.get('Name', actor_id)}")
                    except Exception as e:
                        print(f"  Warning: Failed to kill actor {actor_id}: {e}")
            print(f"Killed {len(actors) if actors else 0} Ray actors")
        except Exception as e:
            print(f"Warning: Error while killing Ray actors: {e}")

        # Shutdown Ray
        print("Shutting down Ray...")
        ray.shutdown()

    # Clean up temporary directories
    for temp_dir in _TEMP_DIRS:
        if Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                print(f"Removed temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to remove {temp_dir}: {e}")

    print("Cleanup completed\n")


def install_cleanup_hooks():
    """Install cleanup hooks for normal exit and signals"""
    atexit.register(cleanup_ray)
    
    def signal_handler(signum, frame):
        cleanup_ray()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

