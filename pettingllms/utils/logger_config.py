import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import is_dataclass, asdict

def safe_serialize(obj):
    """Safely serialize objects, handle OmegaConf, dataclasses and other non-serializable objects"""
    # First, try to handle OmegaConf objects (including ListConfig, DictConfig)
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
    except (ImportError, Exception):
        pass
    
    # Handle dataclass objects
    if is_dataclass(obj):
        try:
            # Convert dataclass to dict, then recursively serialize
            dataclass_dict = asdict(obj)
            return safe_serialize(dataclass_dict)
        except Exception:
            # If asdict fails, fall back to a custom representation
            return {
                "__dataclass__": obj.__class__.__name__,
                "__fields__": {
                    field: safe_serialize(getattr(obj, field, None))
                    for field in obj.__dataclass_fields__.keys()
                }
            }
    
    # Handle basic Python types
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items()}
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # For complex objects, try JSON serialization test first
        try:
            json.dumps(obj)  # Test if serializable
            return obj
        except (TypeError, ValueError):
            # If not serializable, convert to string
            return str(obj)

class MultiLoggerConfig:
    """
    Multi-logger system configuration, supports creating different types of loggers
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "default"):
        """
        Initialize multi-logger configuration
        
        Args:
            log_dir: Log file storage directory
        """
        # Create directory structure with date only
        current_time = datetime.now()
        date_folder = current_time.strftime("%m-%d")
        timestamp_folder = current_time.strftime("%H-%M-%S")
        
        self.log_dir = Path(log_dir) / experiment_name / date_folder / timestamp_folder
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger dictionary
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Create four main loggers
        self._setup_env_agent_logger()
        self._setup_model_logger()
        self._setup_async_logger()
        self._setup_summary_logger()
    
    def _setup_env_agent_logger(self):
        """Setup base env_agent logger without file handler (handlers are per-rollout)."""
        logger_name = "env_agent"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        self.loggers[logger_name] = logger
    
    def _setup_model_logger(self):
        """Setup base model logger without file handler (handlers are per-rollout)."""
        logger_name = "model"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        self.loggers[logger_name] = logger
    
    def _setup_async_logger(self):
        """Setup base async logger without file handler (handlers are per-rollout)."""
        logger_name = "async"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        self.loggers[logger_name] = logger

    def _get_or_create_rollout_logger(self, base_name: str, mode: str, env_idx: int, rollout_idx: int) -> logging.Logger:
        """Create or get a per-rollout logger writing under date/time/env_idx/rollout_idx/{base_name}.log"""
        logger_key = f"{base_name}:{mode}:{env_idx}:{rollout_idx}"
        if logger_key in self.loggers:
            return self.loggers[logger_key]

        logger = logging.getLogger(logger_key)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False

        # Create hierarchical directory structure: env_idx/rollout_idx/
        env_dir = self.log_dir / str(mode) / str(env_idx)
        rollout_dir = env_dir / str(rollout_idx)
        rollout_dir.mkdir(parents=True, exist_ok=True)

        file_path = rollout_dir / f"{base_name}.log"
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        if base_name == "env_agent":
            formatter = logging.Formatter(
                '[%(asctime)s] [ENV:%(env_idx)s] [ROLLOUT:%(rollout_idx)s] [TURN:%(turn_idx)s] [AGENT:%(agent_name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        elif base_name == "model":
            formatter = logging.Formatter(
                '[%(asctime)s] [ENV:%(env_idx)s] [ROLLOUT:%(rollout_idx)s] [POLICY:%(policy_name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:  # async
            formatter = logging.Formatter(
                '[%(asctime)s] [ENV:%(env_idx)s] [ROLLOUT:%(rollout_idx)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        self.loggers[logger_key] = logger
        return logger
    
    def _setup_summary_logger(self):
        """Setup summary.log logger"""
        logger_name = "summary"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(
            self.log_dir / "summary.log", 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Set format
        formatter = logging.Formatter(
            '[%(asctime)s] [ROLLOUT:%(rollout_idx)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        self.loggers[logger_name] = logger
    
    def log_env_agent_info(self, mode: str, env_idx: int, rollout_idx: int, turn_idx: int, agent_name: str, 
                          message: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Log environment and agent related information
        
        Args:
            env_idx: Environment index
            rollout_idx: Rollout index
            turn_idx: Turn index
            agent_name: Agent name
            message: Log message
            extra_data: Additional structured data
        """
        logger = self._get_or_create_rollout_logger("env_agent", mode, env_idx, rollout_idx)

        # Build log content and safely serialize
        log_content = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "extra_data": safe_serialize(extra_data or {})
        }
        
        # Use extra parameter to pass context information
        extra = {
            "env_idx": env_idx,
            "rollout_idx": rollout_idx,
            "turn_idx": turn_idx,
            "agent_name": agent_name
        }
        
        logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def log_model_interaction(self, mode: str, env_idx: int, rollout_idx: int, policy_name: str, 
                            prompt: str, response: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Log model interaction information
        
        Args:
            env_idx: Environment index
            rollout_idx: Rollout index
            policy_name: Policy name
            prompt: Input prompt
            response: Model response
            extra_data: Additional data
        """
        logger = self._get_or_create_rollout_logger("model",mode, env_idx, rollout_idx)

        log_content = {
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "extra_data": safe_serialize(extra_data or {})
        }
        extra = {
            "env_idx": env_idx if env_idx is not None else "N/A",
            "rollout_idx": rollout_idx if rollout_idx is not None else "N/A",
            "policy_name": policy_name
        }
        
        logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def log_async_event(self, mode: str, env_idx: int, rollout_idx: int, event_type: str, 
                       message: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Log asynchronous execution events
        
        Args:
            env_idx: Environment index  
            rollout_idx: Rollout index
            event_type: Event type (start, complete, error, etc.)
            message: Event message
            extra_data: Additional data
        """
        if rollout_idx == -1 or env_idx == -1:
            # For global events, log to summary
            logger = self.loggers["summary"]
            extra = {
                "rollout_idx": rollout_idx if rollout_idx != -1 else "GLOBAL"
            }
        else:
            logger = self._get_or_create_rollout_logger("async", mode, env_idx, rollout_idx)
            extra = {
                "env_idx": env_idx,
                "rollout_idx": rollout_idx
            }

        log_content = {
            "event_type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "extra_data": safe_serialize(extra_data or {})
        }
        
        logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def _check_ray_status(self) -> Dict[str, Any]:
        """
        Check Ray initialization and launch status
        
        Returns:
            Dictionary containing Ray status information
        """
        ray_status = {
            "ray_available": False,
            "ray_initialized": False,
            "ray_address": None,
            "ray_namespace": None,
            "ray_cluster_resources": None,
            "ray_nodes": None,
            "ray_error": None
        }
        
        try:
            import ray
            ray_status["ray_available"] = True
            
            if ray.is_initialized():
                ray_status["ray_initialized"] = True
                
                try:
                    # Get cluster information
                    cluster_resources = ray.cluster_resources()
                    ray_status["ray_cluster_resources"] = safe_serialize(cluster_resources)
                    
                    # Get current address and namespace
                    context = ray.get_runtime_context()
                    ray_status["ray_address"] = getattr(context, 'gcs_address', None)
                    ray_status["ray_namespace"] = getattr(context, 'namespace', None)
                    
                    # Get cluster nodes information
                    try:
                        nodes = ray.nodes()
                        ray_status["ray_nodes"] = len(nodes)
                    except Exception as e:
                        ray_status["ray_nodes"] = f"Error getting nodes: {str(e)}"
                        
                except Exception as e:
                    ray_status["ray_error"] = f"Error getting cluster info: {str(e)}"
            else:
                ray_status["ray_initialized"] = False
                
        except ImportError:
            ray_status["ray_available"] = False
            ray_status["ray_error"] = "Ray not installed"
        except Exception as e:
            ray_status["ray_error"] = f"Error checking Ray status: {str(e)}"
            
        return ray_status
    
    def log_rollout_summary(self, mode: str, env_idx: int, rollout_idx: int, agent_rewards: Dict[str, float], 
                           termination_reason: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Log rollout summary information with Ray status check
        
        Args:
            rollout_idx: Rollout index
            agent_rewards: Dictionary mapping agent names to their final rewards
            termination_reason: Reason for rollout termination
            extra_data: Additional data
        """
        logger = self.loggers["summary"]
        
        log_content = {
            "env_idx": env_idx,
            "rollout_idx": rollout_idx,
            "agent_rewards": agent_rewards,
            "termination_reason": termination_reason,
            "timestamp": datetime.now().isoformat(),
            "extra_data": safe_serialize(extra_data or {})
        }
        
        extra = {
            "rollout_idx": rollout_idx
        }
        
        logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def log_ray_status(self, mode: str = "train", rollout_idx: Optional[int] = None, context: str = "general"):
        """
        Log detailed Ray status information
        
        Args:
            rollout_idx: Optional rollout index for context
            context: Context description for the Ray status check
        """
        logger = self.loggers["summary"]
        ray_status = self._check_ray_status()
        
        log_content = {
            "event_type": "ray_status_check",
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "ray_status": ray_status
        }
        
        extra = {
            "rollout_idx": rollout_idx if rollout_idx is not None else "N/A"
        }
        
        # Log with appropriate level based on Ray status
        if not ray_status["ray_available"]:
            logger.warning(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
        elif not ray_status["ray_initialized"]:
            logger.warning(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
        elif ray_status["ray_error"]:
            logger.error(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
        else:
            logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def log_error(self, mode: str, env_idx: Optional[int], rollout_idx: Optional[int], error_source: str, 
                         error: Exception, context_data: Optional[Dict[str, Any]] = None,
                         severity: str = "ERROR", additional_info: Optional[Dict[str, Any]] = None):
        """
        Log complex error information with detailed context and stack trace
        
        Args:
            env_idx: Environment index (can be None for non-rollout errors)
            rollout_idx: Rollout index (can be None for non-rollout errors)
            error_source: Source of the error (e.g., 'env_agent', 'model', 'async', 'system')
            error: The exception object
            context_data: Contextual data when the error occurred
            severity: Error severity level (ERROR, CRITICAL, WARNING)
            additional_info: Additional information about the error
        """
        import traceback
        import sys
        
        # Determine which logger to use based on error source
        if error_source in ["env_agent", "model", "async"]:
            if rollout_idx is not None and env_idx is not None:
                logger = self._get_or_create_rollout_logger(error_source,mode, env_idx, rollout_idx)
            else:
                # Use base logger if no rollout context
                logger = self.loggers.get(error_source, self.loggers["summary"])
        else:
            # For system errors or unknown sources, use summary logger
            logger = self.loggers["summary"]
        
        # Get full stack trace
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
        
        # Build comprehensive error log content
        error_content = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_source": error_source,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "stack_trace": stack_trace,
            "context_data": safe_serialize(context_data or {}),
            "additional_info": safe_serialize(additional_info or {}),
            "python_version": sys.version,
            "platform_info": {
                "platform": sys.platform,
                "implementation": sys.implementation.name if hasattr(sys, 'implementation') else 'unknown'
            }
        }
        
        # Add rollout context if available
        if rollout_idx is not None:
            error_content["rollout_idx"] = rollout_idx
        if env_idx is not None:
            error_content["env_idx"] = env_idx
        
        # Prepare extra parameters for logging
        extra = {
            "env_idx": env_idx if env_idx is not None else "N/A",
            "rollout_idx": rollout_idx if rollout_idx is not None else "N/A",
            "error_source": error_source,
            "severity": severity
        }
        
        # Log with appropriate level
        if severity == "CRITICAL":
            logger.critical(json.dumps(error_content, ensure_ascii=False, indent=2), extra=extra)
        elif severity == "WARNING":
            logger.warning(json.dumps(error_content, ensure_ascii=False, indent=2), extra=extra)
        else:
            logger.error(json.dumps(error_content, ensure_ascii=False, indent=2), extra=extra)
        
        # Also log a simplified version to summary logger for overview
        summary_content = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_source": error_source,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "env_idx": env_idx if env_idx is not None else "N/A",
            "rollout_idx": rollout_idx if rollout_idx is not None else "N/A"
        }
        
        summary_logger = self.loggers["summary"]
        summary_extra = {
            "env_idx": env_idx if env_idx is not None else "N/A",
            "rollout_idx": rollout_idx if rollout_idx is not None else "N/A"
        }
        
        if severity == "CRITICAL":
            summary_logger.critical(json.dumps(summary_content, ensure_ascii=False, indent=2), extra=summary_extra)
        elif severity == "WARNING":
            summary_logger.warning(json.dumps(summary_content, ensure_ascii=False, indent=2), extra=summary_extra)
        else:
            summary_logger.error(json.dumps(summary_content, ensure_ascii=False, indent=2), extra=summary_extra)
    
    def get_logger(self, logger_name: str) -> Optional[logging.Logger]:
        """Get specified logger"""
        return self.loggers.get(logger_name)

# Global logger configuration instance
_global_logger_config = None

def get_multi_logger(experiment_name: str = "default", log_dir: str = "logs") -> MultiLoggerConfig:
    # Ensure correct parameter binding; previous version passed experiment_name as log_dir by position
    return MultiLoggerConfig(log_dir=log_dir, experiment_name=experiment_name)
