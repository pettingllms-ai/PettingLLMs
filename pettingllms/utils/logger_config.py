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
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize multi-logger configuration
        
        Args:
            log_dir: Log file storage directory
        """
        # Create directory structure with date and timestamp
        current_time = datetime.now()
        date_folder = current_time.strftime("%Y-%m-%d")
        timestamp_folder = current_time.strftime("%H-%M-%S")
        
        self.log_dir = Path(log_dir) / date_folder / timestamp_folder
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger dictionary
        self.loggers: Dict[str, logging.Logger] = {}
        
        # Create three main loggers
        self._setup_env_agent_logger()
        self._setup_model_logger()
        self._setup_async_logger()
    
    def _setup_env_agent_logger(self):
        """Setup env_agent.log logger"""
        logger_name = "env_agent"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(
            self.log_dir / "env_agent.log", 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Set format
        formatter = logging.Formatter(
            '[%(asctime)s] [ROLLOUT:%(rollout_idx)s] [TURN:%(turn_idx)s] [AGENT:%(agent_name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        self.loggers[logger_name] = logger
    
    def _setup_model_logger(self):
        """Setup model.log logger"""
        logger_name = "model"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(
            self.log_dir / "model.log", 
            mode='a', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Set format
        formatter = logging.Formatter(
            '[%(asctime)s] [ROLLOUT:%(rollout_idx)s] [POLICY:%(policy_name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        self.loggers[logger_name] = logger
    
    def _setup_async_logger(self):
        """Setup async.log logger"""
        logger_name = "async"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(
            self.log_dir / "async.log", 
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
    
    def log_env_agent_info(self, rollout_idx: int, turn_idx: int, agent_name: str, 
                          message: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Log environment and agent related information
        
        Args:
            rollout_idx: Rollout index
            turn_idx: Turn index
            agent_name: Agent name
            message: Log message
            extra_data: Additional structured data
        """
        logger = self.loggers["env_agent"]
        
        # Build log content and safely serialize
        log_content = {
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "extra_data": safe_serialize(extra_data or {})
        }
        
        # Use extra parameter to pass context information
        extra = {
            "rollout_idx": rollout_idx,
            "turn_idx": turn_idx,
            "agent_name": agent_name
        }
        
        logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def log_model_interaction(self, rollout_idx: int, policy_name: str, 
                            prompt: str, response: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Log model interaction information
        
        Args:
            rollout_idx: Rollout index
            policy_name: Policy name
            prompt: Input prompt
            response: Model response
            extra_data: Additional data
        """
        logger = self.loggers["model"]
        
        log_content = {
            "prompt": prompt,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "extra_data": safe_serialize(extra_data or {})
        }
        if rollout_idx is not None:
            extra = {
                "rollout_idx": rollout_idx,
                "policy_name": policy_name
            }
        else:
            extra = {
                "policy_name": policy_name
            }
        
        logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def log_async_event(self, rollout_idx: int, event_type: str, 
                       message: str, extra_data: Optional[Dict[str, Any]] = None):
        """
        Log asynchronous execution events
        
        Args:
            rollout_idx: Rollout index
            event_type: Event type (start, complete, error, etc.)
            message: Event message
            extra_data: Additional data
        """
        logger = self.loggers["async"]
        
        log_content = {
            "event_type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "extra_data": safe_serialize(extra_data or {})
        }
        
        extra = {
            "rollout_idx": rollout_idx
        }
        
        logger.info(json.dumps(log_content, ensure_ascii=False, indent=2), extra=extra)
    
    def get_logger(self, logger_name: str) -> Optional[logging.Logger]:
        """Get specified logger"""
        return self.loggers.get(logger_name)

# Global logger configuration instance
_global_logger_config = None

def get_multi_logger() -> MultiLoggerConfig:
    """Get global multi-logger configuration instance"""
    global _global_logger_config
    if _global_logger_config is None:
        _global_logger_config = MultiLoggerConfig()
    return _global_logger_config

def init_multi_logger(log_dir: str = "logs") -> MultiLoggerConfig:
    """Initialize global multi-logger configuration"""
    global _global_logger_config
    _global_logger_config = MultiLoggerConfig(log_dir)
    return _global_logger_config
