#!/usr/bin/env python3
"""
测试summary logger功能的简单脚本
"""

import tempfile
import os
from pathlib import Path
from pettingllms.utils.logger_config import MultiLoggerConfig

def test_summary_logger():
    """测试summary logger的基本功能"""
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        # 初始化logger
        logger_config = MultiLoggerConfig(log_dir=temp_dir)
        
        # 测试数据
        test_rollout_idx = 1
        test_agent_rewards = {
            "code_generator": 0.85,
            "test_generator": 0.72
        }
        test_termination_reason = "max_turns_reached"
        test_extra_data = {
            "max_turns": 8,
            "env_state": "completed",
            "final_actions": {
                "code_generator": "generate_function",
                "test_generator": "generate_tests"
            }
        }
        
        # 记录summary信息
        logger_config.log_rollout_summary(
            rollout_idx=test_rollout_idx,
            agent_rewards=test_agent_rewards,
            termination_reason=test_termination_reason,
            extra_data=test_extra_data
        )
        
        # 检查日志文件是否创建
        log_files = list(Path(temp_dir).rglob("*.log"))
        print(f"Created log files: {[f.name for f in log_files]}")
        
        # 查找summary.log文件
        summary_log_file = None
        for log_file in log_files:
            if log_file.name == "summary.log":
                summary_log_file = log_file
                break
        
        if summary_log_file:
            print(f"Summary log file found: {summary_log_file}")
            
            # 读取并显示内容
            with open(summary_log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print("Summary log content:")
                print(content)
        else:
            print("Summary log file not found!")
        
        return summary_log_file is not None

if __name__ == "__main__":
    success = test_summary_logger()
    print(f"Test {'PASSED' if success else 'FAILED'}")
