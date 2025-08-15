#!/usr/bin/env python3
"""
测试多日志系统功能
"""

import os
import sys
import time
from pathlib import Path

# 添加项目路径到sys.path
sys.path.insert(0, str(Path(__file__).parent))

from pettingllms.utils.logger_config import init_multi_logger, get_multi_logger

def test_logger_config():
    """测试日志配置基本功能"""
    print("=== 测试日志配置基本功能 ===")
    
    # 初始化日志系统
    log_dir = "test_logs"
    multi_logger = init_multi_logger(log_dir)
    
    # 测试各种日志记录
    print("1. 测试 env_agent.log...")
    multi_logger.log_env_agent_info(
        rollout_idx=0,
        turn_idx=1,
        agent_name="test_agent",
        message="测试环境智能体日志记录",
        extra_data={
            "test_data": "这是测试数据",
            "value": 42
        }
    )
    
    print("2. 测试 model.log...")
    multi_logger.log_model_interaction(
        rollout_idx=0,
        policy_name="test_policy",
        prompt="这是一个测试提示",
        response="这是一个测试响应",
        extra_data={
            "event": "test_generation",
            "tokens": 100
        }
    )
    
    print("3. 测试 async.log...")
    multi_logger.log_async_event(
        rollout_idx=0,
        event_type="test_event",
        message="测试异步事件记录",
        extra_data={
            "start_time": time.time(),
            "task_id": "test_task_001"
        }
    )
    
    # 检查日志文件是否创建
    log_files = ["env_agent.log", "model.log", "async.log"]
    log_dir_path = Path(log_dir)
    
    print("\n=== 检查日志文件创建 ===")
    for log_file in log_files:
        log_path = log_dir_path / log_file
        if log_path.exists():
            print(f"✓ {log_file} 创建成功，大小: {log_path.stat().st_size} 字节")
            
            # 显示前几行内容
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"  内容示例 (共{len(lines)}行):")
                for i, line in enumerate(lines[:3]):  # 只显示前3行
                    print(f"    {i+1}: {line.strip()}")
                if len(lines) > 3:
                    print(f"    ... 还有 {len(lines)-3} 行")
        else:
            print(f"✗ {log_file} 创建失败")
    
    print(f"\n✓ 测试完成，日志文件存储在: {log_dir_path.absolute()}")

def test_agent_logging():
    """测试智能体日志记录功能"""
    print("\n=== 测试智能体日志记录功能 ===")
    
    try:
        from pettingllms.multi_agent_env.code.agents.code_agent import CodeGenerationAgent
        from pettingllms.multi_agent_env.code.agents.unit_test_agent import UnitTestGenerationAgent
        
        # 测试代码生成智能体
        print("1. 测试 CodeGenerationAgent...")
        code_agent = CodeGenerationAgent(rollout_idx=1)
        
        # 模拟环境数据
        class MockEnv:
            def __init__(self):
                self.state = MockState()
        
        class MockState:
            def __init__(self):
                self.problem = "编写一个函数计算两个数的和"
                self.current_code = None
                self.current_test_input = None
        
        mock_env = MockEnv()
        code_agent.update_from_env(mock_env)
        print("   ✓ CodeGenerationAgent.update_from_env() 完成")
        
        # 模拟模型响应
        test_response = """
        **Code:**
        ```python
        def add(a, b):
            return a + b
        ```
        
        **Explanation:**
        这是一个简单的加法函数。
        """
        code_agent.update_from_model(test_response)
        print("   ✓ CodeGenerationAgent.update_from_model() 完成")
        
        # 测试单元测试生成智能体
        print("2. 测试 UnitTestGenerationAgent...")
        test_agent = UnitTestGenerationAgent(rollout_idx=1)
        test_agent.update_from_env(mock_env)
        print("   ✓ UnitTestGenerationAgent.update_from_env() 完成")
        
        test_response_test = """
        **Test Input:**
        ```
        add(2, 3)
        ```
        
        **Test Output:**
        ```
        5
        ```
        
        **Explanation:**
        测试基本的加法功能。
        """
        test_agent.update_from_model(test_response_test)
        print("   ✓ UnitTestGenerationAgent.update_from_model() 完成")
        
        print("✓ 智能体日志记录测试完成")
        
    except ImportError as e:
        print(f"✗ 导入智能体类失败: {e}")
    except Exception as e:
        print(f"✗ 智能体日志记录测试失败: {e}")

def test_multiple_rollouts():
    """测试多个rollout的日志记录"""
    print("\n=== 测试多个rollout日志记录 ===")
    
    multi_logger = get_multi_logger()
    
    # 模拟多个rollout
    for rollout_idx in range(3):
        for turn_idx in range(2):
            # 记录环境智能体信息
            multi_logger.log_env_agent_info(
                rollout_idx=rollout_idx,
                turn_idx=turn_idx + 1,
                agent_name="code_generator" if turn_idx % 2 == 0 else "test_generator",
                message=f"处理 rollout {rollout_idx}, turn {turn_idx + 1}",
                extra_data={"processing": True}
            )
            
            # 记录模型交互
            multi_logger.log_model_interaction(
                rollout_idx=rollout_idx,
                policy_name="code_generator",
                prompt=f"rollout {rollout_idx} prompt",
                response=f"rollout {rollout_idx} response",
                extra_data={"turn": turn_idx + 1}
            )
            
            # 记录异步事件
            multi_logger.log_async_event(
                rollout_idx=rollout_idx,
                event_type="turn_complete",
                message=f"Turn {turn_idx + 1} completed for rollout {rollout_idx}",
                extra_data={"duration": 1.5}
            )
    
    print("✓ 多rollout日志记录测试完成")

def main():
    """主测试函数"""
    print("开始测试多日志系统...")
    
    # 清理之前的测试日志
    test_log_dir = Path("test_logs")
    if test_log_dir.exists():
        import shutil
        shutil.rmtree(test_log_dir)
    
    try:
        # 基本功能测试
        test_logger_config()
        
        # 智能体日志测试
        test_agent_logging()
        
        # 多rollout测试
        test_multiple_rollouts()
        
        print("\n🎉 所有测试完成！")
        print("请检查 test_logs/ 目录下的日志文件内容。")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
