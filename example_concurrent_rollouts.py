#!/usr/bin/env python3
"""
并发rollout执行示例

这个示例演示了如何使用改进后的MultiAgentsExecutionEngine
来并发执行多个rollout任务
"""

import asyncio
from omegaconf import OmegaConf
from pettingllms.trainer.multi_agents_execution_engine import MultiAgentsExecutionEngine

async def example_concurrent_rollouts():
    """
    示例：如何使用并发rollout执行功能
    """
    print("=== 并发Rollout执行示例 ===")
    
    # 加载配置（这里需要根据实际情况调整配置路径）
    try:
        trainer_config = OmegaConf.load("pettingllms/config/code/ppo_trainer/base.yaml")
        config = OmegaConf.load("pettingllms/config/code/code_test.yaml")
    except Exception as e:
        print(f"配置加载失败: {e}")
        return
    
    # 初始化执行引擎（这里需要根据实际情况提供tokenizer_dict和server_manager_dict）
    # 注意：在实际使用中，您需要正确初始化这些依赖项
    tokenizer_dict = {}  # 实际使用时需要提供
    server_manager_dict = {}  # 实际使用时需要提供
    
    try:
        execution_engine = MultiAgentsExecutionEngine(
            config=config,
            tokenizer_dict=tokenizer_dict,
            server_manager_dict=server_manager_dict
        )
        
        # 定义要并发执行的rollout索引
        rollout_indices = [0, 1, 2, 3, 4]  # 执行5个rollout
        max_concurrent_tasks = 3  # 最大并发任务数
        
        print(f"准备并发执行 {len(rollout_indices)} 个rollout")
        print(f"最大并发任务数: {max_concurrent_tasks}")
        
        # 使用新的并发方法执行rollouts
        results = await execution_engine.generate_multiple_rollouts_concurrent(
            rollout_indices=rollout_indices,
            max_concurrent_tasks=max_concurrent_tasks
        )
        
        # 处理结果
        print(f"\n=== 并发执行结果 ===")
        print(f"成功完成的rollout数量: {len(results)}")
        
        for rollout_idx, trajectory_data in results.items():
            print(f"Rollout {rollout_idx}: {type(trajectory_data)}")
            # 这里可以添加更多的结果处理逻辑
            
    except Exception as e:
        print(f"并发执行失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    主函数 - 运行并发rollout示例
    """
    print("启动并发rollout执行示例...")
    asyncio.run(example_concurrent_rollouts())

if __name__ == "__main__":
    main()
