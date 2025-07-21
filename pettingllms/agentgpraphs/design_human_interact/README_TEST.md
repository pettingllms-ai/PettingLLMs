# Frontend Design Agent Graph 测试指南

本目录包含用于测试双智能体协作前端设计系统的脚本和工具。

## 🚀 快速开始

### 方法1: 使用完整测试脚本 (推荐)

```bash
cd pettingllms/agentgpraphs/design_human_interact
./test_multi_agent_graph.sh
```

### 方法2: 使用快速测试脚本

```bash
cd pettingllms/agentgpraphs/design_human_interact
./quick_test.sh
```

## 📋 前置条件

1. **安装依赖**
   ```bash
   pip install sglang[all]
   pip install datasets
   pip install selenium
   pip install pillow
   ```

2. **确保端口可用**
   - 默认使用端口 8000 和 8001
   - 如果端口被占用，脚本会自动清理

## 🛠️ 脚本说明

### test_multi_agent_graph.sh (完整版)

**功能特性:**
- ✅ 完整的依赖检查
- ✅ 自动端口冲突处理
- ✅ 详细的日志记录
- ✅ 服务器健康检查
- ✅ 自动清理资源
- ✅ 支持自定义参数

**使用参数:**
```bash
./test_multi_agent_graph.sh [OPTIONS]

Options:
  --model MODEL_NAME        模型名称 (默认: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
  --code-port PORT          代码生成服务器端口 (默认: 8000)
  --visual-port PORT        视觉分析服务器端口 (默认: 8001)
  --samples NUM             测试样本数 (默认: 5)
  --iterations NUM          最大迭代次数 (默认: 3)
  --output-dir DIR          输出目录 (默认: ./test_results)
  --help, -h                显示帮助信息
```

**示例用法:**
```bash
# 使用默认设置
./test_multi_agent_graph.sh

# 自定义参数
./test_multi_agent_graph.sh --samples 10 --iterations 5 --code-port 8002

# 使用不同模型
./test_multi_agent_graph.sh --model "Qwen/Qwen2.5-1.5B-Instruct"
```

### quick_test.sh (快速版)

**功能特性:**
- ⚡ 快速启动和测试
- 🔧 简化配置
- 📊 基础测试 (2个样本, 2次迭代)

## 🔧 测试流程

1. **启动阶段**
   - 检查Python和SGLang依赖
   - 清理现有的sglang进程
   - 启动两个SGLang服务器 (不同端口)

2. **验证阶段**
   - 等待服务器完全启动
   - 测试API连接性
   - 验证模型响应

3. **测试阶段**
   - 从WebSight数据集加载测试样本
   - 运行多智能体协作测试
   - 生成评估报告

4. **清理阶段**
   - 自动停止所有启动的服务
   - 清理临时文件

## 📊 输出结果

### 完整测试输出
```
test_results/
├── graph_test_results_YYYYMMDD_HHMMSS.json  # 主要测试结果
├── sglang_code_generation_8000.log          # 代码生成服务器日志
├── sglang_visual_analysis_8001.log          # 视觉分析服务器日志
└── temp/                                    # 临时文件目录
```

### 快速测试输出
```
quick_test_results.json  # 测试结果
code_server.log          # 代码服务器日志
visual_server.log        # 视觉服务器日志
```

### 结果JSON格式
```json
{
  "total_tasks": 5,
  "successful_tasks": 3,
  "success_rate": 0.6,
  "average_iterations": 2.4,
  "detailed_results": [
    {
      "task_id": "task_001",
      "success": true,
      "total_iterations": 2,
      "final_html": "...",
      "agent_data": {
        "agent1": {
          "original_name": "visual_agent",
          "total_reward": 1.5,
          "steps": [...]
        },
        "agent2": {
          "original_name": "code_agent", 
          "total_reward": 2.0,
          "steps": [...]
        }
      }
    }
  ]
}
```

## 🔍 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 手动清理端口
   lsof -ti:8000,8001 | xargs kill -9
   ```

2. **模型下载失败**
   ```bash
   # 预先下载模型
   huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
   ```

3. **内存不足**
   ```bash
   # 使用更小的模型
   ./test_multi_agent_graph.sh --model "Qwen/Qwen2.5-0.5B-Instruct"
   ```

4. **SGLang未安装**
   ```bash
   pip install sglang[all]
   # 或者从源码安装
   pip install git+https://github.com/sgl-project/sglang.git
   ```

### 日志查看

```bash
# 查看服务器启动日志
tail -f test_results/sglang_*.log

# 查看测试详细输出
cat test_results/graph_test_results_*.json | jq
```

## 🏃‍♂️ 性能建议

1. **硬件要求**
   - 内存: 至少 8GB RAM
   - GPU: 推荐 4GB+ VRAM (可选)
   - 存储: 至少 10GB 可用空间

2. **优化设置**
   ```bash
   # 减少测试样本数量
   ./test_multi_agent_graph.sh --samples 3 --iterations 2
   
   # 使用CPU模式
   export CUDA_VISIBLE_DEVICES=""
   ```

## 📝 开发模式

如果您要修改和调试代码：

```bash
# 启动服务器但不运行测试
./test_multi_agent_graph.sh &
# 等待启动完成后手动运行
python agent_collaboration_graph.py --hostname localhost --code_port 8000 --visual_port 8001 --num_samples 1

# 或者分别启动服务器
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8000 &
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port 8001 &
```

## 🤝 支持

如果遇到问题，请检查：
1. 所有依赖是否正确安装
2. 网络连接是否正常 (用于下载模型)
3. 系统资源是否充足
4. 端口是否被其他程序占用 