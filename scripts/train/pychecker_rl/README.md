# PyChecker RL Training Scripts

## 📋 脚本说明

本目录包含两个并行训练脚本，可以同时运行在不同的 GPU 组上：

### 1. pychecker_rl_L2_multi_agent.sh
- **GPU 组**: 0,1
- **CPU 分配**: 112 CPUs
- **Worker 数量**: 384 workers
- **实验名称**: `pychecker_rl_after_stl_8B_gpu01`

### 2. pychecker_rl_L2_multi_agent_1.sh
- **GPU 组**: 3,4
- **CPU 分配**: 112 CPUs
- **Worker 数量**: 384 workers
- **实验名称**: `pychecker_rl_after_stl_8B_gpu34`

## 🚀 使用方法

### 方式 1: 单任务运行

**运行 GPU 0,1 任务:**
```bash
cd scripts/train/pychecker_rl
bash pychecker_rl_L2_multi_agent.sh
```

**运行 GPU 3,4 任务:**
```bash
cd scripts/train/pychecker_rl
bash pychecker_rl_L2_multi_agent_1.sh
```

### 方式 2: 两任务并行运行

**Terminal 1** (GPU 0,1):
```bash
cd scripts/train/pychecker_rl
bash pychecker_rl_L2_multi_agent.sh
```

**Terminal 2** (GPU 3,4):
```bash
cd scripts/train/pychecker_rl
bash pychecker_rl_L2_multi_agent_1.sh
```

## 📊 资源分配详情

### 单任务运行 (112 CPUs, 384 workers)

```
Total CPUs: 112
Workers: 384
CPU per worker: 0.2625
CPU utilization: 90.0%
Theoretical concurrent tasks: 384

✅ 所有 384 个 worker 都能被创建
✅ CPU 利用率达到 90%
✅ 足够支持 batch_size × sample_num = 64 × 6 = 384 并发任务
```

### 两任务并行运行 (224 CPUs, 768 workers)

```
Total CPUs: 224 (112 per task)
Total workers: 768 (384 per task)
CPU per worker: 0.2625
Total CPU utilization: 90.0%

✅ 所有 768 个 worker 都能被创建
✅ GPU 组完全隔离（gpu_0_1 vs gpu_3_4）
✅ 临时文件完全隔离
```

## 🎯 配置参数

两个脚本使用相同的训练参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| train_batch_size | 64 | 训练批次大小 |
| train_sample_num | 6 | 每个样本采样数 |
| max_prompt_length | 8192 | 最大提示长度 |
| max_response_length | 8192 | 最大响应长度 |
| total_training_steps | 200 | 总训练步数 |
| num_workers | 384 | Worker 数量 |

## 📁 临时文件路径

### GPU 0,1 任务 (pychecker_rl_L2_multi_agent.sh)
```
tmp/pychecker_tasks/gpu_0_1/worker_0/...
tmp/pychecker_tasks/gpu_0_1/worker_1/...
...
```

### GPU 3,4 任务 (pychecker_rl_L2_multi_agent_1.sh)
```
tmp/pychecker_tasks/gpu_3_4/worker_0/...
tmp/pychecker_tasks/gpu_3_4/worker_1/...
...
```

## 🔧 性能优化建议

### 场景 1: 追求最快速度（单任务）
- 使用单个脚本
- 384 workers with 0.2625 CPU/worker
- 推荐用于快速完成单个实验

### 场景 2: 同时运行多个实验（并行任务）
- 同时运行两个脚本
- 各 384 workers with 0.2625 CPU/worker
- 推荐用于对比不同配置

### 场景 3: 减少 worker 数量以提高单个任务速度
```yaml
# 修改配置文件
training:
  num_workers: 256  # 减少到 256
```
- CPU per worker 会增加到 0.39
- 编译速度更快
- 并发能力降低

## 🐛 故障排查

### 问题: Ray actor creation blocked

**症状:**
```
Ray Actor creation blocked: insufficient CPU resources
```

**解决方案:**
1. 检查 Ray 集群状态: `ray status`
2. 确认 `RAY_NUM_CPUS=112` 已设置
3. 或减少 worker 数量: `training.num_workers=256`

### 问题: GPU 内存不足

**症状:**
```
CUDA out of memory
```

**解决方案:**
```bash
# 修改脚本中的 GPU 内存利用率
$model_0_config_path.actor_rollout_ref.rollout.gpu_memory_utilization=0.6
# 从 0.7 改为 0.6
```

### 问题: 路径冲突

**症状:**
```
File exists error in tmp/pychecker_tasks/
```

**解决方案:**
```bash
# 清理临时文件
rm -rf tmp/pychecker_tasks/

# 重新运行脚本
bash pychecker_rl_L2_multi_agent.sh
```

## 📊 监控和日志

### 查看实验进度

**GPU 0,1 任务:**
```bash
# 查看日志
tail -f checkpoints/pychecker_rl_after_stl_8B_gpu01/logs/train.log

# 查看临时文件
ls tmp/pychecker_tasks/gpu_0_1/
```

**GPU 3,4 任务:**
```bash
# 查看日志
tail -f checkpoints/pychecker_rl_after_stl_8B_gpu34/logs/train.log

# 查看临时文件
ls tmp/pychecker_tasks/gpu_3_4/
```

### 查看 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用 gpustat
watch -n 1 gpustat
```

### 查看 CPU 使用情况

```bash
# 使用 htop
htop

# 或使用 top
top
```

## 📚 相关文档

- [GPU_GROUP_ISOLATION_SUMMARY.md](../../../GPU_GROUP_ISOLATION_SUMMARY.md) - GPU 组隔离详细说明
- [WORKER_PATH_ISOLATION_SUMMARY.md](../../../WORKER_PATH_ISOLATION_SUMMARY.md) - Worker 路径隔离
- [WORKER_OPTIMIZATION_SUMMARY.md](../../../WORKER_OPTIMIZATION_SUMMARY.md) - Worker CPU 优化

## ✅ 快速检查清单

运行前确认：
- [ ] GPU 可用: `nvidia-smi`
- [ ] Ray 已安装: `ray --version`
- [ ] 模型路径正确: `/home/lah003/models/PRO-V-R1`
- [ ] 磁盘空间充足: `df -h`
- [ ] CPU 资源充足: 至少 112 CPUs per task

运行后验证：
- [ ] 日志文件正常生成
- [ ] 临时文件路径正确（包含 gpu_0_1 或 gpu_3_4）
- [ ] GPU 内存使用正常（~70%）
- [ ] CPU 利用率正常（~90%）

---

**最后更新**: 2025-11-11
**版本**: 1.0
