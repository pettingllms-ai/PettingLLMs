# PyChecker Worker CPU Auto-Allocation

## 问题背景

在原始实现中，PyChecker workers 的 CPU 分配是硬编码的：
- 每个 worker 固定占用 0.4 CPU
- 代码创建 1800 个 worker 对象
- 在 200 CPU 的机器上，实际只能运行 500 个 workers (200 / 0.4 = 500)
- 导致大量 worker 对象无法启动，任务排队等待

## 解决方案

### 自动 CPU 分配算法

修改后的代码会根据以下公式自动计算每个 worker 的 CPU 占用：

```python
num_cpus_per_worker = (total_cpus * 0.9) / num_workers
num_cpus_per_worker = max(0.1, min(2.0, num_cpus_per_worker))  # 限制在 [0.1, 2.0]
```

**参数说明：**
- `total_cpus`: Ray 集群可用的总 CPU 数量
- `num_workers`: 配置文件中指定的 worker 数量
- `0.9`: 安全系数，避免过度订阅（over-subscription）
- `[0.1, 2.0]`: CPU 分配的合理范围

### 配置示例

**场景 1: 200 CPU 机器，512 workers**

```yaml
training:
  num_workers: 512
```

自动计算：
```
num_cpus_per_worker = (200 * 0.9) / 512 = 0.35 CPU/worker
最大并发任务数 = 200 / 0.35 = 571 > 512 ✅
```

**所有 512 个 workers 都能正常运行！**

**场景 2: 200 CPU 机器，1800 workers（原配置）**

```yaml
training:
  num_workers: 1800
```

自动计算：
```
num_cpus_per_worker = (200 * 0.9) / 1800 = 0.1 CPU/worker
最大并发任务数 = 200 / 0.1 = 2000 > 1800 ✅
```

**所有 1800 个 workers 都能运行，但每个只有 0.1 CPU，可能较慢**

**场景 3: 200 CPU 机器，100 workers**

```yaml
training:
  num_workers: 100
```

自动计算：
```
num_cpus_per_worker = (200 * 0.9) / 100 = 1.8 CPU/worker
最大并发任务数 = 200 / 1.8 = 111 > 100 ✅
```

**所有 100 个 workers 都能运行，每个有更多 CPU 资源，编译更快**

## 性能对比

### 修改前
| 配置 | Worker 对象 | 实际运行 | 并发任务 | CPU 利用率 |
|------|------------|---------|---------|-----------|
| 默认 | 1800 | 500 | 156 | ~31% |

**问题：**
- 只有 27% 的 workers 能启动 (500/1800)
- 并发任务数远低于预期 (156 vs 500)
- CPU 利用率低

### 修改后
| 配置 | Worker 对象 | 实际运行 | CPU/Worker | 理论并发 | CPU 利用率 |
|------|------------|---------|-----------|---------|-----------|
| 512 | 512 | 512 | 0.35 | 512 | ~90% |
| 1000 | 1000 | 1000 | 0.18 | 1000 | ~90% |

**改进：**
- ✅ 所有 worker 对象都能启动
- ✅ 理论并发任务数 = num_workers
- ✅ CPU 利用率提升到 90%
- ✅ 自动适配不同的硬件配置

## 环境变量覆盖

如果需要手动指定 CPU 分配（不推荐），仍可使用环境变量：

```bash
export PYCHECKER_WORKER_CPUS=0.2  # 手动指定每个 worker 0.2 CPU
export PYCHECKER_WORKER_CONCURRENCY=1  # 每个 worker 同时处理 1 个任务
```

**优先级：**
1. 环境变量 `PYCHECKER_WORKER_CPUS`（如果设置）
2. 自动计算（基于 `num_workers` 和 `total_cpus`）
3. 默认值 0.4

## 推荐配置

**对于 200+ CPU 的机器：**

```yaml
training:
  num_workers: 512  # 推荐值：可以根据实际任务数调整
  step_timeout: 180.0  # 每个任务的超时时间
  generate_timeout: 300.0  # LLM 生成超时
```

**调优建议：**

1. **Worker 数量选择：**
   - `num_workers ≈ train_batch_size × train_sample_num × 2`
   - 例如：32 × 8 × 2 = 512

2. **并发与性能平衡：**
   - 更多 workers (如 1000)：更高并发，但每个任务 CPU 少，编译慢
   - 更少 workers (如 256)：更少并发，但每个任务 CPU 多，编译快
   - 推荐：512 是一个平衡点

3. **监控与调整：**
   - 观察日志中的 CPU 利用率
   - 如果编译超时频繁，减少 num_workers
   - 如果 CPU 利用率低，增加 num_workers

## 代码修改摘要

**修改文件：**
1. `pettingllms/multi_agent_env/pychecker_rl/pychecker_worker.py`
   - `get_ray_pychecker_worker_cls()` 添加 `num_workers` 参数
   - 实现自动 CPU 计算逻辑

2. `pettingllms/trainer/multi_agents_execution_engine.py`
   - 传递 `num_workers` 给 worker 工厂函数

3. `pettingllms/config/pychecker_rl/*.yaml`
   - 添加 `num_workers: 512` 配置
   - 添加 `step_timeout` 和 `generate_timeout` 配置

## 测试验证

启动训练时，查看日志中的 worker 创建信息：

```
Auto-calculated num_cpus_per_worker: 0.35 (total_cpus=200, num_workers=512)
Creating PyChecker worker class with num_cpus=0.35, max_concurrency=1
Creating 512 Ray docker workers
```

确认：
- ✅ `num_cpus_per_worker` 在合理范围 [0.1, 2.0]
- ✅ 总 CPU 需求 `num_cpus × num_workers ≤ total_cpus`
- ✅ 日志中没有 "Ray worker creation failed" 错误

## 相关问题排查

**问题：仍然只有少量并发任务**

可能原因：
1. Ray 资源碎片化 → 重启 Ray 集群
2. 其他进程占用 CPU → 检查系统负载
3. Worker 启动慢 → 增加 `ray_wait_register_center_timeout`

**问题：编译超时频繁**

可能原因：
1. CPU per worker 太少 → 减少 `num_workers`
2. Verilator 编译耗时长 → 检查电路复杂度
3. 超时设置太短 → 增加 `step_timeout`
