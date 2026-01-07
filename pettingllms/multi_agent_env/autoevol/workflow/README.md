# Workflow System

一套现代化、鲁棒、易于扩展的多智能体工作流系统。

## 核心特性

### 🎯 **无字符串解析**
- 使用结构化的 `Message` 对象进行通信
- 类型安全的消息传递
- 避免了脆弱的字符串解析逻辑

### 🧩 **模块化组件**
- `AgentNode`: 支持工具调用的智能体节点
- `EnsembleNode`: 多智能体集成（投票/共识）
- `DebateNode`: 多智能体辩论
- `ReflectionNode`: 自我反思与优化
- `RouterNode`: 条件分支路由

### 🔗 **灵活编排**
- `Workflow`: 顺序执行工作流
- `ConditionalWorkflow`: 条件执行工作流
- `LoopWorkflow`: 循环执行工作流
- 支持链式调用

### 🛡️ **鲁棒性**
- 统一的错误处理
- 完整的日志记录
- 上下文管理

## 快速开始

### 基础示例

```python
from workflow.core import ToolRegistry
from workflow.nodes import AgentNode
from workflow.workflow import Workflow

# 1. 设置工具
tool_registry = ToolRegistry()
tool_registry.register(
    name="search",
    func=my_search_function,
    description="Search the web",
    parameters={...}
)

# 2. 创建智能体
agent = AgentNode(
    name="SearchAgent",
    system_prompt="You are a helpful search assistant.",
    tool_registry=tool_registry
)

# 3. 创建工作流
workflow = Workflow(name="simple_search")
workflow.add_node(agent)

# 4. 运行
result = workflow.run("What is Python?")
print(result.content)
```

### Ensemble (集成)

```python
from workflow.nodes import EnsembleNode

# 创建多个智能体
agent1 = AgentNode(name="Agent1", ...)
agent2 = AgentNode(name="Agent2", ...)
agent3 = AgentNode(name="Agent3", ...)

# 使用投票策略
ensemble = EnsembleNode(
    name="VotingEnsemble",
    agents=[agent1, agent2, agent3],
    strategy="majority_vote"
)

# 或使用共识策略
consensus_agent = AgentNode(name="Synthesizer", ...)
ensemble = EnsembleNode(
    name="ConsensusEnsemble",
    agents=[agent1, agent2, agent3],
    strategy="consensus",
    consensus_agent=consensus_agent
)

workflow = Workflow().add_node(ensemble)
result = workflow.run("Your question here")
```

### Debate (辩论)

```python
from workflow.nodes import DebateNode

# 创建辩论者
debater1 = AgentNode(name="ProDebater", ...)
debater2 = AgentNode(name="ConDebater", ...)
judge = AgentNode(name="Judge", ...)

# 设置辩论
debate = DebateNode(
    name="Debate",
    debaters=[debater1, debater2],
    judge=judge,
    num_rounds=2  # 辩论轮数
)

workflow = Workflow().add_node(debate)
result = workflow.run("Should we use AI in education?")
```

### Reflection (反思)

```python
from workflow.nodes import ReflectionNode

# 创建智能体
agent = AgentNode(name="ThinkingAgent", ...)

# 添加反思能力
reflection = ReflectionNode(
    name="SelfReflection",
    agent=agent,
    num_iterations=2  # 反思迭代次数
)

workflow = Workflow().add_node(reflection)
result = workflow.run("Explain quantum computing")
```

### 复杂工作流

```python
from workflow.workflow import Workflow

# 创建多阶段工作流
researcher = AgentNode(name="Researcher", ...)
fact_checker = AgentNode(name="FactChecker", ...)
writer = AgentNode(name="Writer", ...)

workflow = Workflow(name="research_pipeline")
workflow.add_nodes([researcher, fact_checker, writer])

result = workflow.run("Research the history of AI")
```

### 条件工作流

```python
from workflow.workflow import ConditionalWorkflow

# 创建条件工作流
workflow = ConditionalWorkflow(name="conditional")

# 添加有条件的节点
workflow.add_node(
    node=agent1,
    condition=lambda ctx: "urgent" in ctx.get_latest_message().content
)
workflow.add_node(node=agent2)  # 无条件执行

result = workflow.run("Your input")
```

### Router (路由)

```python
from workflow.nodes import RouterNode, create_keyword_router

# 基于关键词的路由
router = create_keyword_router(
    name="TaskRouter",
    keyword_routes={
        "search": search_agent,
        "calculate": calc_agent,
        "summarize": summary_agent
    },
    default_node=general_agent
)

workflow = Workflow().add_node(router)
result = workflow.run("Please search for...")
```

## 核心概念

### Message (消息)

```python
from workflow.core import Message, MessageType

msg = Message(
    content="Hello",
    message_type=MessageType.USER_INPUT,
    metadata={"key": "value"},
    sender="NodeA",
    recipient="NodeB"
)
```

### Context (上下文)

```python
from workflow.core import Context

context = Context()
context.add_message(message)
context.set_state("key", "value")
latest = context.get_latest_message()
```

### ToolRegistry (工具注册)

```python
from workflow.core import ToolRegistry

registry = ToolRegistry()
registry.register(
    name="tool_name",
    func=my_function,
    description="What this tool does",
    parameters={
        "type": "object",
        "properties": {...},
        "required": [...]
    }
)

# 调用工具
result = registry.call_tool("tool_name", {"param": "value"})
```

## 与旧系统对比

### 旧系统问题
```python
# ❌ 脆弱的字符串解析
if "<submit>" in response:
    submit_result = response.split("<submit>")[1].split("</submit>")[0]
    if "FinalResult:" in submit_result:
        ...

# ❌ 硬编码逻辑
if tool_name == "google-search":
    tool_response = self.environment.search(...)
elif tool_name == "fetch_data":
    tool_response = self.environment.fetch(...)
```

### 新系统优势
```python
# ✅ 结构化消息
if result.message_type == MessageType.FINAL_RESULT:
    return result.content

# ✅ 工具注册表
tool_registry.call_tool(tool_name, parameters)

# ✅ 易于扩展
workflow.add_nodes([agent1, agent2, agent3])
```

## 扩展指南

### 创建自定义 Node

```python
from workflow.core import WorkflowNode, Context, Message, MessageType

class MyCustomNode(WorkflowNode):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        # 初始化你的逻辑
    
    def process(self, context: Context) -> Message:
        # 获取输入
        input_msg = context.get_latest_message()
        
        # 处理逻辑
        result = self.my_processing(input_msg.content)
        
        # 返回结果
        return Message(
            content=result,
            message_type=MessageType.AGENT_RESPONSE,
            metadata={"custom": "data"}
        )
```

### 创建自定义工作流

```python
class MyWorkflow(Workflow):
    def __init__(self):
        super().__init__(name="my_workflow")
        # 添加自定义初始化
    
    def run(self, input_message: str, **kwargs):
        # 添加自定义前处理
        result = super().run(input_message)
        # 添加自定义后处理
        return result
```

## 最佳实践

1. **使用结构化消息**：始终通过 `Message` 对象传递信息
2. **工具化**：将可重用的功能注册为工具
3. **模块化**：将复杂逻辑拆分为多个节点
4. **错误处理**：检查 `MessageType.ERROR`
5. **日志记录**：使用 `self.logger` 记录关键步骤
6. **元数据**：使用 `metadata` 传递额外信息

## 示例项目

查看 `examples/search_workflow_example.py` 获取完整示例：
- 基础搜索
- 集成搜索
- 辩论搜索
- 反思搜索
- 复杂多阶段工作流

## 架构

```
workflow/
├── core.py              # 核心抽象类
├── workflow.py          # 工作流编排
├── nodes/
│   ├── agent_node.py    # 智能体节点
│   ├── ensemble_node.py # 集成节点
│   ├── debate_node.py   # 辩论节点
│   ├── reflection_node.py # 反思节点
│   └── router_node.py   # 路由节点
└── README.md

examples/
└── search_workflow_example.py
```

## 迁移指南

从旧的 `BaseWorkFlow` 迁移：

1. **替换字符串解析**：使用 `Message` 和 `MessageType`
2. **工具注册**：使用 `ToolRegistry` 替代硬编码
3. **节点化**：将智能体包装为 `AgentNode`
4. **组合**：使用 `Workflow` 编排节点

## License

MIT

