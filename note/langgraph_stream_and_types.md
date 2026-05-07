# Q: LangGraph 流式输出有哪些类型？如何安全获取属性？

## 流式输出类型 (stream_mode)

| 类型 | stream_mode | data 内容 |
|------|-------------|-----------|
| `values` | `"values"` | 完整状态（每步后的所有状态值） |
| `updates` | `"updates"` | 节点名称 -> 输出内容的字典 |
| `messages` | `"messages"` | 元组 `(message, metadata)`，message 是 BaseMessage（如 AIMessageChunk） |
| `custom` | `"custom"` | 用户自定义数据（通过 StreamWriter 写入） |
| `checkpoints` | `"checkpoints"` | 检查点事件 |
| `tasks` | `"tasks"` | 任务开始/结束事件 |
| `debug` | `"debug"` | 调试信息 |

## chunk 结构

```python
chunk = {
    "type": "updates",  # 或 "messages", "values" 等
    "ns": (),           # tuple[str, ...]，命名空间（哪个子图）
    "data": {}          # 类型依赖的内容
}
```

## messages 类型的具体结构

```python
if chunk["type"] == "messages":
    msg, meta = chunk["data"]
    # msg: AIMessageChunk 或其他 BaseMessage
    # meta: {"langgraph_step": int, "langgraph_node": str, "langgraph_triggers": list}
```

## Python 安全获取属性的方法

### 1. getattr() - 最常用
```python
# 获取属性，不存在则返回默认值
name = getattr(obj, "name", None)
name = getattr(obj, "name", "默认值")
```

### 2. 字典式的 get()
```python
value = obj.get("key", "默认值")
```

### 3. 安全的部分解包
```python
d = {"name": None, "id": "id001", "age": "001", "gender": "男"}

# 安全提取，不存在的键不会报错
picked = {k: d.get(k) for k in ("name", "id", "not_exist")}
# {"name": None, "id": "id001", "not_exist": None}
```

### 4. 流式输出中的安全访问
```python
chunk = {"type": "messages", "ns": (), "data": (msg, meta)}

# 安全获取嵌套属性
msg = chunk.get("data", (None, {}))[0]
if msg is not None:
    tool_calls = getattr(msg, "tool_calls", [])
    invalid_calls = getattr(msg, "invalid_tool_calls", [])
```

## 判断变量类型
```python
# isinstance() 推荐方式
if isinstance(x, str):
    print("是字符串")

# 流式输出中判断 chunk 类型
for chunk in agent.stream(input_content, stream_mode=["updates", "messages"]):
    t = chunk["type"]
    if t == "messages":
        msg, meta = chunk["data"]
        print(f"消息类型: {type(msg).__name__}")  # AIMessageChunk
    elif t == "updates":
        print(f"节点更新: {chunk['data']}")
```

## ToolCallChunk 与 ToolCall 的区别

| 字段 | ToolCall | ToolCallChunk |
|------|-----------|-----------------|
| `type` | `"tool_call"` | `"tool_call_chunk"` |
| `args` | `dict[str, Any]` | `str \| None` |
| `name` | `str`（必须） | `str \| None` |
| 含义 | 完整工具调用 | 工具调用的片段 |

流式过程中 `args` 是逐步拼接的 JSON 字符串。

## InvalidToolCall

当模型生成的工具调用参数无法解析时（如 invalid JSON），会被归类为 `invalid_tool_calls`：

```python
class InvalidToolCall(TypedDict):
    type: Literal["invalid_tool_call"]
    id: str | None
    name: str | None
    args: str | None
    error: str | None  # 解析错误的原因
```

在 AIMessage 中的位置：
```python
class AIMessage(BaseMessage):
    tool_calls: list[ToolCall] = Field(default_factory=list)
    invalid_tool_calls: list[InvalidToolCall] = Field(default_factory=list)
```