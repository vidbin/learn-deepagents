# AgentEvent 类型定义

## 基础结构

```python
@dataclass
class AgentEvent:
    type: str       # 事件类型
    turn_id: str    # 当前对话轮次 ID
    data: dict      # 事件数据（结构因 type 而异）
```

SSE 传输格式：`data: {"type": "...", "turn_id": "...", "data": {...}}\n\n`

---

## 事件类型一览

| 类型 | 来源 | 说明 |
|------|------|------|
| `status` | chat_stream | 流程状态变更 |
| `thinking_delta` | event_converter | 思考内容增量 |
| `thinking_complete` | event_converter | 思考阶段结束 |
| `text_delta` | event_converter | 文本输出增量 |
| `text_complete` | event_converter | 文本输出结束 |
| `tool_start` | event_converter | 工具调用开始 |
| `tool_result` | event_converter | 工具调用结束 |
| `tool_progress` | event_converter | 工具参数流式进度 |
| `todos_update` | event_converter | TODO 列表更新 |
| `node_enter` | event_converter | LangGraph 节点进入 |
| `node_exit` | event_converter | LangGraph 节点退出 |
| `complete` | chat_stream | 整轮对话完成 |
| `stopped` | chat_stream | 用户主动停止 |
| `error` | chat_stream | 错误 |

---

## 各事件 data 结构

### `status`

流程阶段通知，前端用于显示加载状态文案。

```typescript
{
  status: "preparing" | "loading_skills" | "loading_mcp" | "thinking",
  conversation_id?: string   // 首次创建对话时携带
}
```

---

### `thinking_delta`

模型思考内容的流式增量片段。

```typescript
{
  content: string,          // 增量思考文本
  subagent_id?: string      // 子智能体 ID（仅子智能体事件携带）
}
```

---

### `thinking_complete`

模型思考阶段结束，携带完整思考内容。

```typescript
{
  content: string,          // 完整思考文本（累积缓冲）
  subagent_id?: string
}
```

---

### `text_delta`

模型文本输出的流式增量片段。

```typescript
{
  content: string,          // 增量文本
  subagent_id?: string
}
```

---

### `text_complete`

模型文本输出结束，携带完整文本。

```typescript
{
  content: string,          // 完整文本（累积缓冲）
  subagent_id?: string
}
```

---

### `tool_start`

工具调用开始，参数已完整解析。

```typescript
{
  tool_use_id: string,      // 工具调用唯一 ID（run_id）
  tool_name: string,        // 工具名称（如 "bash", "read_file", "mcp_xxx"）
  tool_type: "builtin" | "mcp",  // 工具分类
  input: Record<string, any>,    // 工具输入参数
  subagent_id?: string
}
```

---

### `tool_result`

工具调用结束，携带执行结果。

```typescript
{
  tool_use_id: string,      // 对应 tool_start 的 ID
  tool_name: string,
  result: string,           // 工具执行结果文本
  is_error: boolean,        // 是否执行出错
  subagent_id?: string
}
```

---

### `tool_progress`

工具参数流式传输进度（大参数时每 0.5s 发送一次）。

```typescript
{
  tool_use_id: string,
  tool_name: string,
  bytes_received: number,   // 已接收参数字节数
  subagent_id?: string
}
```

---

### `todos_update`

TODO 列表状态更新（来自 `write_todos` 工具或节点输出）。

```typescript
{
  todos: Array<{
    id: string,
    content: string,
    status: "pending" | "in_progress" | "done"
  }>
}
```

---

### `node_enter`

LangGraph 图节点进入。

```typescript
{
  node: string,             // 节点名称（agent, tools, task, planner, executor, reviewer 等）
  step: number,            // langgraph_step
  path: string,            // langgraph_path 字符串表示
  is_subagent: boolean,    // 是否来自子智能体
  subagent_id?: string
}
```

---

### `node_exit`

LangGraph 图节点退出。

```typescript
{
  node: string,
  step: number,
  path: string,
  is_subagent: boolean,
  subagent_id?: string
}
```

---

### `complete`

整轮对话正常完成。

```typescript
{
  conversation_id: string,
  duration_ms: number,      // 本轮耗时（毫秒）
  model: string,            // 使用的模型名称
  usage: {
    input_tokens: number,
    output_tokens: number,
    total_tokens: number
  }
}
```

---

### `stopped`

用户主动中断对话。

```typescript
{
  conversation_id: string,
  duration_ms: number
}
```

---

### `error`

错误事件。

```typescript
{
  code?: string,            // 错误码（如 "NOT_FOUND", "NO_MODEL"）
  message: string           // 错误描述
}
```

---

## 子智能体事件

当事件来自子智能体时，`data` 中会附带 `subagent_id` 字段。以下事件类型支持子智能体：

- `thinking_delta` / `thinking_complete`
- `text_delta` / `text_complete`
- `tool_start` / `tool_result` / `tool_progress`
- `node_enter` / `node_exit`

子智能体的 `subagent_id` 等于触发它的 `task` 工具的 `tool_use_id`（即 `run_id`），在整个子智能体生命周期内保持稳定。

### 子智能体生命周期

```
tool_start (tool_name="task", tool_use_id="xxx")   ← 子智能体启动
  ├── thinking_delta (subagent_id="xxx")           ← 子智能体思考
  ├── thinking_complete (subagent_id="xxx")
  ├── text_delta (subagent_id="xxx")               ← 子智能体输出
  ├── tool_start (subagent_id="xxx")               ← 子智能体调用工具
  ├── tool_result (subagent_id="xxx")
  ├── text_delta (subagent_id="xxx")               ← 子智能体继续输出
  └── ...
tool_result (tool_name="task", tool_use_id="xxx")  ← 子智能体完成
```

---

## 事件时序（典型流程）

```
status (preparing)
status (loading_skills)
status (loading_mcp)
status (thinking)
node_enter (agent)
  thinking_delta × N
  thinking_complete
  text_delta × N
  text_complete
node_exit (agent)
node_enter (tools)
  tool_start
  tool_result
node_exit (tools)
node_enter (agent)
  text_delta × N
  text_complete
node_exit (agent)
complete
```
