# Q: LangChain 中 init_chat_model 怎么用？如何自定义 model？agent.invoke 返回值是什么？

## init_chat_model 用法

```python
from langchain.chat_models import init_chat_model
# 使用 OpenAI
model = init_chat_model(model="openai:gpt-4")

# 使用 OpenAI
model = init_chat_model(model="gpt-4", model_provider="openai")

# 使用 Anthropic
model = init_chat_model(model="claude-3-sonnet-20240229", model_provider="anthropic")

# 自定义端点
model = init_chat_model(
    model="custom-model",
    model_provider="openai",
    base_url="https://your-custom-api.com/v1",
    api_key="your-api-key"
)
```

**支持的 model_provider**: openai, anthropic, azure_openai, google_vertexai, ollama, localai

## 函数作为 Tool

函数必须提供 docstring 或使用 @tool 装饰器：

```python
from langchain_core.tools import tool

@tool(description="获取指定城市的天气信息")
def get_weather(city: str) -> str:
    """城市名称，如：北京、上海、武汉"""
    return f"{city} 晴天~"
```

## agent.invoke 返回值

```python
{
    "messages": [
        HumanMessage(content="..."),
        AIMessage(content="..."),
        ToolMessage(content="...", tool_call_id="..."),
        AIMessage(content="最终回复")
    ]
}
```

## Message 类型

| 类型 | 关键属性 |
|------|----------|
| HumanMessage | `.content` |
| AIMessage | `.content`, `.tool_calls` |
| ToolMessage | `.content`, `.tool_call_id` |
| SystemMessage | `.content` |
| ChatMessage | `.content`, `.role` |
| RemoveMessage | 用于删除消息 |

## 解析示例

```python
for msg in messages["messages"]:
    if isinstance(msg, HumanMessage):
        print(f"[用户] {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"[AI] {msg.content}")
    elif isinstance(msg, ToolMessage):
        print(f"[工具] {msg.content}")
```

## Claude Extended Thinking（思考过程）

思考过程出现在 **AIMessage** 的 `content` 中，格式为列表：

```python
# content 可能是字符串或列表
msg.content = [
    {
        "type": "thinking",  # 思考块
        "thinking": "用户问天气，我需要调用 get_weather 工具...",
        "thinking_depth": 5,
    },
    {
        "type": "text",      # 最终回复
        "text": "武汉今天晴天~"
    }
]
```

## 解析思考内容

```python
for msg in messages["messages"]:
    if isinstance(msg, AIMessage):
        content = msg.content
        if isinstance(content, str):
            print(f"[AI] {content}")
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "thinking":
                    print(f"[思考] {block['thinking']}")
                elif block.get("type") == "text":
                    print(f"[AI] {block['text']}")
```

## 提取思考内容函数

```python
def extract_thinking(messages):
    thoughts = []
    for msg in messages["messages"]:
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            for block in msg.content:
                if block.get("type") == "thinking":
                    thoughts.append(block["thinking"])
    return thoughts
```
