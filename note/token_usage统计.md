# Q: 如何进行 token 消耗统计？

## 从 AIMessage 获取（完整响应）

```python
response = agent.invoke(input_content)

for msg in response.get("messages", []):
    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
        usage = msg.usage_metadata
        print(f"输入 tokens: {usage['input_tokens']}")
        print(f"输出 tokens: {usage['output_tokens']}")
        print(f"总 tokens: {usage['total_tokens']}")
```

## 从流式输出获取

```python
from langchain_core.messages.ai import add_usage

total = None
for chunk in agent.stream(input_content, stream_mode=["messages"]):
    if chunk["type"] == "messages":
        msg, _ = chunk["data"]
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total = add_usage(total, msg.usage_metadata)

if total:
    print(f"Input: {total['input_tokens']}")
    print(f"Output: {total['output_tokens']}")
```

## 使用 add_usage 累加

```python
from langchain_core.messages.ai import add_usage, UsageMetadata

# 手动合并
def merge_usage(left: UsageMetadata | None, right: UsageMetadata | None) -> UsageMetadata:
    return add_usage(left, right)
```

## 详细 token 分析（如果提供者支持）

```python
if final_usage and "input_token_details" in final_usage:
    details = final_usage["input_token_details"]
    print(f"Cache hit: {details.get('cache_read', 0)}")
    print(f"Cache miss: {details.get('cache_creation', 0)}")

if "output_token_details" in final_usage:
    output_details = final_usage["output_token_details"]
    if "reasoning" in output_details:
        print(f"Reasoning tokens: {output_details['reasoning']}")
```

## 流式结束后的最终统计

通常 `usage_metadata` 只在流式结束后的最终消息中完整出现：

```python
final_usage = None
for chunk in agent.stream(input_content, stream_mode=["messages"]):
    if chunk["type"] == "messages":
        msg, _ = chunk["data"]
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            final_usage = msg.usage_metadata

print(final_usage)
```

## messages 模式下的 metadata

```python
for chunk in agent.stream(input_content, stream_mode=["messages"]):
    if chunk["type"] == "messages":
        msg, meta = chunk["data"]
        # meta 主要包含 LangGraph 信息，不是 token 统计
        print(f"Step: {meta.get('langgraph_step')}")
        print(f"Node: {meta.get('langgraph_node')}")
```

真正的 token 统计在消息的 `usage_metadata` 字段中。