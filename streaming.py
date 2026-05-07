from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk, ToolMessage

open_ai_chat_model = init_chat_model(model="openai:MiniMax-M2.7",
                                     base_url="https://api.minimaxi.com/v1",
                                     api_key="sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58")

agent = create_deep_agent(model=open_ai_chat_model,
                          system_prompt="你是一个有用的助手，使用中文回答客户",
                          name="main_agent",
                          # debug=True,
                          subagents=[
                              {
                                  "name": "researcher",
                                  "description": "深入研究某一课题",
                                  "system_prompt": "你是一位严谨的研究者。",
                              },
                          ],
                          )
input_text: str = "使用sub agent 同时研究: 司马迁生平和项羽生平"
print()
last_tool_call_id = ''
last_tool_name = ''
state = {}
agent.astream_events()
for chunk in agent.stream({"messages": [{"role": "user", "content": input_text}]}, stream_mode=["updates", "messages"],
                          subgraphs=True, version="v2"):
    print(chunk)
    ns = chunk["ns"]
    is_subagent = len(ns) > 0
    ns_id = ns[0] if is_subagent else 'main:agent'
    stream_mode = chunk["type"]
    data = chunk["data"]
    print(stream_mode, ns, ns_id, is_subagent)
    print(type(data).__name__, len(data), data)

    if stream_mode == 'updates':
        pass
    if stream_mode == 'messages':
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            if token.tool_call_chunks:
                tool_call_chunk = token.tool_call_chunks[0]
                if tool_call_chunk.get("id"):
                    state[ns] = {"last_tool_call_id": tool_call_chunk.get("id"),
                                 "last_tool_name": tool_call_chunk.get("name")}
                tool_id = tool_call_chunk.get("id") or state[ns].get("last_tool_call_id")
                tool_name = tool_call_chunk.get("name") or state[ns].get("last_tool_name")
                tool_args = tool_call_chunk.get("args")

                print(f"tool_id:{tool_id},  tool_name:{tool_name}, tool_args:{tool_args}")

        if isinstance(token, ToolMessage):
            tool_id = token.tool_call_id
            tool_name = token.name
            tool_result = token.content
            print(f"tool_result: tool_id:{tool_id} tool_name:{token.name}, tool_result:{tool_result}")
            if state[ns].get("last_tool_call_id") != tool_id:
                print(f"移除最后的工具使用出错了: last_tool: {state[ns].get("last_tool_call_id")}, result_tool: {tool_id}")
            state[ns] = {"last_tool_call_id": "", "last_tool_name": ""}

    print()

print("all end~")
