from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

open_ai_chat_model = init_chat_model(model="openai:MiniMax-M2.7",
                                     base_url="https://api.minimaxi.com/v1",
                                     api_key="sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58")

agent = create_deep_agent(
    model=open_ai_chat_model,
    system_prompt=(
        "You are a project coordinator. Always delegate research tasks "
        "to your researcher subagent using the task tool. Keep your final response to one sentence."
    ),
    subagents=[
        {
            "name": "researcher",
            "description": "Researches topics thoroughly",
            "system_prompt": (
                "You are a thorough researcher. Research the given topic "
                "and provide a concise summary in 2-3 sentences."
            ),
        },
    ],
)

for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "Write a short summary about AI safety"}]},
        stream_mode="updates",
        subgraphs=True,
        version="v2",
):
    if chunk["type"] == "updates":
        # 主 agent 更新（空 namespace）
        if not chunk["ns"]:
            for node_name, data in chunk["data"].items():
                if node_name == "tools":
                    # 返回给主 agent 的 subagent 结果
                    for msg in data.get("messages", []):
                        if msg.type == "tool":
                            print(f"\nSubagent complete: {msg.name}")
                            print(f"  Result: {str(msg.content)[:200]}...")
                else:
                    print(f"[main agent] step: {node_name}")

        # Subagent 更新（非空 namespace）
        else:
            for node_name, data in chunk["data"].items():
                print(f"  [{chunk['ns'][0]}] step: {node_name}")
