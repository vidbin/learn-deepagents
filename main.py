from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware import SkillsMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver

from HotSkillMiddleware import HotSkillMiddleware

anthropic_chat_model = init_chat_model(
    model='anthropic:MiniMax-M2.7',
    # model_provider='anthropic',
    base_url="https://api.minimaxi.com/anthropic",
    api_key="sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58"
)

open_ai_chat_model = init_chat_model(
    model='openai:MiniMax-M2.7',
    # model_provider='anthropic',
    base_url="https://api.minimaxi.com/v1",
    api_key="sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58"
)
def get_weather(city : str) -> str:
    """获取指定城市的天气"""
    return f"{city} 晴天~"
backend = FilesystemBackend(root_dir="E:/workspace/learn-deepagents",virtual_mode=True)
print(backend.ls_info("/skills"))
checkpointer = MemorySaver()
agent =  create_deep_agent(
    model=open_ai_chat_model,
    system_prompt="你是一个专业的助手,永远不要使用sub agent",
    tools=[get_weather],
    checkpointer=checkpointer,
    backend=backend,
    middleware=[
        HotSkillMiddleware(
            backend=backend,
            sources=["/skills/"]
        )
    ],
    name="main-agent"
)
seq :int = 1
while True:
    input_text = input("user: ")
    for chunk in agent.stream({"messages":[HumanMessage(input_text)]},config={"configurable": {"thread_id": f"t-1"}}, subgraphs=True, stream_mode="updates"):
        print(chunk)
    print()
    seq += 1