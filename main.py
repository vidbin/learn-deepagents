from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

chat_model = init_chat_model(
    model='MiniMax-M2.7',
    model_provider='anthropic',
    base_url="https://api.minimaxi.com/anthropic",
    api_key="sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58"
)
def get_weather(city : str) -> str:
    """获取指定城市的天气"""
    return f"{city} 晴天~"


agent =  create_deep_agent(
    model=chat_model,
    system_prompt="你是一个专业的助手",
    tools=[get_weather],
    debug=True,
    name="main-agent"
)

chat_resp = agent.invoke(
    {"messages": [{"role": "user", "content": "武汉的天气怎么样?"}]}
)

print("for in messages")
for message in chat_resp["messages"]:
    if isinstance(message, HumanMessage):
        print(f"HumanMessage: {message}")
    elif isinstance(message, AIMessage):
        print(f"AIMessage: {message}")
    elif isinstance(message, ToolMessage):
        print(f"ToolMessage: {message}")
    else:
        print(f"OtherMessage: {message}")
    print()