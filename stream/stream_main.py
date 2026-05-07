from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from stream_converter import StreamConverter, AgentEvent


def stream_agent_response(agent, messages, turn_id: str):
    converter = StreamConverter(turn_id=turn_id)

    # 用 stream() 替代 astream_events()
    for chunk in agent.stream(
            {"messages": messages},
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
    ):
        # print(chunk)
        # chunk 格式: {"type": "updates"|"messages", "ns": tuple, "data": ...}
        events: list[AgentEvent] = converter.convert(chunk)
        for event in events:
            # 转为 SSE 推送给前端
            print(event)
        print()
    # 流结束后获取累积 usage
    usage = converter.total_usage
    print(f"usage: {usage}")


def add_num(a: int, b: int) -> int:
    """计算两个数相加"""
    return a + b


if __name__ == '__main__':
    open_ai_chat_model = init_chat_model(model="openai:MiniMax-M2.7",
                                         base_url="https://api.minimaxi.com/v1",
                                         api_key="sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58")
    anthropic_chat_model = init_chat_model(model="anthropic:MiniMax-M2.7",
                                         base_url="https://api.minimaxi.com/anthropic",
                                         api_key="sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58")


    deep_agent = create_deep_agent(model=anthropic_chat_model,
                                   system_prompt="你是一个有用的助手，使用中文回答客户",
                                   name="main_agent",
                                   # debug=True,
                                   subagents=[
                                       {
                                           "name": "researcher",
                                           "description": "深入研究某一课题",
                                           "system_prompt": "你是一位严谨的研究者。",
                                       },
                                       {
                                           "name": "add_calc",
                                           "description": "加法运算器",
                                           "system_prompt": "你是一个两数加法计算助手，请使用add_num 工具进行两数加法运算",
                                           "tools":[add_num]
                                       }
                                   ],
                                   )
    # "使用sub agent同时研究: 司马迁生平和项羽生平，每一个sub agent的都列好一个研究要点的待办列表，然后依次完成"
    stream_agent_response(deep_agent, HumanMessage("使用待办列表来完成 1 + 2 、  2 + 4、3 + 6 、80 + 18 的计算"),
                          'turn_001')
