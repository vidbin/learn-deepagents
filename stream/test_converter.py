"""测试 stream_converter — 模拟 OpenAI 和 Anthropic 两种格式的流式事件

使用 mock 对象模拟 AIMessageChunk，不依赖 langchain 包。
运行: python test_converter.py
"""

from stream_converter import StreamConverter, AgentEvent


# ─── Mock AIMessageChunk ──────────────────────────────────────────────────────

class MockChunk:
    """模拟 AIMessageChunk 的最小接口"""

    def __init__(
            self,
            content=None,
            tool_calls=None,
            invalid_tool_calls=None,
            tool_call_chunks=None,
            usage_metadata=None,
            chunk_position=None,
            additional_kwargs=None,
            response_metadata=None,
    ):
        self.content = content if content is not None else ""
        self.tool_calls = tool_calls or []
        self.invalid_tool_calls = invalid_tool_calls or []
        self.tool_call_chunks = tool_call_chunks or []
        self.usage_metadata = usage_metadata
        self.chunk_position = chunk_position
        self.additional_kwargs = additional_kwargs or {}
        self.response_metadata = response_metadata or {}


class MockToolMessage:
    """模拟 ToolMessage"""

    def __init__(self, name: str, content: str, tool_call_id: str, status: str = ""):
        self.type = "tool"
        self.name = name
        self.content = content
        self.tool_call_id = tool_call_id
        self.status = status


class MockAIMessage:
    """模拟 AIMessage（updates 中的完整消息）"""

    def __init__(self, content, tool_calls=None, usage_metadata=None):
        self.type = "ai"
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage_metadata = usage_metadata


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def print_events(events: list[AgentEvent], label: str = ""):
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
    for e in events:
        aid: str = e.data.get("agent_id", "?")[:12]
        aname: str = e.data.get("agent_name", "?")
        tag: str = f"[{aname}:{aid}]"
        if e.type == "thinking_delta":
            print(f"  {e.type} {tag}: {e.data['content'][:50]}...")
        elif e.type == "thinking_complete":
            print(f"  {e.type} {tag}: ({len(e.data['content'])} chars)")
        elif e.type == "text_delta":
            print(f"  {e.type} {tag}: {e.data['content'][:60]}")
        elif e.type == "text_complete":
            print(f"  {e.type} {tag}: ({len(e.data['content'])} chars)")
        elif e.type == "tool_start":
            print(f"  {e.type} {tag}: {e.data['tool_name']} (id={e.data['tool_use_id'][:20]})")
        elif e.type == "tool_result":
            print(f"  {e.type} {tag}: {e.data['tool_name']} -> {e.data['result'][:40]}")
        elif e.type == "tool_progress":
            print(f"  {e.type} {tag}: {e.data['tool_name']} args={e.data['args_delta'][:40]}...")
        elif e.type == "todos_update":
            print(f"  {e.type} {tag}: {len(e.data['todos'])} items")
        else:
            print(f"  {e.type} {tag}: {e.data}")


def test_openai_main_agent():
    """OpenAI 格式：主 agent thinking → text → 并行 tool_call"""
    converter = StreamConverter(turn_id="turn-001")
    all_events: list[AgentEvent] = []
    meta_openai = {"ls_provider": "openai", "ls_model_name": "gpt-4o", "langgraph_node": "model", "lc_agent_name": "main_agent"}

    chunks = [
        {"type": "messages", "ns": (), "data": (
            MockChunk(content="<think>\n用户想要"),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content="使用sub agent同时研究两个课题。"),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content="可以并行处理。\n</think>\n\n"),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content="好的，我来帮你同时研究这两个课题。"),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(
                content="",
                tool_call_chunks=[
                    {"name": "task", "args": '{"description": "研究司马迁', "id": "call_001", "index": 0, "type": "tool_call_chunk"},
                ],
            ),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": '的生平", "subagent_type": "researcher"}', "id": None, "index": 0, "type": "tool_call_chunk"},
                ],
            ),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(
                content="",
                tool_call_chunks=[
                    {"name": "task", "args": '{"description": "研究项羽', "id": "call_002", "index": 1, "type": "tool_call_chunk"},
                ],
            ),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(
                content="",
                tool_call_chunks=[
                    {"name": None, "args": '的生平", "subagent_type": "researcher"}', "id": None, "index": 1, "type": "tool_call_chunk"},
                ],
            ),
            meta_openai,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(
                content="",
                chunk_position="last",
                usage_metadata={"input_tokens": 1200, "output_tokens": 150, "total_tokens": 1350},
            ),
            meta_openai,
        )},
    ]

    for chunk in chunks:
        all_events.extend(converter.convert(chunk))

    print_events(all_events, "TEST 1: OpenAI 主 agent — thinking + text + 并行 tool_call")

    types = [e.type for e in all_events]
    assert "thinking_delta" in types
    assert "thinking_complete" in types
    assert "text_delta" in types
    assert "text_complete" in types
    assert types.count("tool_start") == 2

    # 验证所有事件都有 agent_id 和 agent_name
    for e in all_events:
        assert e.data["agent_id"] == "main_agent", f"{e.type} 缺少正确的 agent_id"
        assert e.data["agent_name"] == "main_agent", f"{e.type} 缺少正确的 agent_name"

    # 验证工具参数累积
    state = converter._get_agent(())
    assert "司马迁" in state.active_tools[0].args_buf
    assert "项羽" in state.active_tools[1].args_buf
    assert converter.total_usage["input_tokens"] == 1200

    # 验证无 node_enter/node_exit
    assert "node_enter" not in types
    assert "node_exit" not in types
    print("  PASS")


def test_anthropic_main_agent():
    """Anthropic 格式：主 agent thinking → tool_use（content list 格式）"""
    converter = StreamConverter(turn_id="turn-002")
    all_events: list[AgentEvent] = []
    meta_anthropic = {"ls_provider": "anthropic", "ls_model_name": "claude-sonnet-4-6", "langgraph_node": "model", "lc_agent_name": "main_agent"}

    chunks = [
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[]),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[{"thinking": "用户想要", "type": "thinking", "index": 0}]),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[{"thinking": "使用sub agent同时研究两个课题。", "type": "thinking", "index": 0}]),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[{"signature": "abc123def456", "type": "thinking", "index": 0}]),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(
                content=[{"id": "call_ant_001", "input": {}, "name": "task", "type": "tool_use", "index": 1}],
                tool_call_chunks=[{"name": "task", "args": "", "id": "call_ant_001", "index": 1, "type": "tool_call_chunk"}],
            ),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[{"partial_json": '{"description": "研究司马迁的生平', "type": "input_json_delta", "index": 1}]),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[{"partial_json": '", "subagent_type": "researcher"}', "type": "input_json_delta", "index": 1}]),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(
                content=[{"id": "call_ant_002", "input": {}, "name": "task", "type": "tool_use", "index": 2}],
                tool_call_chunks=[{"name": "task", "args": "", "id": "call_ant_002", "index": 2, "type": "tool_call_chunk"}],
            ),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[{"partial_json": '{"description": "研究项羽的生平", "subagent_type": "researcher"}', "type": "input_json_delta", "index": 2}]),
            meta_anthropic,
        )},
        {"type": "messages", "ns": (), "data": (
            MockChunk(content=[], chunk_position="last", usage_metadata={"input_tokens": 5836, "output_tokens": 261, "total_tokens": 6097}),
            meta_anthropic,
        )},
    ]

    for chunk in chunks:
        all_events.extend(converter.convert(chunk))

    print_events(all_events, "TEST 2: Anthropic 主 agent — thinking + 并行 tool_use")

    types = [e.type for e in all_events]
    assert "thinking_delta" in types
    assert "thinking_complete" in types
    assert types.count("tool_start") == 2

    # 验证 agent_id/agent_name
    for e in all_events:
        assert e.data["agent_id"] == "main_agent"
        assert e.data["agent_name"] == "main_agent"

    state = converter._get_agent(())
    assert "司马迁" in state.active_tools[1].args_buf
    assert "项羽" in state.active_tools[2].args_buf
    assert converter.total_usage["total_tokens"] == 6097
    print("  PASS")


def test_openai_subagent():
    """OpenAI 格式：并行子智能体 thinking + text"""
    converter = StreamConverter(turn_id="turn-003")
    all_events: list[AgentEvent] = []
    meta_sub_a = {"ls_provider": "openai", "ls_model_name": "gpt-4o", "lc_agent_name": "researcher"}
    meta_sub_b = {"ls_provider": "openai", "ls_model_name": "gpt-4o", "lc_agent_name": "researcher"}

    ns_a = ("tools:2cbc0d99-dffb-ae77-ae8e-c53b657bd4e0",)
    ns_b = ("tools:eb503023-f409-43b2-066e-29b3f540c20b",)

    chunks = [
        {"type": "messages", "ns": ns_a, "data": (MockChunk(content="<think>"), meta_sub_a)},
        {"type": "messages", "ns": ns_a, "data": (MockChunk(content="研究司马迁的生平需要系统性整理。"), meta_sub_a)},
        {"type": "messages", "ns": ns_b, "data": (MockChunk(content="<think>"), meta_sub_b)},
        {"type": "messages", "ns": ns_b, "data": (MockChunk(content="项羽是秦末重要人物。"), meta_sub_b)},
        {"type": "messages", "ns": ns_a, "data": (MockChunk(content="</think>\n\n司马迁（约前145年—？），字子长。"), meta_sub_a)},
        {"type": "messages", "ns": ns_a, "data": (MockChunk(content="西汉史学家、文学家。"), meta_sub_a)},
        {"type": "messages", "ns": ns_b, "data": (MockChunk(content="</think>\n\n项羽（前232年—前202年），名籍。"), meta_sub_b)},
        {"type": "messages", "ns": ns_a, "data": (
            MockChunk(content="", chunk_position="last", usage_metadata={"input_tokens": 800, "output_tokens": 200, "total_tokens": 1000}),
            meta_sub_a,
        )},
        {"type": "messages", "ns": ns_b, "data": (
            MockChunk(content="", chunk_position="last", usage_metadata={"input_tokens": 900, "output_tokens": 250, "total_tokens": 1150}),
            meta_sub_b,
        )},
    ]

    for chunk in chunks:
        all_events.extend(converter.convert(chunk))

    print_events(all_events, "TEST 3: OpenAI 并行子智能体 — 交错 thinking + text")

    # 验证两个子智能体独立（通过 agent_id 区分）
    sub_a_events = [e for e in all_events if e.data.get("agent_id") == "2cbc0d99-dffb-ae77-ae8e-c53b657bd4e0"]
    sub_b_events = [e for e in all_events if e.data.get("agent_id") == "eb503023-f409-43b2-066e-29b3f540c20b"]

    assert len(sub_a_events) > 0, "subagent A 应有事件"
    assert len(sub_b_events) > 0, "subagent B 应有事件"

    # 验证 agent_name
    assert all(e.data["agent_name"] == "researcher" for e in sub_a_events)
    assert all(e.data["agent_name"] == "researcher" for e in sub_b_events)

    # A 的 thinking 包含司马迁
    a_thinking = [e for e in sub_a_events if e.type == "thinking_delta"]
    assert any("司马迁" in e.data["content"] for e in a_thinking)

    # B 的 thinking 包含项羽
    b_thinking = [e for e in sub_b_events if e.type == "thinking_delta"]
    assert any("项羽" in e.data["content"] for e in b_thinking)

    # A 的 text 包含子长
    a_text = [e for e in sub_a_events if e.type == "text_delta"]
    assert any("子长" in e.data["content"] for e in a_text)

    assert converter.total_usage["total_tokens"] == 2150
    print("  PASS")


def test_anthropic_subagent_with_tools():
    """Anthropic 格式：子智能体内部调用工具"""
    converter = StreamConverter(turn_id="turn-004")
    all_events: list[AgentEvent] = []
    meta = {"ls_provider": "anthropic", "ls_model_name": "claude-sonnet-4-6", "lc_agent_name": "researcher"}

    ns_sub = ("tools:3915a820-1ff7-ef6e-f827-4c5120e26f49",)

    chunks = [
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content=[{"thinking": "需要搜索相关资料", "type": "thinking", "index": 0}]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content=[{"signature": "sig_xyz", "type": "thinking", "index": 0}]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content=[{"text": "让我搜索相关资料。", "type": "text", "index": 1}]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(
                content=[{"id": "tool_bash_001", "input": {}, "name": "bash", "type": "tool_use", "index": 2}],
                tool_call_chunks=[{"name": "bash", "args": "", "id": "tool_bash_001", "index": 2, "type": "tool_call_chunk"}],
            ),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content=[{"partial_json": '{"command": "echo hello"}', "type": "input_json_delta", "index": 2}]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content=[], chunk_position="last", usage_metadata={"input_tokens": 500, "output_tokens": 80, "total_tokens": 580}),
            meta,
        )},
        {"type": "updates", "ns": ns_sub, "data": {
            "tools": {
                "messages": [MockToolMessage(name="bash", content="hello\n", tool_call_id="tool_bash_001")],
            }
        }},
    ]

    for chunk in chunks:
        all_events.extend(converter.convert(chunk))

    print_events(all_events, "TEST 4: Anthropic 子智能体 — thinking + text + tool_use + tool_result")

    types = [e.type for e in all_events]
    assert "thinking_delta" in types
    assert "thinking_complete" in types
    assert "text_delta" in types
    assert "text_complete" in types
    assert "tool_start" in types
    assert "tool_result" in types
    assert "node_enter" not in types
    assert "node_exit" not in types

    # 验证 agent_id/agent_name
    for e in all_events:
        assert e.data["agent_id"] == "3915a820-1ff7-ef6e-f827-4c5120e26f49"
        assert e.data["agent_name"] == "researcher"

    tool_start_evt = next(e for e in all_events if e.type == "tool_start")
    assert tool_start_evt.data["tool_name"] == "bash"

    tool_result_evt = next(e for e in all_events if e.type == "tool_result")
    assert tool_result_evt.data["result"] == "hello\n"
    print("  PASS")


def test_updates_todos():
    """updates 事件中的 todos 提取"""
    converter = StreamConverter(turn_id="turn-005")

    ns_sub = ("tools:abc123",)
    # 先发一个 messages 让 agent_name 被记录
    converter.convert({"type": "messages", "ns": ns_sub, "data": (
        MockChunk(content="计划中"),
        {"ls_provider": "openai", "lc_agent_name": "planner"},
    )})

    chunk = {"type": "updates", "ns": ns_sub, "data": {
        "model": {
            "messages": [MockAIMessage(
                content="我来制定计划。",
                tool_calls=[{"name": "write_todos", "args": {"todos": [
                    {"content": "研究背景", "status": "in_progress"},
                    {"content": "撰写报告", "status": "pending"},
                ]}, "id": "call_todos_001", "type": "tool_call"}],
            )],
            "todos": [
                {"content": "研究背景", "status": "in_progress"},
                {"content": "撰写报告", "status": "pending"},
            ],
        }
    }}

    events = converter.convert(chunk)
    print_events(events, "TEST 5: updates 中的 todos 提取")

    todos_events = [e for e in events if e.type == "todos_update"]
    assert len(todos_events) == 1
    assert len(todos_events[0].data["todos"]) == 2
    assert todos_events[0].data["agent_id"] == "abc123"
    assert todos_events[0].data["agent_name"] == "planner"
    print("  PASS")


def test_openai_parallel_tools_in_subagent():
    """OpenAI 格式：子智能体内并行调用多个工具"""
    converter = StreamConverter(turn_id="turn-006")
    all_events: list[AgentEvent] = []
    meta = {"ls_provider": "openai", "ls_model_name": "gpt-4o", "lc_agent_name": "researcher"}

    ns_sub = ("tools:sub_parallel_001",)

    chunks = [
        {"type": "messages", "ns": ns_sub, "data": (MockChunk(content="我需要同时搜索两个资料。"), meta)},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content="", tool_call_chunks=[
                {"name": "bash", "args": '{"command": "curl http://api1', "id": "call_p1", "index": 0, "type": "tool_call_chunk"},
            ]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content="", tool_call_chunks=[
                {"name": "read_file", "args": '{"path": "/tmp/data', "id": "call_p2", "index": 1, "type": "tool_call_chunk"},
            ]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content="", tool_call_chunks=[
                {"name": None, "args": '.com"}', "id": None, "index": 0, "type": "tool_call_chunk"},
            ]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (
            MockChunk(content="", tool_call_chunks=[
                {"name": None, "args": '.txt"}', "id": None, "index": 1, "type": "tool_call_chunk"},
            ]),
            meta,
        )},
        {"type": "messages", "ns": ns_sub, "data": (MockChunk(content="", chunk_position="last"), meta)},
    ]

    for chunk in chunks:
        all_events.extend(converter.convert(chunk))

    print_events(all_events, "TEST 6: OpenAI 子智能体内并行工具调用")

    tool_starts = [e for e in all_events if e.type == "tool_start"]
    assert len(tool_starts) == 2
    assert tool_starts[0].data["tool_name"] == "bash"
    assert tool_starts[1].data["tool_name"] == "read_file"

    # 验证参数完整
    state = converter._get_agent(ns_sub)
    assert state.active_tools[0].args_buf == '{"command": "curl http://api1.com"}'
    assert state.active_tools[1].args_buf == '{"path": "/tmp/data.txt"}'

    # 验证 agent_id/agent_name
    assert all(e.data["agent_id"] == "sub_parallel_001" for e in all_events)
    assert all(e.data["agent_name"] == "researcher" for e in all_events)
    print("  PASS")


def test_tool_progress():
    """tool_progress 事件：每个非空 delta 都产生 progress"""
    converter = StreamConverter(turn_id="turn-007")
    all_events: list[AgentEvent] = []
    meta = {"ls_provider": "anthropic", "ls_model_name": "claude-sonnet-4-6", "lc_agent_name": "main_agent"}

    # tool_use 开始
    all_events.extend(converter.convert({"type": "messages", "ns": (), "data": (
        MockChunk(content=[{"id": "tool_write_001", "input": {}, "name": "write_file", "type": "tool_use", "index": 0}],
                  tool_call_chunks=[{"name": "write_file", "args": "", "id": "tool_write_001", "index": 0, "type": "tool_call_chunk"}]),
        meta,
    )}))

    # 空 delta — 不应产生 progress
    all_events.extend(converter.convert({"type": "messages", "ns": (), "data": (
        MockChunk(content=[{"partial_json": "", "type": "input_json_delta", "index": 0}]),
        meta,
    )}))

    # 第一个有效 delta
    all_events.extend(converter.convert({"type": "messages", "ns": (), "data": (
        MockChunk(content=[{"partial_json": '{"path": "/tmp/file.txt", "content": "hello', "type": "input_json_delta", "index": 0}]),
        meta,
    )}))

    # 第二个有效 delta
    all_events.extend(converter.convert({"type": "messages", "ns": (), "data": (
        MockChunk(content=[{"partial_json": ' world"}', "type": "input_json_delta", "index": 0}]),
        meta,
    )}))

    print_events(all_events, "TEST 7: tool_progress 事件（无节流）")

    progress_events = [e for e in all_events if e.type == "tool_progress"]
    assert len(progress_events) == 2, f"应有 2 个 tool_progress，实际 {len(progress_events)}"

    # 验证 args_delta 递增（累积）
    assert progress_events[0].data["args_delta"] == '{"path": "/tmp/file.txt", "content": "hello'
    assert progress_events[1].data["args_delta"] == '{"path": "/tmp/file.txt", "content": "hello world"}'

    # 验证 agent_id/agent_name
    assert progress_events[0].data["agent_id"] == "main_agent"
    assert progress_events[0].data["agent_name"] == "main_agent"
    assert progress_events[0].data["tool_name"] == "write_file"
    print("  PASS")


def test_tool_message_in_messages_stream():
    """ToolMessage 出现在 messages 流中应被识别为 tool_result，而非 AI 输出"""
    converter = StreamConverter(turn_id="turn-008")
    all_events: list[AgentEvent] = []
    meta = {"ls_provider": "openai", "lc_agent_name": "main_agent"}

    # 模拟 ToolMessage 通过 messages 流返回（子智能体 task 工具的结果）
    tool_msg = MockToolMessage(
        name="task",
        content='<think>\n分析项羽生平\n</think>\n\n# 项羽生平研究报告\n\n项羽（前232年—前202年）...',
        tool_call_id="call_function_scocb94gva9d_2",
    )

    chunk = {"type": "messages", "ns": (), "data": (tool_msg, meta)}
    all_events.extend(converter.convert(chunk))

    print_events(all_events, "TEST 8: ToolMessage 在 messages 流中 — 应为 tool_result")

    # 应只产生一个 tool_result，不应有 thinking_delta/text_delta
    types = [e.type for e in all_events]
    assert "tool_result" in types, "应产生 tool_result"
    assert "thinking_delta" not in types, "不应产生 thinking_delta"
    assert "text_delta" not in types, "不应产生 text_delta"
    assert "thinking_complete" not in types, "不应产生 thinking_complete"

    result_evt = all_events[0]
    assert result_evt.data["tool_use_id"] == "call_function_scocb94gva9d_2"
    assert result_evt.data["tool_name"] == "task"
    assert "项羽" in result_evt.data["result"]
    assert result_evt.data["agent_id"] == "main_agent"
    assert result_evt.data["agent_name"] == "main_agent"
    print("  PASS")


def test_tool_result_deduplication():
    """messages 和 updates 中同一 tool_call_id 的 tool_result 不应重复"""
    converter = StreamConverter(turn_id="turn-009")
    all_events: list[AgentEvent] = []

    ns_sub = ("tools:dedup_test_001",)
    meta = {"ls_provider": "openai", "lc_agent_name": "researcher"}

    # 先让 agent 状态初始化
    converter.convert({"type": "messages", "ns": ns_sub, "data": (
        MockChunk(content="搜索中..."), meta,
    )})

    # ToolMessage 通过 messages 流到达
    tool_msg = MockToolMessage(
        name="bash",
        content="hello world\n",
        tool_call_id="call_dedup_001",
    )
    events1 = converter.convert({"type": "messages", "ns": ns_sub, "data": (tool_msg, meta)})
    all_events.extend(events1)

    # 同一个 tool_result 通过 updates 流再次到达
    events2 = converter.convert({"type": "updates", "ns": ns_sub, "data": {
        "tools": {
            "messages": [MockToolMessage(name="bash", content="hello world\n", tool_call_id="call_dedup_001")],
        }
    }})
    all_events.extend(events2)

    print_events(all_events, "TEST 9: tool_result 去重 — messages + updates 不重复")

    tool_results = [e for e in all_events if e.type == "tool_result"]
    assert len(tool_results) == 1, f"应只有 1 个 tool_result，实际 {len(tool_results)}"
    assert tool_results[0].data["tool_use_id"] == "call_dedup_001"
    assert tool_results[0].data["agent_name"] == "researcher"
    print("  PASS")


# ─── 运行所有测试 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_openai_main_agent()
    test_anthropic_main_agent()
    test_openai_subagent()
    test_anthropic_subagent_with_tools()
    test_updates_todos()
    test_openai_parallel_tools_in_subagent()
    test_tool_progress()
    test_tool_message_in_messages_stream()
    test_tool_result_deduplication()

    print(f"\n{'=' * 60}")
    print("  ALL 9 TESTS PASSED")
    print(f"{'=' * 60}")
