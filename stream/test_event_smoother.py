"""EventSmoother 测试"""

import pytest

from stream_converter import AgentEvent
from event_smoother import EventSmoother


def _make_event(type: str, content_field: str, content: str, agent_id: str = "main_agent") -> AgentEvent:
    data = {"agent_id": agent_id, "agent_name": "main_agent", content_field: content}
    return AgentEvent(type=type, turn_id="turn-1", data=data)


# ─── TEST 1: 短 content 不拆分，缓冲到 flush ───

def test_short_content_buffered():
    smoother = EventSmoother(chunk_size=20)
    event = _make_event("text_delta", "content", "hello")
    result = smoother.push([event])
    assert result == []
    flushed = smoother.flush()
    assert len(flushed) == 1
    assert flushed[0].data["content"] == "hello"


# ─── TEST 2: 恰好等于 chunk_size 直接输出 ───

def test_exact_chunk_size():
    smoother = EventSmoother(chunk_size=5)
    event = _make_event("text_delta", "content", "abcde")
    result = smoother.push([event])
    assert len(result) == 1
    assert result[0].data["content"] == "abcde"
    assert smoother.flush() == []


# ─── TEST 3: 长 content 拆分为多个均匀事件 ───

def test_long_content_split():
    smoother = EventSmoother(chunk_size=10)
    event = _make_event("text_delta", "content", "a" * 25)
    result = smoother.push([event])
    assert len(result) == 2
    assert result[0].data["content"] == "a" * 10
    assert result[1].data["content"] == "a" * 10
    flushed = smoother.flush()
    assert len(flushed) == 1
    assert flushed[0].data["content"] == "a" * 5


# ─── TEST 4: 非 content 事件原样透传 ───

def test_passthrough_events():
    smoother = EventSmoother(chunk_size=10)
    events = [
        AgentEvent(type="tool_start", turn_id="turn-1", data={"tool_name": "bash"}),
        AgentEvent(type="tool_result", turn_id="turn-1", data={"result": "ok"}),
        AgentEvent(type="text_complete", turn_id="turn-1", data={"content": "full text"}),
        AgentEvent(type="thinking_complete", turn_id="turn-1", data={"content": "full thinking"}),
    ]
    result = smoother.push(events)
    assert result == events


# ─── TEST 5: buffer 跨 push 拼接 ───

def test_cross_push_buffer():
    smoother = EventSmoother(chunk_size=10)
    e1 = _make_event("text_delta", "content", "abcdef")  # 6 chars, buffered
    r1 = smoother.push([e1])
    assert r1 == []

    e2 = _make_event("text_delta", "content", "ghij")  # 6+4=10, emit one chunk
    r2 = smoother.push([e2])
    assert len(r2) == 1
    assert r2[0].data["content"] == "abcdefghij"
    assert smoother.flush() == []


# ─── TEST 6: tool_progress 的 args_delta 平滑 ───

def test_tool_progress_smoothing():
    smoother = EventSmoother(chunk_size=5)
    event = AgentEvent(
        type="tool_progress",
        turn_id="turn-1",
        data={"agent_id": "main_agent", "agent_name": "main_agent",
              "tool_use_id": "tc-1", "tool_name": "bash", "args_delta": "0123456789ab"},
    )
    result = smoother.push([event])
    assert len(result) == 2
    assert result[0].data["args_delta"] == "01234"
    assert result[1].data["args_delta"] == "56789"
    assert result[0].data["tool_use_id"] == "tc-1"
    flushed = smoother.flush()
    assert len(flushed) == 1
    assert flushed[0].data["args_delta"] == "ab"


# ─── TEST 7: thinking_delta 平滑 ───

def test_thinking_delta_smoothing():
    smoother = EventSmoother(chunk_size=8)
    event = _make_event("thinking_delta", "content", "x" * 20)
    result = smoother.push([event])
    assert len(result) == 2
    assert all(r.type == "thinking_delta" for r in result)
    assert result[0].data["content"] == "x" * 8
    assert result[1].data["content"] == "x" * 8
    flushed = smoother.flush()
    assert flushed[0].data["content"] == "x" * 4


# ─── TEST 8: 多 agent 的 buffer 互不干扰 ───

def test_multi_agent_buffers():
    smoother = EventSmoother(chunk_size=10)
    e1 = _make_event("text_delta", "content", "aaaa", agent_id="agent-1")
    e2 = _make_event("text_delta", "content", "bbbb", agent_id="agent-2")
    smoother.push([e1, e2])

    e3 = _make_event("text_delta", "content", "cccccc", agent_id="agent-1")  # 4+6=10
    e4 = _make_event("text_delta", "content", "dddddd", agent_id="agent-2")  # 4+6=10
    result = smoother.push([e3, e4])
    assert len(result) == 2
    assert result[0].data["content"] == "aaaacccccc"
    assert result[0].data["agent_id"] == "agent-1"
    assert result[1].data["content"] == "bbbbdddddd"
    assert result[1].data["agent_id"] == "agent-2"


# ─── TEST 9: 混合事件序列 ───

def test_mixed_events():
    smoother = EventSmoother(chunk_size=5)
    events = [
        AgentEvent(type="tool_start", turn_id="turn-1", data={"tool_name": "bash"}),
        _make_event("text_delta", "content", "hello world!"),  # 12 chars -> 2 chunks + 2 buffered
        AgentEvent(type="tool_result", turn_id="turn-1", data={"result": "ok"}),
    ]
    result = smoother.push(events)
    assert result[0].type == "tool_start"
    assert result[1].type == "text_delta"
    assert result[1].data["content"] == "hello"
    assert result[2].type == "text_delta"
    assert result[2].data["content"] == " worl"
    assert result[3].type == "tool_result"
    flushed = smoother.flush()
    assert flushed[0].data["content"] == "d!"
