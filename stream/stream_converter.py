"""stream() API → AgentEvent 事件转换器

基于 LangGraph stream(stream_mode=["updates", "messages"], subgraphs=True, version="v2")
统一处理 OpenAI 和 Anthropic 两种格式的流式事件。

每个 AgentEvent.data 都包含 agent_id 和 agent_name：
  - 主 agent: agent_id="main_agent", agent_name="main_agent"
  - 子 agent: agent_id=ns中的tools:UUID, agent_name=metadata中的lc_agent_name
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentEvent:
    type: str
    turn_id: str
    data: dict


@dataclass
class ToolCallState:
    """跟踪单个工具调用的流式状态"""
    id: str
    name: str
    args_buf: str = ""
    complete: bool = False


@dataclass
class AgentStreamState:
    """每个 agent（主/子）独立的流式状态"""
    thinking_buf: str = ""
    text_buf: str = ""
    in_thinking: bool = False
    thinking_done: bool = False
    active_tools: dict[int, ToolCallState] = field(default_factory=dict)
    # agent 名称（从 metadata 中获取）
    agent_name: str = ""


@dataclass
class StreamConverter:
    """顶层转换器，管理所有 agent 的状态"""
    turn_id: str = ""
    agents: dict[tuple, AgentStreamState] = field(default_factory=dict)
    total_usage: dict = field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0
    })
    # 已发出 tool_result 的 tool_call_id 集合，用于去重
    emitted_tool_results: set[str] = field(default_factory=set)

    def _get_agent(self, ns: tuple) -> AgentStreamState:
        if ns not in self.agents:
            self.agents[ns] = AgentStreamState()
        return self.agents[ns]

    def _agent_id(self, ns: tuple) -> str:
        """获取 agent_id：主 agent 返回 'main_agent'，子 agent 返回 ns 中的 UUID"""
        for seg in ns:
            if seg.startswith("tools:"):
                return seg.split(":", 1)[1]
        return "main_agent"

    def _is_subagent(self, ns: tuple) -> bool:
        return any(seg.startswith("tools:") for seg in ns)

    def _base_data(self, ns: tuple, state: AgentStreamState) -> dict:
        """构建每个事件的基础 data（包含 agent_id 和 agent_name）"""
        return {
            "agent_id": self._agent_id(ns),
            "agent_name": state.agent_name or "main_agent",
        }

    def convert(self, chunk: dict) -> list[AgentEvent]:
        """转换单个 stream chunk 为 AgentEvent 列表"""
        chunk_type: str = chunk.get("type", "")
        if chunk_type == "messages":
            return self._handle_messages(chunk)
        elif chunk_type == "updates":
            return self._handle_updates(chunk)
        return []

    # ─── messages 处理 ────────────────────────────────────────────────────

    def _handle_messages(self, chunk: dict) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        ns: tuple = chunk.get("ns", ())
        data = chunk.get("data")
        if not data or not isinstance(data, tuple) or len(data) < 2:
            return events

        token = data[0]
        metadata: dict = data[1] if len(data) > 1 else {}
        state: AgentStreamState = self._get_agent(ns)

        # ToolMessage 检测：tool 执行结果，直接发 tool_result
        token_type: str = getattr(token, "type", "")
        if token_type == "tool" or hasattr(token, "tool_call_id"):
            tool_call_id: str = getattr(token, "tool_call_id", "") or ""
            if tool_call_id and tool_call_id not in self.emitted_tool_results:
                self.emitted_tool_results.add(tool_call_id)
                evt_data: dict = {
                    **self._base_data(ns, state),
                    "tool_use_id": tool_call_id,
                    "tool_name": getattr(token, "name", "") or "",
                    "result": str(getattr(token, "content", "")),
                    "is_error": getattr(token, "status", "") == "error",
                }
                events.append(AgentEvent(type="tool_result", turn_id=self.turn_id, data=evt_data))
            return events

        provider: str = metadata.get("ls_provider", "")

        # 记录 agent_name
        agent_name: str = metadata.get("lc_agent_name", "")
        if agent_name and not state.agent_name:
            state.agent_name = agent_name
        if not state.agent_name:
            state.agent_name = "main_agent"

        # chunk_position='last' 标记模型输出结束
        chunk_position: str = getattr(token, "chunk_position", "") or ""
        if chunk_position == "last":
            events.extend(self._flush_agent(ns, state))
            um = getattr(token, "usage_metadata", None)
            if um:
                self.total_usage["input_tokens"] += um.get("input_tokens", 0) or 0
                self.total_usage["output_tokens"] += um.get("output_tokens", 0) or 0
                self.total_usage["total_tokens"] += um.get("total_tokens", 0) or 0
            return events

        content = getattr(token, "content", None)
        tool_call_chunks = getattr(token, "tool_call_chunks", None) or []

        # ─── 处理 content ───
        handled_tool_in_content: bool = False
        if content:
            if provider == "anthropic":
                handled_tool_in_content = isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") in ("tool_use", "input_json_delta")
                    for b in content
                )
                events.extend(self._parse_anthropic_content(content, ns, state))
            else:
                events.extend(self._parse_openai_content(content, ns, state))

        # ─── 处理 tool_call_chunks（Anthropic 已在 content 中处理则跳过）───
        if tool_call_chunks and not handled_tool_in_content:
            if state.text_buf and not state.in_thinking:
                events.extend(self._emit_text_complete(ns, state))
            elif state.in_thinking:
                events.extend(self._emit_thinking_complete(ns, state))

            for tc in tool_call_chunks:
                idx: int = tc.get("index", 0)
                name: str | None = tc.get("name")
                args: str = tc.get("args", "") or ""
                tc_id: str | None = tc.get("id")

                if name is not None and tc_id is not None:
                    state.active_tools[idx] = ToolCallState(id=tc_id, name=name, args_buf=args)
                    evt_data: dict = {
                        **self._base_data(ns, state),
                        "tool_use_id": tc_id,
                        "tool_name": name,
                        "tool_type": "mcp" if name.startswith("mcp_") else "builtin",
                        "input": {},
                    }
                    events.append(AgentEvent(type="tool_start", turn_id=self.turn_id, data=evt_data))
                elif idx in state.active_tools:
                    state.active_tools[idx].args_buf += args
                    events.extend(self._maybe_tool_progress(ns, state, idx))

        return events

    # ─── tool_progress ────────────────────────────────────────────────────

    def _maybe_tool_progress(self, ns: tuple, state: AgentStreamState, idx: int) -> list[AgentEvent]:
        tool: ToolCallState = state.active_tools[idx]
        if not tool.args_buf:
            return []
        evt_data: dict = {
            **self._base_data(ns, state),
            "tool_use_id": tool.id,
            "tool_name": tool.name,
            "args_delta": tool.args_buf,
        }
        return [AgentEvent(type="tool_progress", turn_id=self.turn_id, data=evt_data)]

    # ─── Anthropic content 解析 ───────────────────────────────────────────

    def _parse_anthropic_content(
        self, content: list, ns: tuple, state: AgentStreamState
    ) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        if not isinstance(content, list):
            return events

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type: str = block.get("type", "")

            if block_type == "thinking":
                if "signature" in block:
                    events.extend(self._emit_thinking_complete(ns, state))
                elif "thinking" in block:
                    thinking_text: str = block["thinking"]
                    if not state.in_thinking:
                        state.in_thinking = True
                    state.thinking_buf += thinking_text
                    evt_data: dict = {**self._base_data(ns, state), "content": thinking_text}
                    events.append(AgentEvent(type="thinking_delta", turn_id=self.turn_id, data=evt_data))

            elif block_type == "text":
                text: str = block.get("text", "")
                if text:
                    state.text_buf += text
                    evt_data = {**self._base_data(ns, state), "content": text}
                    events.append(AgentEvent(type="text_delta", turn_id=self.turn_id, data=evt_data))

            elif block_type == "input_json_delta":
                idx: int = block.get("index", 0)
                partial: str = block.get("partial_json", "")
                if idx in state.active_tools:
                    state.active_tools[idx].args_buf += partial
                    events.extend(self._maybe_tool_progress(ns, state, idx))

            elif block_type == "tool_use":
                tc_id: str = block.get("id", "")
                name: str = block.get("name", "")
                idx = block.get("index", 0)
                state.active_tools[idx] = ToolCallState(id=tc_id, name=name)
                evt_data = {
                    **self._base_data(ns, state),
                    "tool_use_id": tc_id,
                    "tool_name": name,
                    "tool_type": "mcp" if name.startswith("mcp_") else "builtin",
                    "input": {},
                }
                events.append(AgentEvent(type="tool_start", turn_id=self.turn_id, data=evt_data))

        return events

    # ─── OpenAI content 解析（<think> 标签） ──────────────────────────────

    def _parse_openai_content(
        self, content, ns: tuple, state: AgentStreamState
    ) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        if not isinstance(content, str) or not content:
            return events

        remaining: str = content

        while remaining:
            if state.in_thinking:
                end_idx: int = remaining.find("</think>")
                if end_idx == -1:
                    state.thinking_buf += remaining
                    evt_data: dict = {**self._base_data(ns, state), "content": remaining}
                    events.append(AgentEvent(type="thinking_delta", turn_id=self.turn_id, data=evt_data))
                    remaining = ""
                else:
                    thinking_part: str = remaining[:end_idx]
                    if thinking_part:
                        state.thinking_buf += thinking_part
                        evt_data = {**self._base_data(ns, state), "content": thinking_part}
                        events.append(AgentEvent(type="thinking_delta", turn_id=self.turn_id, data=evt_data))
                    events.extend(self._emit_thinking_complete(ns, state))
                    remaining = remaining[end_idx + len("</think>"):]
            else:
                start_idx: int = remaining.find("<think>")
                if start_idx == -1:
                    text: str = remaining.lstrip("\n") if state.thinking_done and not state.text_buf else remaining
                    if text:
                        state.text_buf += text
                        evt_data = {**self._base_data(ns, state), "content": text}
                        events.append(AgentEvent(type="text_delta", turn_id=self.turn_id, data=evt_data))
                    remaining = ""
                else:
                    before: str = remaining[:start_idx]
                    if before:
                        state.text_buf += before
                        evt_data = {**self._base_data(ns, state), "content": before}
                        events.append(AgentEvent(type="text_delta", turn_id=self.turn_id, data=evt_data))
                    state.in_thinking = True
                    remaining = remaining[start_idx + len("<think>"):]
                    if remaining.startswith("\n"):
                        remaining = remaining[1:]

        return events

    # ─── updates 处理 ─────────────────────────────────────────────────────

    def _handle_updates(self, chunk: dict) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        ns: tuple = chunk.get("ns", ())
        data: dict = chunk.get("data", {})
        state: AgentStreamState = self._get_agent(ns)

        for node_name, node_data in data.items():
            if node_data is None:
                continue

            # 从 updates 中提取 tool_result（tools 节点完成时），去重
            if node_name == "tools" and isinstance(node_data, dict):
                messages = node_data.get("messages", [])
                for msg in messages:
                    if getattr(msg, "type", "") == "tool":
                        tool_call_id: str = getattr(msg, "tool_call_id", "") or ""
                        if tool_call_id and tool_call_id in self.emitted_tool_results:
                            continue
                        if tool_call_id:
                            self.emitted_tool_results.add(tool_call_id)
                        evt_data: dict = {
                            **self._base_data(ns, state),
                            "tool_use_id": tool_call_id,
                            "tool_name": getattr(msg, "name", ""),
                            "result": str(getattr(msg, "content", "")),
                            "is_error": getattr(msg, "status", "") == "error",
                        }
                        events.append(AgentEvent(type="tool_result", turn_id=self.turn_id, data=evt_data))

            # 从 model 节点记录 agent_name 并补全工具参数
            if node_name == "model" and isinstance(node_data, dict):
                messages = node_data.get("messages", [])
                for msg in messages:
                    name_from_msg: str = getattr(msg, "name", "") or ""
                    if name_from_msg and not state.agent_name:
                        state.agent_name = name_from_msg
                    tool_calls = getattr(msg, "tool_calls", [])
                    for tc in tool_calls:
                        for idx, tool_state in state.active_tools.items():
                            if tool_state.id == tc.get("id") and not tool_state.complete:
                                tool_state.complete = True

            # 检查 todos
            if isinstance(node_data, dict) and "todos" in node_data:
                todos_raw = node_data["todos"]
                if isinstance(todos_raw, list):
                    evt_data = {**self._base_data(ns, state), "todos": todos_raw}
                    events.append(AgentEvent(type="todos_update", turn_id=self.turn_id, data=evt_data))

        return events

    # ─── flush / emit 辅助 ────────────────────────────────────────────────

    def _flush_agent(self, ns: tuple, state: AgentStreamState) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        if state.in_thinking:
            events.extend(self._emit_thinking_complete(ns, state))
        if state.text_buf:
            events.extend(self._emit_text_complete(ns, state))
        return events

    def _emit_thinking_complete(self, ns: tuple, state: AgentStreamState) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        if state.thinking_buf:
            evt_data: dict = {**self._base_data(ns, state), "content": state.thinking_buf}
            events.append(AgentEvent(type="thinking_complete", turn_id=self.turn_id, data=evt_data))
        state.in_thinking = False
        state.thinking_done = True
        state.thinking_buf = ""
        return events

    def _emit_text_complete(self, ns: tuple, state: AgentStreamState) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        if state.text_buf:
            evt_data: dict = {**self._base_data(ns, state), "content": state.text_buf}
            events.append(AgentEvent(type="text_complete", turn_id=self.turn_id, data=evt_data))
        state.text_buf = ""
        return events