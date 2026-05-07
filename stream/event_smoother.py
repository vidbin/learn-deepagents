"""AgentEvent 平滑输出器

将大块 content 的事件拆分为均匀小块，使前端逐步渲染更流畅。
同步缓冲模式：调用方控制 push 节奏，无异步定时器。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stream_converter import AgentEvent


_SMOOTHABLE_FIELDS: dict[str, str] = {
    "text_delta": "content",
    "thinking_delta": "content",
    "tool_progress": "args_delta",
}


@dataclass
class _BufferEntry:
    buf: str
    template_event: AgentEvent
    content_field: str


@dataclass
class EventSmoother:
    chunk_size: int = 20
    _buffers: dict[tuple[str, str], _BufferEntry] = field(default_factory=dict)

    def push(self, events: list[AgentEvent]) -> list[AgentEvent]:
        """接收一批事件，返回平滑后的事件列表。"""
        result: list[AgentEvent] = []
        for event in events:
            content_field: str | None = _SMOOTHABLE_FIELDS.get(event.type)
            if content_field is None:
                result.append(event)
                continue

            text: str = event.data.get(content_field, "")
            if not isinstance(text, str) or not text:
                result.append(event)
                continue

            agent_id: str = event.data.get("agent_id", "")
            key: tuple[str, str] = (event.type, agent_id)

            entry: _BufferEntry | None = self._buffers.get(key)
            if entry is not None:
                text = entry.buf + text
            else:
                entry = _BufferEntry(buf="", template_event=event, content_field=content_field)

            entry.template_event = event

            chunks: list[str] = []
            i: int = 0
            while i + self.chunk_size <= len(text):
                chunks.append(text[i:i + self.chunk_size])
                i += self.chunk_size

            remainder: str = text[i:]

            for chunk in chunks:
                data: dict = {k: v for k, v in event.data.items() if k != content_field}
                data[content_field] = chunk
                result.append(AgentEvent(type=event.type, turn_id=event.turn_id, data=data))

            if remainder:
                entry.buf = remainder
                self._buffers[key] = entry
            elif key in self._buffers:
                del self._buffers[key]

        return result

    def flush(self) -> list[AgentEvent]:
        """刷出内部缓冲区剩余内容。"""
        result: list[AgentEvent] = []
        for key, entry in self._buffers.items():
            if entry.buf:
                event: AgentEvent = entry.template_event
                data: dict = {k: v for k, v in event.data.items() if k != entry.content_field}
                data[entry.content_field] = entry.buf
                result.append(AgentEvent(type=event.type, turn_id=event.turn_id, data=data))
        self._buffers.clear()
        return result
