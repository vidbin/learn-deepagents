import asyncio
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend, CompositeBackend
from deepagents.backends.protocol import ExecuteResponse
from deepagents.middleware._utils import append_to_system_message
from dotenv import load_dotenv
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from HotSkillMiddleware import HotSkillMiddleware


DEFAULT_CONVERSATION_ID = "dynamic-mcp-demo"


UTF8_ENV = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONLEGACYWINDOWSSTDIO": "0",
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
}


def force_utf8_environment() -> None:
    """Force UTF-8 defaults before any subprocess or MCP stdio server starts."""
    for key, value in UTF8_ENV.items():
        os.environ.setdefault(key, value)


def normalize_mcp_connection(connection: dict[str, Any]) -> dict[str, Any]:
    """Make stdio MCP connections tolerant of Windows console encodings."""
    normalized = dict(connection)
    if normalized.get("transport") != "stdio":
        return normalized

    env = dict(os.environ)
    env.update(UTF8_ENV)
    env.update(normalized.get("env") or {})
    normalized["env"] = env
    normalized.setdefault("encoding", "utf-8")
    normalized.setdefault("encoding_error_handler", "replace")
    return normalized


@dataclass
class AgentContext:
    conversation_id: str


class DynamicMCPRegistry:
    """Conversation-scoped MCP registry.

    The agent is created once. This registry is updated when the UI adds or
    removes MCP servers. The middleware reads a snapshot from here on each run.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._connections: dict[str, dict[str, dict[str, Any]]] = {}
        self._clients: dict[str, MultiServerMCPClient] = {}
        self._tools: dict[str, dict[str, BaseTool]] = {}

    async def add_server(
        self,
        conversation_id: str,
        server_name: str,
        connection: dict[str, Any],
    ) -> list[BaseTool]:
        async with self._lock:
            proposed = dict(self._connections.get(conversation_id, {}))
            proposed[server_name] = normalize_mcp_connection(connection)

        client = MultiServerMCPClient(proposed, tool_name_prefix=True)
        tools = await client.get_tools()

        async with self._lock:
            self._connections[conversation_id] = proposed
            self._clients[conversation_id] = client
            self._tools[conversation_id] = {tool.name: tool for tool in tools}
            return list(self._tools[conversation_id].values())

    async def remove_server(self, conversation_id: str, server_name: str) -> list[BaseTool]:
        async with self._lock:
            current = self._connections.get(conversation_id, {})
            if server_name not in current:
                raise KeyError(f"MCP server not found: {server_name}")
            proposed = {name: value for name, value in current.items() if name != server_name}

        client = MultiServerMCPClient(proposed, tool_name_prefix=True)
        tools = await client.get_tools()

        async with self._lock:
            self._connections[conversation_id] = proposed
            self._clients[conversation_id] = client
            self._tools[conversation_id] = {tool.name: tool for tool in tools}
            return list(self._tools[conversation_id].values())

    async def list_servers(self, conversation_id: str) -> list[str]:
        async with self._lock:
            return sorted(self._connections.get(conversation_id, {}).keys())

    async def get_tools(self, conversation_id: str) -> list[BaseTool]:
        async with self._lock:
            return list(self._tools.get(conversation_id, {}).values())

    async def get_tool(self, conversation_id: str, tool_name: str) -> BaseTool | None:
        async with self._lock:
            return self._tools.get(conversation_id, {}).get(tool_name)


class DynamicMCPMiddleware(AgentMiddleware):
    """Expose current conversation MCP tools without rebuilding the agent."""

    def __init__(self, registry: DynamicMCPRegistry) -> None:
        super().__init__()
        self.registry = registry

    @staticmethod
    def _conversation_id(request: ModelRequest | ToolCallRequest) -> str:
        context = getattr(request.runtime, "context", None)
        if isinstance(context, dict):
            return str(context.get("conversation_id") or DEFAULT_CONVERSATION_ID)
        return str(getattr(context, "conversation_id", DEFAULT_CONVERSATION_ID))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        conversation_id = self._conversation_id(request)
        mcp_tools = await self.registry.get_tools(conversation_id)
        if mcp_tools:
            request = request.override(tools=[*request.tools, *mcp_tools])
        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command[Any]:
        conversation_id = self._conversation_id(request)
        tool_name = request.tool_call["name"]
        dynamic_tool = await self.registry.get_tool(conversation_id, tool_name)
        if dynamic_tool is not None:
            return await handler(request.override(tool=dynamic_tool))
        return await handler(request)


class CapabilityFreshnessMiddleware(AgentMiddleware):
    """Tell the model that current capabilities override checkpoint history."""

    def __init__(self, registry: DynamicMCPRegistry) -> None:
        super().__init__()
        self.registry = registry

    @staticmethod
    def _conversation_id(request: ModelRequest) -> str:
        context = getattr(request.runtime, "context", None)
        if isinstance(context, dict):
            return str(context.get("conversation_id") or DEFAULT_CONVERSATION_ID)
        return str(getattr(context, "conversation_id", DEFAULT_CONVERSATION_ID))

    @staticmethod
    def _skill_names(request: ModelRequest) -> list[str]:
        skills = request.state.get("skills_metadata", [])
        names = []
        for skill in skills:
            name = skill.get("name") if isinstance(skill, dict) else None
            if isinstance(name, str):
                names.append(name)
        return sorted(names)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ) -> ModelResponse:
        conversation_id = self._conversation_id(request)
        mcp_tools = await self.registry.get_tools(conversation_id)
        mcp_tool_names = sorted(tool.name for tool in mcp_tools)
        skill_names = self._skill_names(request)

        current_mcp = ", ".join(mcp_tool_names) if mcp_tool_names else "(none)"
        current_skills = ", ".join(skill_names) if skill_names else "(none)"
        freshness_prompt = f"""
## Current Capability Snapshot

The available MCP tools and skills below are authoritative for this model call.

- Current MCP tools: {current_mcp}
- Current skills: {current_skills}

If prior conversation history, checkpointed state, tool results, or old instructions mention an MCP tool or skill that is not listed above, treat it as deleted or unavailable. Do not call, suggest, or rely on unavailable MCP tools or skills.
""".strip()

        request = request.override(
            system_message=append_to_system_message(
                request.system_message,
                freshness_prompt,
            )
        )
        return await handler(request)


def decode_process_output(data: bytes | None) -> str:
    if not data:
        return ""
    for encoding in ("utf-8", "gbk", "cp936"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


class Utf8SafeLocalShellBackend(LocalShellBackend):
    """LocalShellBackend variant that never crashes on Windows output decoding."""

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            result = subprocess.run(
                command,
                check=False,
                shell=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=False,
                timeout=effective_timeout,
                env=self._env,
                cwd=str(self.cwd),
            )

            stdout = decode_process_output(result.stdout)
            stderr = decode_process_output(result.stderr)

            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                stderr_lines = stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"

            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Error: Command timed out after {effective_timeout} seconds.",
                exit_code=124,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )


def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city} 晴天~"


def create_demo_math_server() -> dict[str, Any]:
    """Create a local stdio MCP server file for quick manual testing."""
    server_path = Path(__file__).resolve().parent / "tmp" / "dynamic_math_mcp_server.py"
    server_path.parent.mkdir(parents=True, exist_ok=True)
    server_path.write_text(
        """
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Dynamic Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    \"\"\"Multiply two integers.\"\"\"
    return a * b


if __name__ == "__main__":
    mcp.run(transport="stdio")
""".lstrip(),
        encoding="utf-8",
    )
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-X", "utf8", str(server_path)],
    }


def build_model():
    force_utf8_environment()
    model = os.getenv("DYNAMIC_AGENT_MODEL", "openai:MiniMax-M2.7")
    base_url = os.getenv("DYNAMIC_AGENT_BASE_URL", "https://api.minimaxi.com/v1")
    api_key = os.getenv("DYNAMIC_AGENT_API_KEY", "sk-cp-OQ2uK_HIh_lGeW9FZTNkOVOJmbGfgAMxJz4UktFM-n781LT3Qnj2U47y9iBGkui7Tm3Z3FlUDvX12UguIKXAPAbM6S8YqXHRsCywvidLLnBw5tvEqPHZy58")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set DYNAMIC_AGENT_API_KEY, MINIMAX_API_KEY, or OPENAI_API_KEY."
        )
    return init_chat_model(model=model, base_url=base_url, api_key=api_key)


def build_agent(registry: DynamicMCPRegistry):
    force_utf8_environment()
    project_root = Path(__file__).resolve().parent
    shell_env = dict(os.environ)
    shell_env.update(UTF8_ENV)
    backend = Utf8SafeLocalShellBackend(
        root_dir=str(project_root),
        virtual_mode=True,
        inherit_env=True,
        env=shell_env,
    )
    checkpointer = MemorySaver()
    model = build_model()

    return create_deep_agent(
        model=model,
        system_prompt="你是一个专业的助手。优先使用当前会话可用的工具回答问题。永远不要使用 sub agent。",
        tools=[get_weather],
        checkpointer=checkpointer,
        backend=CompositeBackend,
        middleware=[
            HotSkillMiddleware(backend=backend, sources=["/skills/"]),
            DynamicMCPMiddleware(registry),
            CapabilityFreshnessMiddleware(registry),
        ],
        context_schema=AgentContext,
        name="dynamic-mcp-agent",
    )


def print_help() -> None:
    print(
        """
Commands:
  /mcp add-demo
      Add a local stdio MCP server with add/multiply tools.

  /mcp add-http <name> <url>
      Add an HTTP MCP server.
      Example: /mcp add-http docs https://docs.langchain.com/mcp

  /mcp add-stdio <name> <command> [args...]
      Add a stdio MCP server.
      Example: /mcp add-stdio math python C:/path/to/server.py

  /mcp add-json <name> <json>
      Add an MCP server using raw connection JSON.
      Example: /mcp add-json weather {"transport":"http","url":"http://localhost:8000/mcp"}

  /mcp remove <name>
  /mcp list
  /mcp tools
  /help
  /quit
""".strip()
    )


async def handle_command(
    command_text: str,
    registry: DynamicMCPRegistry,
    conversation_id: str,
) -> bool:
    parts = shlex.split(command_text, posix=False)
    if not parts:
        return True

    if parts[0] in {"/quit", "/exit"}:
        return False

    if parts[0] == "/help":
        print_help()
        return True

    if parts[0] != "/mcp":
        print(f"Unknown command: {parts[0]}")
        return True

    if len(parts) == 1 or parts[1] == "list":
        servers = await registry.list_servers(conversation_id)
        print("MCP servers:", ", ".join(servers) if servers else "(none)")
        return True

    action = parts[1]

    if action == "tools":
        tools = await registry.get_tools(conversation_id)
        print("MCP tools:", ", ".join(tool.name for tool in tools) if tools else "(none)")
        return True

    if action == "add-demo":
        tools = await registry.add_server(conversation_id, "demo_math", create_demo_math_server())
        print("Added demo_math. Tools:", ", ".join(tool.name for tool in tools))
        return True

    if action == "add-http":
        if len(parts) != 4:
            print("Usage: /mcp add-http <name> <url>")
            return True
        server_name, url = parts[2], parts[3]
        tools = await registry.add_server(
            conversation_id,
            server_name,
            {"transport": "http", "url": url},
        )
        print(f"Added {server_name}. Tools:", ", ".join(tool.name for tool in tools))
        return True

    if action == "add-stdio":
        if len(parts) < 4:
            print("Usage: /mcp add-stdio <name> <command> [args...]")
            return True
        server_name, command, args = parts[2], parts[3], parts[4:]
        tools = await registry.add_server(
            conversation_id,
            server_name,
            {"transport": "stdio", "command": command, "args": args},
        )
        print(f"Added {server_name}. Tools:", ", ".join(tool.name for tool in tools))
        return True

    if action == "add-json":
        if len(parts) < 4:
            print("Usage: /mcp add-json <name> <json>")
            return True
        server_name = parts[2]
        raw_json = command_text.split(server_name, 1)[1].strip()
        connection = json.loads(raw_json)
        tools = await registry.add_server(conversation_id, server_name, connection)
        print(f"Added {server_name}. Tools:", ", ".join(tool.name for tool in tools))
        return True

    if action == "remove":
        if len(parts) != 3:
            print("Usage: /mcp remove <name>")
            return True
        tools = await registry.remove_server(conversation_id, parts[2])
        print(f"Removed {parts[2]}. Remaining tools:", ", ".join(tool.name for tool in tools) if tools else "(none)")
        return True

    print(f"Unknown /mcp action: {action}")
    return True


def print_agent_result(result: dict[str, Any]) -> None:
    messages = result.get("messages", [])
    print("print agent results:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for message in messages[-4:]:
        if isinstance(message, ToolMessage):
            print(f"[tool:{message.name}] {message.content}")
        elif isinstance(message, AIMessage) and message.content:
            print(f"assistant: {message.content}")


async def chat_loop() -> None:
    load_dotenv()
    force_utf8_environment()
    conversation_id = os.getenv("DYNAMIC_AGENT_THREAD_ID", DEFAULT_CONVERSATION_ID)
    registry = DynamicMCPRegistry()
    agent = build_agent(registry)

    print("Dynamic MCP agent started.")
    print_help()

    while True:
        user_input = input("\nuser: ").strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            keep_running = await handle_command(user_input, registry, conversation_id)
            if not keep_running:
                break
            continue

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": conversation_id}},
            context={"conversation_id": conversation_id},
        )
        print_agent_result(result)


if __name__ == "__main__":
    force_utf8_environment()
    asyncio.run(chat_loop())
