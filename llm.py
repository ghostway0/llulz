from typing import Any
from dataclasses import dataclass

class Context:
    def __init__(self, messages: list[dict[str, Any]] = None, metadata: dict[str, Any] = None):
        self.messages = messages or []
        self.metadata = metadata or {}

    def add_message(self, role: str, content: str, tool: str = None, target: str = None):
        msg = {"role": role, "content": content}
        if tool:
            msg["tool"] = tool
        if target:
            msg["target"] = target
        self.messages.append(msg)

class Conversation:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def add_tool(self, tool_name: str, output: str, target: str = "global"):
        self.messages.append({"role": "tool", "tool": tool_name, "content": output, "target": target})

    def to_context(self) -> Context:
        return Context(messages=list(self.messages))

@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 0.0
    max_context: int = 16384
    max_tokens: int = 512

@dataclass
class ToolCall:
    name: str
    input: str
    target: str = "global"

@dataclass
class Response:
    text: str
    token_ids: list[int] = None
    tool_calls: list[ToolCall] = None

class LLM:
    def generate(self, context: Context, params: SamplingParams) -> Response:
        raise NotImplementedError

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
