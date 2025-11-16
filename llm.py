from typing import Any
from dataclasses import dataclass
import re


def chunk_text(
    text: str,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> str:
    def count_tokens(s: str):
        return len(s.split())

    def split_into_sentences(text: str) -> list[str]:
        parts = re.split(r"([.!?]\s+)", text)
        if len(parts) == 1:
            return [text]

        sentences = []
        for i in range(0, len(parts), 2):
            sent = parts[i]
            if i + 1 < len(parts):
                sent += parts[i + 1]
            sentences.append(sent.strip())
        return sentences

    sentences = split_into_sentences(text)
    window: list[str] = []
    window_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if window_tokens + sent_tokens > max_tokens:
            yield " ".join(window).strip()

            overlap = []
            overlap_count = 0
            for s in reversed(window):
                t = count_tokens(s)
                if overlap_count + t > overlap_tokens:
                    break
                overlap.append(s)
                overlap_count += t
            overlap.reverse()

            window = overlap + [sent]
            window_tokens = sum(count_tokens(s) for s in window)
        else:
            window.append(sent)
            window_tokens += sent_tokens

    if window:
        yield " ".join(window).strip()

class Context:
    def __init__(self, messages: list[dict[str, Any]] = None, metadata: dict[str, Any] = None):
        self.messages = messages or []
        self.metadata = metadata or {}

    def replace(self, role: str, content: str, tool: str = None, target: str = None):
        msg = {"role": role, "content": content}
        if tool:
            msg["tool"] = tool
        if target:
            msg["target"] = target
        self.messages.append(msg)

    def add(self, role: str, content: str, tool: str = None, target: str = None):
        msg = {"role": role, "content": content}
        if tool:
            msg["tool"] = tool
        if target:
            msg["target"] = target
        self.messages.insert(len(self.messages) - 1, msg)

class Conversation:
    def __init__(self):
        self.messages: list[dict[str, Any]] = []

    def add(self, message: dict[str, Any]):
        self.messages.append(message)

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.messages.append({"role": "assistant", "content": text})

    def add_tool(self, tool_name: str, output: str, target: str = "global"):
        self.messages.append({"role": "tool", "tool": tool_name, "content": output, "target": target})

    def to_context(self) -> Context:
        return Context(messages=list(self.messages))
    
    def __repr__(self):
        return f"<Conversation messages={len(self.messages)}>"

    def __str__(self):
        lines = []
        for msg in self.messages:
            role = msg["role"]
            if role == "tool":
                lines.append(f"[{msg['role']}:{msg['tool']}]\n{msg['content']}")
            else:
                lines.append(f"[{role}]\n{msg['content']}")
        return "\n".join(lines)

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

    def generate_stream(self, context: Context, params: SamplingParams) -> str | Response:
        raise NotImplementedError

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
