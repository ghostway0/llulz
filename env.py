from typing import Callable
from llm import LLM, Conversation, SamplingParams, Response

Plugin = Callable[[Conversation], Conversation]

class Environment:
    def __init__(self, plugins: list[Plugin] = None):
        self.plugins = plugins or []

    def step(self, conv: Conversation) -> Conversation:
        for plugin in self.plugins:
            conv = plugin(conv)
        return conv

    def play(self, llm: LLM, conv: Conversation, params: SamplingParams, role: str = "assistant") -> tuple[Response, Conversation]:
        while True:
            conv = self.step(conv)
            resp = llm.generate(conv.to_context(), params)

            if not resp.text:
                return None, conv

            msg_id = conv.add_assistant(resp.text)

            if not resp.tool_calls:
                return resp, conv

            for call in resp.tool_calls:
                conv.add_meta(msg_id, {
                    "type": "tool_call",
                    "name": call.name,
                    "input": call.input,
                    "target": call.target,
                })
