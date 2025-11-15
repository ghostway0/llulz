from typing import Callable
from llm import LLM, Conversation, Context, SamplingParams, Response

Plugin = Callable[[Conversation, Context], Context]

class Environment:
    def __init__(self, llm: LLM, plugins: list[Plugin] = None):
        self.llm = llm
        self.plugins = plugins or []

    def step(self, conv: Conversation, params: SamplingParams) -> Response:
        ctx = conv.to_context()
        for plugin in self.plugins:
            ctx = plugin(conv, ctx)
        response = self.llm.generate(ctx, params)
        conv.add_assistant(response.text)
        return response
