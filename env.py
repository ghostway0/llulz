from typing import Callable
from llm import LLM, Conversation, Context, SamplingParams, Response

Plugin = Callable[[Conversation, Context], Context]

class RAGPlugin:
    def __init__(self, vector_index, embedder, k: int = 5):
        self.index = vector_index
        self.embedder = embedder
        self.k = k

    def __call__(self, conv: Conversation, ctx: Context) -> Context:
        query = ctx.messages[-1]["content"]
        vec = self.embedder.encode([query])
        D, I = self.index.search(vec, self.k)
        retrieved_chunks = [self.index.data[i] for i in I[0]]
        ctx.add_message("tool", "\n".join(retrieved_chunks))
        return ctx

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


