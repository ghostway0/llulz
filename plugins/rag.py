from llm import Conversation, Context
import numpy as np

class RAGPlugin:
    def __init__(self, vector_index, embedder, k: int = 5):
        self.index = vector_index
        self.embedder = embedder
        self.k = k

    def __call__(self, conv: Conversation, ctx: Context) -> Context:
        query = ctx.messages[-1]["content"]
        vec = np.array(self.embedder.encode([query]), dtype=np.float32)
        D, I = self.index.search(vec, self.k)
        retrieved_chunks = [self.index.data[i] for i in I[0] if i >= 0]

        if retrieved_chunks:
            ctx.add_message("tool", "\n".join(retrieved_chunks))

        return ctx

rag = RAGPlugin(config.models["ollama"], config.models["ollama"])
