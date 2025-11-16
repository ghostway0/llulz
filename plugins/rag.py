from llm import Conversation, Context, chunk_text
from annoy import AnnoyIndex
import numpy as np

class RAGPlugin:
    def __init__(self, vector_index, embedder, k: int = 5):
        self.index = vector_index
        self.embedder = embedder
        self.k = k

    def __call__(self, conv: Conversation, ctx: Context) -> Context:
        query = ctx.messages[-1]["content"]
        vec = np.array(self.embedder.encode([query])[0], dtype=np.float32)
        D, I = self.index.search(vec, self.k)
        retrieved_chunks = [self.index.data[i] for i in I if i >= 0]

        if retrieved_chunks:
            ctx.add("tool", "RAG relevant content:\n" + "\n".join(retrieved_chunks))

        return ctx

class Indexer:
    def __init__(self, dim: int, metric: str = "angular"):
        self.index = AnnoyIndex(dim, metric)
        # TODO: this is for testing purposes. eventually this should be in storage
        self.data: list[str] = []
        self._built = False

    def add(self, vector, text: str):
        i = len(self.data)
        self.data.append(text)
        self.index.add_item(i, vector)

    def build(self, n_trees: int = 10):
        self.index.build(n_trees)
        self._built = True

    def search(self, vector, k: int = 5):
        if not self._built:
            self.build()

        ids, distances = self.index.get_nns_by_vector(
            vector, k, include_distances=True
        )
        return distances, ids

    def save(self, index_path: str, data_path: str):
        if not self._built:
            self.build()

        self.index.save(index_path)
        with open(data_path, "w") as f:
            json.dump(self.data, f)

    @classmethod
    def load(cls, index_path: str, data_path: str, dim: int, metric="angular"):
        idx = cls(dim, metric)
        idx.index.load(index_path)

        with open(data_path, "r") as f:
            idx.data = json.load(f)

        idx._built = True
        return idx

if config["rag"] is None:
    config["rag"] = {}

index = Indexer(len(config.models["ollama"]("magistral:latest").encode(["Hello"])[0]))
index.add(config.models["ollama"]("magistral:latest").encode(["Hello"])[0], "Hello")
rag = RAGPlugin(index, config.models["ollama"]("magistral:latest"))
