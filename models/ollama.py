import requests, json
from llm import SamplingParams, LLM, Response, Context

class OllamaLLM(LLM):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, context: Context, params: SamplingParams) -> Response:
        payload = {
            "model": self.model,
            "messages": context.messages,
            "stream": False,
            "options": {"num_ctx": params.max_context, "temperature": params.temperature}
        }
        resp = requests.post(self.host + "/api/chat", json=payload)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return Response(text=content, token_ids=None, tool_calls=[])

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            payload = {
                "model": self.model,
                "input": text
            }
            resp = requests.post(self.host + "/api/embed", json=payload)
            resp.raise_for_status()

            data = resp.json()
            embeddings.append(data["embeddings"][0])

        return embeddings

def ollama(model: str, host: str = "http://localhost:11434") -> LLM:
    return OllamaLLM(model)

