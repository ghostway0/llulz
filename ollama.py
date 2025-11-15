import requests, json
from llm import SamplingParams, LLM, Response, Context

class OllamaLLM(LLM):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.url = f"{host}/api/chat"

    def generate(self, context: Context, params: SamplingParams) -> Response:
        payload = {
            "model": self.model,
            "messages": context.messages,
            "stream": False,
            "options": {"num_ctx": params.max_context, "temperature": params.temperature}
        }
        resp = requests.post(self.url, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return Response(text=content, token_ids=None, tool_calls=[])
