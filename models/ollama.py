import requests, json
from llm import SamplingParams, LLM, Response

class OllamaLLM(LLM):
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate_stream(self, context: list[dict[str, any]], params: SamplingParams) -> str | Response:
        payload = {
            "model": self.model,
            "messages": context,
            "stream": True,
            "options": {
                "num_ctx": params.max_context,
                "num_predict": params.max_tokens,
                "temperature": params.temperature,
            },
        }

        with requests.post(self.host + "/api/chat", json=payload, stream=True) as resp:
            resp.raise_for_status()
            buffer = []

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                delta = data.get("message", {}).get("content")
                if delta:
                    buffer.append(delta)
                    yield delta

            final_text = "".join(buffer)
            yield Response(text=final_text, token_ids=None, tool_calls=[])


    def generate(self, context: list[dict[str, any]], params: SamplingParams) -> Response:
        payload = {
            "model": self.model,
            "messages": context,
            "stream": False,
            "options": {
                "num_ctx": params.max_context,
                "num_predict": params.max_tokens,
                "temperature": params.temperature
            }
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

