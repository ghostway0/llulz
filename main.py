from dataclasses import dataclass
from typing import Any, Callable
import importlib, os, types

from config import Config
from llm import SamplingParams, Conversation
from env import Environment

if __name__ == "__main__":
    config = Config(".")
    conv = Conversation()
    conv.add_user("Explain quantum tunneling simply.")
    llm = config.models["ollama"]("magistral:latest")
    params = SamplingParams(temperature=0.0, max_tokens=512)
    env = Environment(llm=llm, plugins=[]) # config.plugins["rag"]
    response = env.step(conv, params)
    print("Assistant:", response.text)

