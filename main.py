from dataclasses import dataclass
from typing import Any, Callable
import importlib, os, types

from config import Config
from ollama import OllamaLLM
from llm import SamplingParams, Conversation
from env import Environment

if __name__ == "__main__":
    config = Config(".")
    for filename in os.listdir(config.plugin_dir):
        filepath = os.path.abspath(os.path.join(config.plugin_dir, filename))
        if not filepath.startswith(os.path.abspath(config.plugin_dir)):
            raise ValueError("Plugin outside allowed path")

        if os.path.isfile(filepath):
            print(filepath)
            config.register_plugin(filename[:filename.rfind(".")], filepath)
    config["hello"]

    conv = Conversation()
    conv.add_user("Explain quantum tunneling simply.")
    llm = OllamaLLM("magistral:latest")
    params = SamplingParams(temperature=0.0, max_tokens=512)
    env = Environment(llm=llm, plugins=[])
    response = env.step(conv, params)
    print("Assistant:", response.text)

