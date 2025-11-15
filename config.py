from typing import Callable
from llm import LLM
import types, os, importlib

class ModelDirectory:
    def __init__(self):
        self.loaders: dict[str, Callable] = {}
        self.loaded: dict[str, LLM] = {}

    def register(self, model: str, loader: Callable):
        if model in self.loaders:
            # TODO: log warning
            pass

        self.loaders[model] = loader

    def __getitem__(self, name: str) -> LLM:
        if name in self.models:
            return self.loaded[name]

        if name not in self.loaders:
            raise ValueError(f"`{name}` is not a known model name.")

        self.models[name] = self.loaders[name](self)
        return self.models[name]

class Config:
    def __init__(self, path: str):
        self.directory = ModelDirectory()
        self.path = path
        self.plugin_dir = os.path.join(self.path, "plugins")
        self.plugins: dict[str, types.ModuleType | str] = {}

    def register_plugin(self, name: str, filename: str):
        self.plugins[name] = filename

    def __getitem__(self, name: str) -> types.ModuleType:
        if name not in self.plugins:
             raise ValueError(f"Plugin {name} not registered")

        if isinstance(self.plugins[name], str):
            spec = importlib.util.spec_from_file_location(name, self.plugins[name])
            if spec is None:
                raise ValueError(f"Could not find plugin {name} at {self.plugins[name]}")

            module = importlib.util.module_from_spec(spec)
            module.config = self

            spec.loader.exec_module(module)
            self.plugins[name] = module

        if hasattr(self.plugins[name], name):
            return getattr(self.plugins[name], name)
        else:
            return self.plugins[name]

