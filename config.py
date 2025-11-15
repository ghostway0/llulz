from typing import Callable, Any
from llm import LLM
import os, importlib

class PluginDirectory:
    def __init__(self, dir_name: str, config: "Config"):
        self.paths: dict[str, str] = {}
        self.loaded: dict[str, Any] = {}
        self.config = config
        self.dir_name = dir_name

    def register(self, name: str, filename: str):
        if name in self.paths:
            # TODO: log warning
            pass

        self.paths[name] = filename

    def __getitem__(self, name: str) -> Any:
        if name not in self.paths:
             raise ValueError(f"`{name}` is not registered in {self.dir_name}")

        if name not in self.loaded:
            spec = importlib.util.spec_from_file_location(name, self.paths[name])
            if spec is None:
                raise ValueError(f"Could not find plugin `{name}` at {self.paths[name]}")

            module = importlib.util.module_from_spec(spec)
            module.config = self.config

            spec.loader.exec_module(module)
            self.loaded[name] = module

        if hasattr(self.loaded[name], name):
            return getattr(self.loaded[name], name)
        else:
            return self.loaded[name]

def register_all_for(plugin_dir: PluginDirectory, path: str):
    for filename in os.listdir(path):
        filepath = os.path.abspath(os.path.join(path, filename))
        if not filepath.startswith(os.path.abspath(path)):
            raise ValueError("Plugin outside allowed path")

        if os.path.isfile(filepath):
            plugin_dir.register(filename[:filename.rfind(".")], filepath)

class Config:
    def __init__(self, path: str):
        self.path = path
        self.plugins = PluginDirectory("plugins", self)
        self.models = PluginDirectory("models", self)
        self.plugins_dir = os.path.join(self.path, "plugins")
        self.models_dir = os.path.join(self.path, "models")
        
        register_all_for(self.plugins, self.plugins_dir)
        register_all_for(self.models, self.models_dir)
