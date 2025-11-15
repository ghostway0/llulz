from typing import Callable, Any
from llm import LLM
from pathlib import Path
import os, importlib.util

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

def register_all_for(plugin_dir: PluginDirectory, path: Path):
    path = path.resolve()  # absolute path
    for file in path.iterdir():
        filepath = file.resolve()
        if not str(filepath).startswith(str(path)):
            raise ValueError("Plugin outside allowed path")

        if file.is_file() and file.suffix == ".py":
            plugin_dir.register(file.stem, filepath)

class Config:
    def __init__(self, path: str):
        self.path = Path(path)
        self.plugins_dir = self.path / "plugins"
        self.models_dir = self.path / "models"
        self.run_dir = Path.cwd()
        
        self.plugins = PluginDirectory("plugins", self)
        self.models = PluginDirectory("models", self)

        register_all_for(self.plugins, self.plugins_dir)
        register_all_for(self.models, self.models_dir)

    def __getitem__(self, name: str) -> dict[str, Any] | None:
        import configparser

        config_path = self.path / f"{name}.ini"
        parser = configparser.ConfigParser()
        if not os.path.exists(config_path):
            return None
        
        parser.read(config_path)
        return {section: dict(parser[section]) for section in parser.sections()}

    def __setitem__(self, name: str, value: dict[str, Any]):
        import configparser

        parser = configparser.ConfigParser()
        for section, options in value.items():
            parser[section] = {str(k): str(v) for k, v in options.items()}

        config_path = self.path / f"{name}.ini"
        with config_path.open("w", encoding="utf-8") as f:
            parser.write(f)
