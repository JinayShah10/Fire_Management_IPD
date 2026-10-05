import logging
import os

import yaml

logger = logging.getLogger(__name__)

class Config:
    _instance = None

    def __init__(self, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    @classmethod
    def load(cls, path="configs/default.yaml"):
        if cls._instance is None:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Config file not found at {path}")
            
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                
            cls._instance = cls(data)
        return cls._instance

def get_config():
    return Config.load()
