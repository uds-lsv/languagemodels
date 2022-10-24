import json

from dataclasses import dataclass


@dataclass
class LanguageModelConfig():
    """Base class for all language model configurations

    """
    vocab_size: int = None
    block_size: int = None

    @classmethod
    def from_json_file(cls, json_file):
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
