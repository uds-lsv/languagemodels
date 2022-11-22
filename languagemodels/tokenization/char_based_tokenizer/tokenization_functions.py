from abc import ABC, abstractmethod
import re
from typing import Dict, List
from lingpy import ipa2tokens


class CharTokenizationFunction(ABC):

    name = ""

    @abstractmethod
    def __call__(self, text) -> List[str]:
        return text.split()

    @abstractmethod
    def get_config(self) -> Dict[str, str]:
        return {}

    @classmethod
    def from_config(cls, config: Dict[str, str]) -> "CharTokenizationFunction":
        raise NotImplementedError


class RegexTokenizationFunction(CharTokenizationFunction):

    name = "regex"

    def __init__(self, pattern: str):
        self.pattern = re.compile(pattern)

    def __call__(self, text) -> List[str]:
        return self.pattern.findall(text)

    def get_config(self) -> Dict[str, str]:
        return {"pattern": self.pattern.pattern}

    def from_config(cls, config: Dict[str, str]):
        pattern = re.compile(config["pattern"])
        return cls(pattern)


class IpaTokenizationFunction(CharTokenizationFunction):

    name = "ipa"

    def __call__(self, text) -> List[str]:
        print(text)
        return ipa2tokens(text)

    def get_config(self) -> Dict[str, str]:
        return {}

    def from_config(cls, config: Dict[str, str]):
        return cls()