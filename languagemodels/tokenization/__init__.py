from typing import List

from .char_level_tokenizer.char_level_tokenizer import CharLevelTokenizer
from .char_level_tokenizer.char_level_tokenization_functions import (
    RegexTokenizationFunction,
    IpaTokenizationFunction,
    CharTokenizationFunction
)

_TOKENIZATION_FUNCTIONS = {
    "naive": RegexTokenizationFunction(r"."),
    "ged": RegexTokenizationFunction(r"\w'?"),
    "ipa": IpaTokenizationFunction()
}

def available_tokenization_functions() -> List[str]:
    return list(_TOKENIZATION_FUNCTIONS.keys())

def get_tokenization_function(name: str) -> CharTokenizationFunction:
    return _TOKENIZATION_FUNCTIONS[name]
