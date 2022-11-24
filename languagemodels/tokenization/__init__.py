from typing import List

from .char_based_tokenizer.tokenizer import CharacterBasedTokenizer
from .char_based_tokenizer.tokenization_functions import (
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
