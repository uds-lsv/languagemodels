from dataclasses import dataclass

from languagemodels.configuration_utils import LanguageModelConfig


@dataclass
class BigramLMConfig(LanguageModelConfig):
    # the bigram lm expects a bigram as input, hence the block_size = 2.
    block_size: int = 2
