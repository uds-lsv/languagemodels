from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig


@dataclass
class BigramLMConfig(PretrainedConfig):
    # the bigram lm expects a bigram as input, hence the block_size = 2.
    vocab_size: int = 2
    block_size: int = 2
