from dataclasses import dataclass


@dataclass
class BigramLMConfig:
    vocab_size: int = None
    # the bigram lm expects a bigram as input, hence the block_size = 2.
    block_size: int = 2
