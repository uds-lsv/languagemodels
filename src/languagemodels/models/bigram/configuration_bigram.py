from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig


@dataclass
class BigramLMConfig(PretrainedConfig):
    # the bigram lm expects a bigram as input, hence the block_size = 2.
    
    model_type = "bigram-lm"
    
    def __init__(
        self,
        vocab_size: int=2,
        max_length: int=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        super().__init__(**kwargs)

    vocab_size: int = 2
    max_length: int = 2
