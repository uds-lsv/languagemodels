from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig

@dataclass
class RnnConfig(PretrainedConfig):
    vocab_size: int = None
    block_size: int = None
    embedding_dim: int = None
    hidden_dim: int = None
    num_layers: int = None
    cell_type: str = None
    add_bias: bool = None
    embedding_dropout: float = None
    dropout: float = None
