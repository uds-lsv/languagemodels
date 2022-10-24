from dataclasses import dataclass

from languagemodels.configuration_utils import LanguageModelConfig


@dataclass
class RnnConfig(LanguageModelConfig):
    embedding_dim: int = None
    hidden_dim: int = None
    num_layers: int = None
    cell_type: str = None
    add_bias: bool = None
    dropout: float = None
