from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig

@dataclass
class RnnLMConfig(PretrainedConfig):

    model_type = "rnn-lm"

    def __init__(
            self,
        vocab_size=10000,
        block_size=128,
        embedding_dim=256,
        hidden_dim=256,
        num_layers=4,
        cell_type="lstm",
        add_bias=True,
        embedding_dropout=0.1,
        dropout=0.1,
        **kwargs   
    ):
        if not cell_type in ["lstm", "gru", "rnn"]:
            raise ValueError(f"cell_type must be either 'lstm', 'gru' or 'rnn', got {cell_type}")
        
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.add_bias = add_bias
        self.embedding_dropout = embedding_dropout
        self.dropout = dropout
        super().__init__(**kwargs)

from transformers import OPTConfig

config = OPTConfig()
config.model_type