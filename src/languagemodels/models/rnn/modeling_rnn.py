import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_utils import PreTrainedModel

from .configuration_rnn import RnnLMConfig


class RnnLM(PreTrainedModel):

    config_class = RnnLMConfig

    def __init__(self, config=None):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.max_length = config.block_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.add_bias = config.add_bias
        self.dropout = config.dropout
        self.cell_type = config.cell_type
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        # the initial hidden state of time step 0
        # shape is: (num_layers, batch_size, hidden_dim). We use 1 as a placeholder for the bsz and expand it during the forward pass
        # the LSTM cell needs an additional context state

        self.initial_hidden_state = self._init_hidden()

        # the emedding matrix
        self.wte = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        # the encoder
        if self.cell_type == "rnn":
            self.encoder = nn.RNN(
                self.embedding_dim,
                self.hidden_dim,
                self.num_layers,
                "tanh",
                self.add_bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=False
            )
        elif self.cell_type == "gru":
            self.encoder = nn.GRU(
                self.embedding_dim,
                self.hidden_dim,
                self.num_layers,
                self.add_bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=False
            )
        elif self.cell_type == "lstm":
            self.encoder = nn.LSTM(
                self.embedding_dim,
                self.hidden_dim,
                self.num_layers,
                self.add_bias,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=False,
            )
        else:
            raise NotImplementedError(
                f"Unsupported cell_type {self.cell_type}")

        # the output layer
        # TODO(mm): support tying weights
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)

    def _init_hidden(self):
        if self.cell_type == "lstm":
            self.h_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))
            self.c_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))
            return (self.h_0, self.c_0)
        else:
            return nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim))

    def _expand_hidden(self, batch_size: int):
        if self.cell_type == "lstm":
            return (
                self.h_0.expand((-1, batch_size, -1)).contiguous(),
                self.c_0.expand((-1, batch_size, -1)).contiguous() 
            )
        else:
            return self.initial_hidden_state.expand((-1, batch_size, -1)).contiguous()

    # @classmethod
    # def load_model(cls, config, pre_trained=False, model_name_or_path=None):
    #     # TODO(mm): move this functionality to the parent class

    #     if pre_trained:
    #         # TODO(mm): load a pre-trained model from disc
    #         # TODO(mm): if config is None, search for a config in model_name_or_path
    #         model = cls(config)  # create a model based on the config
    #         model.state_dict(
    #             torch.load(model_name_or_path)
    #         )  # overwrite the state_dict
    #     else:
    #         model = cls(config)
    #     return model

    # def save_model(self, path):
    #     state_dict = self.state_dict()
    #     path_name = os.path.join(path, f"{self.cell_type}-lm")
    #     print("Saving model to:", path_name)
    #     torch.save(state_dict, path_name)

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls._from_config(config, **kwargs)

    def forward(self, input_ids, labels=None, hidden_state=None, pad_id=-100, reduction="mean", return_dict=True, **kwargs):
        # get the input embeddings
        embeddings = self.wte(input_ids)
        embeddings = self.embedding_dropout(embeddings)

        if hidden_state is None:
            # expand the initial hidden state to have the correct batch_size
            hidden_state = self._expand_hidden(input_ids.shape[0])

        # outputs.shape = (batch_size, block_size, hidden_dim) contains encoder outputs for every timestep t
        # final_hidden_state.shape = (num_layers, batch_size, hidden_dim) contains the final hidden state per layer for each sequence in the batch
        outputs, final_hidden_state = self.encoder(embeddings, hidden_state)
        
        # decode predictions by applying lm_head to the hidden state of every token
        logits = self.lm_head(outputs)

        # compute loss
        loss = None
        if labels is not None:
            # shift logits and labels so that tokens < n predict n
            # all tokens but the last
            shift_logits = logits[..., :-1, :].contiguous()
            # all tokens but the first
            shift_labels = labels[..., 1:].contiguous()

            # compute loss
            loss_fct = CrossEntropyLoss(ignore_index=pad_id, reduction=reduction)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if return_dict:
            return dict(logits=logits, loss=loss, final_hidden_state=final_hidden_state)
        return logits, loss, final_hidden_state
