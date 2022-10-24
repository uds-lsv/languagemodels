import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from languagemodels.modeling_utils import LanguageModel


class ElmanRnnCell(nn.Module):
    """A simple RNN cell.

    Args:
        nn (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """

    def __init__(self):
        super().__init__()
        # TODO(mm): Implement
        # use https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html

    def forward(self):
        # TODO(mm): Implement
        pass


class GruCell(nn.Module):
    """A GRU cell.

    Args:
        embedding_dim (int): dim of the inputs
        hidden_dim (int): dim of the hidden states
        num_layers (int): number of GRU layers to stack
        add_bias (bool): whether or not to add a bias term
        dropout (float): dropout probability
    """

    def __init__(self, embedding_dim, hidden_dim, num_layers, add_bias, dropout):
        super().__init__()
        # initialize a GRU "model"
        self.cell = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=add_bias,
            batch_first=True,
            dropout=dropout,  # dropout is added to the output of every GRU layer execept the last layer
            bidirectional=False,  # this is always false for autoregressive LMs
        )

    def forward(self, x_t, hprev):
        # output.shape = (batch_size, seq_len, hidden_dim)
        # these are the output features from the last layer of the GRU for every timestep t
        output, h_final = self.cell(x_t, hprev)
        return output


class LstmCell(nn.Module):
    """A GRU cell.

    Args:
        nn (_type_): _description_

    Raises:
        NotImplementedError: _description_
    """

    def __init__(self):
        super().__init__()
        # TODO(mm): Implement

    def forward(self):
        # TODO(mm): Implement
        pass


class RnnLM(LanguageModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.add_bias = config.add_bias
        self.dropout = config.dropout
        self.cell_type = config.cell_type

        # the initial hidden state of time step 0
        self.initial_hidden_state = nn.Parameter(
            torch.zeros(self.num_layers, 1, self.hidden_dim))

        # the emedding matrix
        self.wte = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)

        # the output layer
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)

        if self.cell_type == "rnn":
            raise NotImplementedError(
                f"Unsupported cell_type {self.cell_type}")
        elif self.cell_type == "gru":
            self.model = GruCell(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                add_bias=self.add_bias,
                dropout=self.dropout
            )
        elif self.cell_type == "lstm":
            raise NotImplementedError(
                f"Unsupported cell_type {self.cell_type}")
        else:
            raise NotImplementedError(
                f"Unsupported cell_type {self.cell_type}")

    @classmethod
    def load_model(cls, config, pre_trained=False, model_name_or_path=None):
        # TODO(mm): move this functionality to the parent class

        if pre_trained:
            # TODO(mm): load a pre-trained model from disc
            # TODO(mm): if config is None, search for a config in model_name_or_path
            model = cls(config)  # create a model based on the config
            model.state_dict(
                torch.load(model_name_or_path)
            )  # overwrite the state_dict
        else:
            model = cls(config)
        return model

    def save_model(self, path):
        pass

    def forward(self, input_ids, labels=None, **kwargs):
        # get the input embeddings
        embeddings = self.wte(input_ids)

        # run the inputs through the rnn cell (potentially > 1 layers)
        initial_hidden_state = self.initial_hidden_state.expand(
            (-1, input_ids.shape[0], -1))  # expand the batch dimension of the initial hidden state
        outputs = self.model(embeddings, initial_hidden_state.contiguous())
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
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return logits, loss
