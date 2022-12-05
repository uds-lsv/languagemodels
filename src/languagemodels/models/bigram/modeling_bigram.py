import os

import torch
from torch import nn
from torch.nn import functional as F

from transformers.modeling_utils import PreTrainedModel

class BigramLM(PreTrainedModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        # the parameters of this model are a simple lookup table
        self.logits = nn.Parameter(
            torch.zeros((self.vocab_size, self.vocab_size))
        )

    # @classmethod
    # def load_model(cls, config, pre_trained=False, model_name_or_path=None):
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
        # state_dict = self.state_dict()
        # path_name = os.path.join(path, "bigram-lm")
        # print("Saving model to:", path_name)
        # torch.save(state_dict, path_name)

    def forward(self, input_ids, labels=None, **kwargs):
        # input_ids is a batch of bigrams, hence input_ids.shape == (batch_size, 2)
        assert input_ids.size(-1) == 2
        assert labels.size(-1) == 2

        # we predict the second token based on the first, hence we have to consider only the first token per batch
        logits = self.logits[input_ids[..., 0]]

        loss = None
        if labels is not None:  # compute loss
            # labels are a copy of input_ids. we have to shift labels to the right as we want to predict the second token
            labels = labels[..., 1:].contiguous()
            # softmax happens internally
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        return logits, loss
