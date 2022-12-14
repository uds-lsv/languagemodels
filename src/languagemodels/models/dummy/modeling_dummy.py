import torch
from torch import nn

from transformers.modeling_utils import PreTrainedModel


class DummyLM(PreTrainedModel):
    def __init__(self, config=None):
        super().__init__(config)

    @classmethod
    def load_model(cls, config, pre_trained=False, model_name_or_path=None):
        return cls(config)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = torch.rand(size=input_ids.size())
        return logits
