import torch

from transformers.modeling_utils import PreTrainedModel

from languagemodels.models.dummy.configuration_dummy import DummyLMConfig


class DummyLM(PreTrainedModel):
    
    config_class = DummyLMConfig

    def __init__(self, config=None):
        super().__init__(config)

    @classmethod
    def load_model(cls, config, pre_trained=False, model_name_or_path=None):
        return cls(config)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = torch.rand(size=input_ids.size())
        return logits
