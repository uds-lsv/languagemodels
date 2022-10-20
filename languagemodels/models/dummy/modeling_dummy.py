import torch
from torch import nn

from languagemodels.modeling_utils import LanguageModel


class DummyLM(LanguageModel):
    def __init__(self, config=None):
        super().__init__(config)

    @classmethod
    def load_model(cls, config, pre_trained=False, model_name_or_path=None):
        return DummyLM(config)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = torch.rand(size=input_ids.size())
        # probs = torch.nn.functional.softmax(logits, dim=1)
        return logits

    def compute_surprisal(self, input_ids, **kwargs):
        self.eval()  # put model in evaluation mode (disable dropout)
        probs = self.forward(input_ids)
        surprisal = - torch.log2(probs)
        return surprisal
