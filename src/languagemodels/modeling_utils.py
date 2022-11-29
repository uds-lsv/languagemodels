import torch

from torch import nn
from torch.nn import functional as F


class LanguageModel(nn.Module):
    """Base class for all auto-regressive language models

    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def load_model(cls, config, pre_trained=False, model_name_or_path=None):
        pass

    def save_model(self, path):
        pass

    def forward(self, input_ids, labels=None, **kwargs):
        pass

    def compute_surprisal(self, input_ids, **kwargs):
        self.eval()  # put model in evaluation mode (disable dropout)
        logits, _ = self.forward(input_ids)
        probs = F.softmax(logits)
        surprisal = - torch.log2(probs)
        return surprisal
