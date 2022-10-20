from torch import nn


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
        pass
