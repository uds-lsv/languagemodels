from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @classmethod
    def load_model(cls, config, pre_trained=False):
        pass

    def compute_surprisal(self, input_ids, **kwargs):
        pass
