from torch import nn

from languagemodels.models.dummy import DummyLM

LMs = {
    "dummy-lm": DummyLM,
}


class LMFactory():
    @classmethod
    def get_lm(cls, name_or_path, config, pre_trained=False):
        assert name_or_path in LMs
        lm = LMs.get(name_or_path).load_model(config, pre_trained)
        return lm
