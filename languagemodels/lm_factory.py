from torch import nn

from languagemodels.models.bigram.modeling_bigram import BigramLM

LMs = {
    # "dummy-lm": DummyLM,
    "bigram-lm": BigramLM,
}


class LMFactory():
    @classmethod
    def get_lm(cls, model_type, config, pre_trained=False, model_name_or_path=None):
        assert model_type in LMs
        if pre_trained:
            assert model_name_or_path is not None
        lm = LMs.get(model_type).load_model(config, pre_trained)
        return lm
