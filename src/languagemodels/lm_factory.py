from torch import nn

from languagemodels.models import (
    DummyLMConfig,
    DummyLM,
    BigramLMConfig,
    BigramLM,
    RnnConfig,
    RnnLM,
)

LMs = {
    "dummy-lm": (DummyLMConfig, DummyLM),
    "bigram-lm": (BigramLMConfig, BigramLM),
    "rnn-lm": (RnnConfig, RnnLM),
}


class LMFactory():
    @classmethod
    def get_lm(cls, model_type, config_name_or_path, pre_trained=False, model_name_or_path=None):

        assert model_type in LMs

        # load the config file
        # for now we only support local config files
        assert ".json" in config_name_or_path
        config = LMs.get(model_type)[0].from_json_file(config_name_or_path)
        
        if pre_trained:
            assert model_name_or_path is not None

            # load the model
            lm = LMs.get(model_type)[1].from_pretrained(model_name_or_path, config=config)
        
        else:
            lm = LMs.get(model_type)[1].from_config(config)
        
        return lm, config
