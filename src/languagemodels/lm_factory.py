import os

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)


def get_automodel(model_type: str):
    return {
        "causal": AutoModelForCausalLM,
        "masked": AutoModelForMaskedLM,
        "seq2seq": AutoModelForSeq2SeqLM
    }[model_type]


class LMFactory():

    @classmethod
    def from_pretrained(cls, model_type, model_name_or_path=None):
        auto_cls = get_automodel(model_type)
        # if not os.path.isdir(model_name_or_path):
        #     raise FileNotFoundError(f"Model dir {model_name_or_path} does not exist!")
        config_loc = model_name_or_path
        config_file_path = os.path.join(model_name_or_path, "config.json")
        if os.path.isfile(config_file_path):
            config_loc = config_file_path
            # raise FileNotFoundError(f"No config file found at {model_name_or_path}")
        config = AutoConfig.from_pretrained(config_loc)
        model = auto_cls.from_pretrained(model_name_or_path)
        return model, config
    
    @classmethod
    def from_config(cls, model_type, config_name_or_path=None):
        auto_cls = get_automodel(model_type)
        config = AutoConfig.from_pretrained(config_name_or_path)
        model = auto_cls.from_config(config)
        return model, config
