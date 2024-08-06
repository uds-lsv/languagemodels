from dataclasses import dataclass

from transformers.configuration_utils import PretrainedConfig

@dataclass
class DummyLMConfig(PretrainedConfig):
    model_type = "DummyLM"
    pass
