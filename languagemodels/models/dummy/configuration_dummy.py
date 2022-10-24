from dataclasses import dataclass

from languagemodels.configuration_utils import LanguageModelConfig


@dataclass
class DummyLMConfig(LanguageModelConfig):
    pass
