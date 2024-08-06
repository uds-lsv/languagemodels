"""Make key classes available in the languagemodels namespace"""
from languagemodels.lm_factory import LMFactory
from languagemodels.trainer_factory import TrainerFactory
from languagemodels.argparser_factory import ArgumentParserFactory

from languagemodels.models.dummy.configuration_dummy import DummyLMConfig
from languagemodels.models.dummy.modeling_dummy import DummyLM

from languagemodels.models.bigram.configuration_bigram import BigramLMConfig
from languagemodels.models.bigram.modeling_bigram import BigramLM

from languagemodels.models.rnn.configuration_rnn import RnnLMConfig
from languagemodels.models.rnn.modeling_rnn import RnnLM

from languagemodels.models.opt.configuration_opt import OPTWithALiBiConfig
from languagemodels.models.opt.modeling_opt import OPTWithALiBiForCausalLM, OPTWithALiBiModel, OPTWithAliBiForSequenceClassification

# register auto classes
from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
)

AutoConfig.register("bigram-lm", BigramLMConfig)
AutoModelForCausalLM.register(BigramLMConfig, BigramLM)

AutoConfig.register("OPTWithALiBi", OPTWithALiBiConfig)
AutoModel.register(OPTWithALiBiConfig, OPTWithALiBiModel)
AutoModelForCausalLM.register(OPTWithALiBiConfig, OPTWithALiBiForCausalLM)
AutoModelForSequenceClassification.register(OPTWithALiBiConfig, OPTWithAliBiForSequenceClassification)

AutoConfig.register("rnn-lm", RnnLMConfig)
AutoModelForCausalLM.register(RnnLMConfig, RnnLM)

# If perceiver-io is installed, register the relevant auto classes
# import importlib.util

# if importlib.util.find_spec("perceiver") is not None:

#     from perceiver.model.text import clm
#     from transformers import AutoModelForCausalLM, AutoConfig

#     AutoConfig.register("perceiver-ar-causal-language-model", clm.PerceiverCausalLanguageModelConfig)
#     AutoModelForCausalLM.register(clm.PerceiverCausalLanguageModelConfig, clm.PerceiverCausalLanguageModel)