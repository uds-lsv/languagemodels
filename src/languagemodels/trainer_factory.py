from transformers import Trainer

from languagemodels.trainer import RnnLMTrainer


def get_trainer_cls(model_type: str):
    if model_type == "rnn-lm":
        trainer_cls = RnnLMTrainer
    # elif model_type == "perceiver-ar-causal-language-model": 
    #     trainer_cls = PerceiverCausalLanguageModelTrainer
    else:
        trainer_cls = Trainer
    return trainer_cls


class TrainerFactory:

    @classmethod
    def get_trainer(cls, model_type: str, **trainer_args):
        trainer_cls = get_trainer_cls(model_type)
        trainer = trainer_cls(**trainer_args)
        return trainer