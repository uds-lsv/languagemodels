from typing import Literal, Optional

from dataclasses import dataclass, field

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class BaseArgs:

    block_size: Optional[int] = field(
        default = None,
        metadata = {
            "help": (
                "Input size of the model."
            )
        }
    )

    auto_model_class: Literal["causal", "masked", "seq2seq"] = field(
        default = "causal",
        metadata = {
            "help": (
                "Which automodel class to use."
            )
        }
    )


@dataclass
class TrainingInputArgs:

    input_files_path: str = field(
        default = None,
        metadata = {
            "help": (
                "Path to train and validation & test files. Files should be named "
                "'train.txt', 'validation.txt' and 'test.txt' if not specified otherwise."
            )
        }
    )

    train_file_name: str = field(
        default = "train.txt",
        metadata = {
            "help": (
                "Filename of the train data file."
            )
        }
    )

    validation_file_name: str = field(
        default = "validation.txt",
        metadata = {
            "help": (
                "Filename of the validation data file."
            )
        }
    )

    test_file_name: str = field(
        default = "test.txt",
        metadata = {
            "help": (
                "Filename of the test data file."
            )
        }
    )

    tokenizer_name_or_path: str = field(
        default = None,
        metadata = {
            "help": (
                "Either the name of a HuggingFace tokenizer or the path to a self-trained tokenizer."
            )
        }
    )

    model_type: str = field(
        default = None,
        metadata = {
            "help": (
                "Which AutoModel class to use."
            )
        }
    )
    
    model_name_or_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": (
                "Either the name of a HuggingFace model or the path to a self-trained model."
            )
        }
    )

    config_name_or_path: Optional[str] = field(
        default = None,
        metadata = {
            "help": (
                "Path to a model config file."
            )
        }
    )

    use_early_stopping: bool = field(
        default = False,
        metadata = {
            "help": (
                "Stop training if validation loss does not improve for a certain number of steps."
            )
        }
    )

    early_stopping_patience: int = field(
        default = 1,
        metadata = {
            "help": (
                "Number of eval steps to wait before stopping training (if using early stopping)."
            )
        }
    )

    lossy_context: bool = field(
        default = False,
        metadata = {
            "help": (
                "Whether to break sequences randomly before concatenation."
            )
        }
    )

    lossy_context_token: str = field(
        default = "<b>",
        metadata = {
            "help": (
                "Sequence breaking token to be used with lossy context (will be added to the tokenizer)"
            )
        }
    )


@dataclass
class WandbArgs:

    wandb_group_name: str = field(
        default = None,
        metadata = {
            "help": (
                "Name of the wandb group."
            )
        }
    )

    wandb_project_name: str = field(
        default = None,
        metadata = {
            "help": (
                "Name of the wandb project."
            )
        }
    )


@dataclass
class EvalArgs():

    input_files_path: str = field(
        default = None,
        metadata = {
            "help": (
                "Path to the file with the evaluation data. File should be named 'test.txt' if not specified otherwise."
            )
        }
    )

    eval_file_name: Optional[str] = field(
        default = None,
        metadata = {
            "help": (
                "Filename of the train data file."
            )
        }
    )

    eval_string: Optional[str] = field(
        default = None,
        metadata={
            "help": (
                "Json string with strings to be annotated for surprisal."
            )
        }
    )

    tokenizer_name_or_path: str = field(
        default = None,
        metadata = {
            "help": (
                "Either the name of a HuggingFace tokenizer or the path to a self-trained tokenizer."
            )
        }
    )

    model_type: str = field(
        default = None,
        metadata = {
            "help": (
                "Which AutoModel class to use."
            )
        }
    )
    
    model_name_or_path: str = field(
        default = None,
        metadata = {
            "help": (
                "Either the name of a HuggingFace model or the path to a self-trained model."
            )
        }
    )
    
    batch_size: int = field(
        default = 8,
        metadata = {
            "help": "Batch size for evaluation."
        }
    )

    device: Optional[str] = field(
        default = "cpu",
        metadata = {
            "help": ""
        }
    )
    
    output_dir: str = field(
        default = None,
        metadata = {
            "help": "Path to the evaluation results."
        }
    )

    output_file_name: str = field(
        default = "eval_results.tsv",
        metadata = {
            "help": (
                "Filename used to save evaluation results."
            )
        }
    )

    log_level: Literal["info", "warning", "error", "debug"] = field(
        default = "warning",
        metadata = {
            "help": (
                "Log level for evaluation, default 'transformers.logging.INFO'."
            )
        }
    )

    sum_subword_surprisal: bool = field(
        default = False,
        metadata = {
            "help": (
                "Sum surprisals based on '--subword_prefix', i.e., if a token"
                "does not begin with the prefix its suprisal is added to that"
                "of the previous token."
            )
        }
    )

    subword_prefix: str = field(
        default = "Ġ", # byte pair pretokenizer (e.g. GPT-2)
        metadata = {
            "help": (
                "Prefix of words (of the first part of a word split into subwords)."
            )
        }
    )

    prepend_token: bool = field(
        default = False,
        metadata = {
            "help": (
                "Prepend an eos token to each batch of sequences. This is necessary for some tokenizers, like those used by the pre-trained gpt2 model."
            )
        }
    )


@dataclass
class TokenizerArgs:

    input_files_path: str = field(
        default = None,
        metadata = {
            "help" : (
                "Path to the directory containing the training files of the tokenizer."
            )
        }
    )

    input_files_type: str = field(
        default = "txt",
        metadata = {
            "help": "File type of input files, e.g. '.txt'"
        }
    )

    tokenizer_type: Literal["bpe", "unigram", "word-level"] = field(
        default = None,
        metadata = {
            "help": (
                "Algorithm of the tokenizer. Should be 'bpe', 'unigram' or 'word-level'."
            )
        }
    )

    vocab_size: int = field(
        default = None,
        metadata = {
            "help": (
                "The vocabulary size of the tokenizer."
            )
        }
    )

    output_dir: str = field(
        default = None,
        metadata = {
            "help": (
                "Where to save the tokenizer."
            )
        }
    )

    pad_token: str = field(
        default = "<pad>",
    )

    eos_token: str = field(
        default = "</s>",
    )

    unk_token: str = field(
        default = "<unk>",
    )

    lossy_context: bool = field(
        default = False,
        metadata = {
            "help": (
                "Whether to add the lossy context token to the tokenizer."
            )
        }
    )

    lossy_context_token: str = field(
        default = "<b>",
        metadata = {
            "help": (
                "Sequence breaking token to be used with lossy context (will be added to the tokenizer)"
            )
        }
    )


@dataclass
class CoNLLUArgs:

    annotation_style: Literal["misc", "column"] = field(
        default = "misc",
        metadata = {
            "help": (
                "How annotation should take place. 'misc': surprisal is added to the misc column."
                "'column': surprisal is added in a new column, breaking the conllu format."
            )
        }
    )

    tag: str = field(
        default = None,
        metadata = {
            "help": (
                "Tag for the surprisal column."
            )
        }
    )


def get_args(parser_type):
    return {
        "train": [BaseArgs, TrainingArguments, TrainingInputArgs, WandbArgs], 
        "eval": [BaseArgs, EvalArgs],
        "eval-conllu": [BaseArgs, CoNLLUArgs, EvalArgs],
        "tokenization": [TokenizerArgs]
    }[parser_type]


class ArgumentParserFactory():
    @classmethod
    def get_argparser(cls, parser_type):
        args = get_args(parser_type)
        parser = HfArgumentParser(args)
        return parser
