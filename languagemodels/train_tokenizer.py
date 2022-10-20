import argparse
import os
import time

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


SUPPORTED_TOKKENIZERS = [
    "word-level"
]


def _train_word_level_tokenizer(vocab_size, files):
    """Train a word-level tokenizer on a list of files

    Args:
        vocab_size (int): vocabulary size of the tokenizer
        files (List[str]): list of files on which the tokenizer will be trained 

    Returns:
        Tokenizer: a tokenizer
    """
    tokenizer = Tokenizer(model=WordLevel(unk_token="<unk>"))

    # these are default and can be overwritten
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>"]
    )

    # TODO(mm): look into Normalizers and additional Pre-tokenizers and provide good defaults

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[(t, tokenizer.token_to_id(t)) for t in ["<s>", "</s>"]]
    )

    return tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="languagemodels")
    parser.add_argument('--tokenizer-type', type=str, default="word-level",
                        choices=SUPPORTED_TOKKENIZERS,
                        help="directory which contains the input files")
    parser.add_argument('--input-files-dir', type=str,
                        help="directory which contains the input files")
    parser.add_argument('--input-files-type', type=str,
                        help="file type of input files, e.g. .raw")
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help="the tokenizer's vocabulary size")
    parser.add_argument('--output-dir', type=str,
                        help="where to save the tokenizer")
    parser.add_argument('--tokenizer-name', type=str,
                        help="name of the tokenizer")

    args = parser.parse_args()

    # get input files
    files = [
        os.path.join(args.input_files_dir, f) for f in os.listdir(args.input_files_dir) if
        os.path.isfile(os.path.join(args.input_files_dir, f)) and
        f.endswith(args.input_files_type)
    ]

    print("Creating tokenizer from files:", files)

    # load some data from the input files
    with open(files[0], encoding="utf-8") as f:
        data = f.readlines()  # read everything at once

    # print first 5 lines
    for idx, line in enumerate(data[:5]):
        print(idx, line)

    if args.tokenizer_type == "word-level":
        print("Training word-level tokenizer ...")
        tokenizer = _train_word_level_tokenizer(args.vocab_size, files)

    # print some tokenized and encoded data
    encoded_data = tokenizer.encode_batch(data[:5])
    for idx, encoded_line in enumerate(encoded_data):
        print(encoded_line.tokens, encoded_line.ids)

    # save tokenizer
    tokenizer_name = f"tokenizer-{args.tokenizer_name}.json"
    os.makedirs(args.output_dir, exist_ok=True)  # make sure output dir exists
    print("Done. Saving tokenizer to:", os.path.join(
        args.output_dir, tokenizer_name))
    tokenizer.save(os.path.join(args.output_dir, tokenizer_name))
