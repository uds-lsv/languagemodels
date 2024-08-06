#!/usr/bin/env python

import os
from typing import List

from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE, Unigram
from tokenizers.trainers import WordLevelTrainer, BpeTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import ByteLevel, Metaspace, Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from languagemodels.argparser_factory import ArgumentParserFactory


SUPPORTED_TOKENIZERS = {
    "word-level" : (WordLevel, WordLevelTrainer, Whitespace), 
    "bpe": (BPE, BpeTrainer, ByteLevel), 
    "unigram": (Unigram, UnigramTrainer, Metaspace), 
}


def train_tokenizer(tokenizer_type: str, data: List[str], vocab_size: int, \
                    eos_token: str="</s>", pad_token: str="<pad>", unk_token: str="<unk>", \
                    lossy_context: bool=False, lossy_context_token="<b>") -> Tokenizer:
    
    tokenizer_cls, trainer_cls, pre_tokenizer_cls = SUPPORTED_TOKENIZERS[tokenizer_type]
    tokenizer = Tokenizer(model=tokenizer_cls(unk_token=unk_token))
    special_tokens = [unk_token, eos_token, pad_token]
    if lossy_context:
        special_tokens.append(lossy_context_token)
    print(special_tokens)
    trainer = trainer_cls(
        vocab_size=vocab_size, 
        special_tokens=special_tokens
    )
    tokenizer.pre_tokenizer = pre_tokenizer_cls()
    tokenizer.train(data, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single=f"{eos_token} $A",
        special_tokens=[(t, tokenizer.token_to_id(t)) for t in [eos_token]]
    )

    return tokenizer


def main():

    parser = ArgumentParserFactory.get_argparser("tokenization")

    args, = parser.parse_args_into_dataclasses()

    # get input files
    files = [
        os.path.join(args.input_files_path, f) for f in os.listdir(args.input_files_path) if
        os.path.isfile(os.path.join(args.input_files_path, f)) and
        f.endswith(args.input_files_type)
    ]

    print("Creating tokenizer from files:", files)

    # load some data from the input files
    with open(files[0], encoding="utf-8") as f:
        data = f.read().split("\n")  # read everything at once

    # print first 5 lines
    for idx, line in enumerate(data[:5]):
        print(idx, line)

    tokenizer = train_tokenizer(
        args.tokenizer_type, files, args.vocab_size,
        args.eos_token, args.pad_token, args.unk_token,
        args.lossy_context, args.lossy_context_token
    )
    
    wrapped_tokenizer =  PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=args.pad_token,
        bos_token=args.eos_token,
        eos_token=args.eos_token
    )

    # print some tokenized and encoded data
    encoded_data = wrapped_tokenizer(data[:5])
    print(encoded_data)
    for input_ids in encoded_data["input_ids"]:
        tokens = wrapped_tokenizer.convert_ids_to_tokens(input_ids)
        print(input_ids, tokens)

    # save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)  # make sure output dir exists
    print("Done. Saving tokenizer to:", args.output_dir)
    wrapped_tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
