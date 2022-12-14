# Based on https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py
from abc import ABC, abstractmethod
import itertools
import json
import os
from pathlib import Path
import collections

# from transformers.tokenization_utils_base import AddedToken, PreTrainedTokenizerBase
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from typing import Dict, List, Union, Optional, Tuple

from languagemodels.tokenization.char_level_tokenizer.char_level_tokenization_functions import (
    CharTokenizationFunction,
    IpaTokenizationFunction,
    RegexTokenizationFunction
)

TOKENIZATION_FUNCTIONS = {
    "ipa": IpaTokenizationFunction,
    "regex": RegexTokenizationFunction
}

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class CharLevelTokenizer(PreTrainedTokenizer):
    """Simple character tokenizer.
        Args:
            model_max_length (int): Model maximum sequence length.
            vocab_file (Union[str, os.PathLike]): Path to the vocabulary file. 
                If present the characters in this file will be form the vocabulary.
    """
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, 
        model_max_length: int,
        vocab_file: Union[str, os.PathLike]=None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="<sep",
        cls_token="<cls>",
        pad_token="<pad>",
        mask_token="<mask>",
        unk_token="<unk>",
        **kwargs
    ):
        self.model_max_length = model_max_length
        self.tokenization_function = None

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            # add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

        # TODO (js): This is quite different from e. g. the implementation of the BERT 
        # tokenizer (https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/tokenization_bert.py)
        # This class SHOULD behave exactly like the BERT tokenizer.
        if vocab_file is not None:
            assert ".json" in vocab_file
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            self._vocab_int_to_str = collections.OrderedDict(
                [(ids, token) for token, ids in vocab.items()])
       
        else:
            self._vocab_int_to_str = collections.OrderedDict([
                (0, "<cls>"),
                (1, "<sep>"),
                (2, "<s>"),
                (3, "</s>"),
                (4, "<pad>"),
                (5, "<unk>"),
                (6, "<mask>")
            ])

        self._vocab_str_to_int = {v: k for k, v in self._vocab_int_to_str.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        tokenized = self.tokenization_function(text)
        tokenized = [self.bos_token] + tokenized + [self.eos_token]
        return tokenized

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> int:
        if isinstance(tokens, list):
            return [self._vocab_str_to_int.get(token, self._vocab_str_to_int["<unk>"]) \
                for token in tokens]
        else:
            return self._vocab_str_to_int.get(tokens, self._vocab_str_to_int["<unk>"])

    def convert_ids_to_tokens(self, indices: Union[int, List[int]]) -> str:
        if isinstance(indices, list):
            return [self._vocab_int_to_str.get(index, self.unk_token) \
                for index in indices]
        else:
            return self._vocab_int_to_str.get(indices, self.unk_token)

    def tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def train(self, files):
        assert self.tokenization_function
        chars = set()
        for file in files:
            with open(file, 'r') as in_file:
                sentences = in_file.read().split('\n')
                sentences = [sent.split() for sent in sentences]
                sentences_flat = list(itertools.chain.from_iterable(sentences))
                s = "".join(sentences_flat)
                s = self.tokenization_function(s)
                chars.update(s)

        for c in chars:
            self._vocab_str_to_int[c] = len(self._vocab_str_to_int)
            self._vocab_int_to_str[len(self._vocab_int_to_str)] = c

    def encode_batch(self, input, add_special_tokens=False):
        # Tokenizer.batch_decode_plus and the like are deprecated.
        encodings = self(input, add_special_tokens=add_special_tokens, 
            padding='max_length', truncation=True)
        self._add_items_to_encodings(encodings)
        return encodings

    def _add_items_to_encodings(self, encodings):
        """ Adds tokens, word ids, word offsets and the special tokens mask 
            to a BatchEncoding object
        """
        encodings_ids = []
        encodings_tokens = []
        encodings_word_ids = []
        encodings_word_offsets = []
        encodings_special_tokens_masks = []
        for ids in encodings["input_ids"]:
            tokens = [self.convert_ids_to_tokens(i) for i in ids]
            word_ids = [i if id != self.pad_token_id else 0 for i, id in enumerate(ids)]
            word_offsets = [(idx, idx) if i != self.pad_token_id else (0, 0) \
                for idx, i in enumerate(ids)]
            special_tokens_masks = self.get_special_tokens_mask(ids, \
                already_has_special_tokens=True)
            encodings_ids.append(ids)
            encodings_tokens.append(tokens)
            encodings_word_ids.append(word_ids)
            encodings_word_offsets.append(word_offsets)
            encodings_special_tokens_masks.append(special_tokens_masks)

        encodings["tokens"] = encodings_tokens 
        encodings["word_ids"] = encodings_word_ids
        encodings["offsets"] = encodings_word_offsets
        encodings["special_tokens_mask"] = encodings_special_tokens_masks

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES["vocab_file"])
        
        with open(vocab_file, 'w') as f:
            out_str = json.dumps(self._vocab_str_to_int, ensure_ascii=False)
            f.write(out_str)
        return (vocab_file, )

    def set_tokenization_function(self, tokenization_function: CharTokenizationFunction):
        self.tokenization_function  = tokenization_function

    @property
    def special_tokens(self) -> List[str]:
        return list(self.special_tokens_map.values())

    @property
    def characters(self) -> List[str]:
        return [char for char in self._vocab_str_to_int if char not in self.special_tokens]