# Based on https://github.com/dariush-bahrami/character-tokenizer/blob/master/charactertokenizer/core.py
from abc import ABC, abstractmethod
import itertools
import json
import os
from pathlib import Path

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from typing import Dict, List, Union, Optional

from languagemodels.tokenization.char_based_tokenizer.tokenization_functions import (
    CharTokenizationFunction,
    IpaTokenizationFunction,
    RegexTokenizationFunction
)

TOKENIZATION_FUNCTIONS = {
    "ipa": IpaTokenizationFunction,
    "regex": RegexTokenizationFunction
}

class CharacterBasedTokenizer(PreTrainedTokenizer):
    def __init__(self, model_max_length: int, tokenization_function: CharTokenizationFunction, \
         characters=None, **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                <unk> with id=6.
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
            char_match (str): Regular expression defining what is to be understood as a character.
        """
        self.model_max_length = model_max_length
        self.tokenization_function = tokenization_function
        bos_token = AddedToken("<s>", lstrip=False, rstrip=False)
        eos_token = AddedToken("</s>", lstrip=False, rstrip=False)
        sep_token = AddedToken("<sep>", lstrip=False, rstrip=False)
        cls_token = AddedToken("<cls>", lstrip=False, rstrip=False)
        pad_token = AddedToken("<pad>", lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)

        mask_token = AddedToken("<mask>", lstrip=True, rstrip=False)      

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

        if characters:
            self._vocab_str_to_int = {
                "<cls>": 0,
                "<sep>": 1,
                "<s>": 2,
                "</s>": 3,
                "<pad>": 4,
                "<unk>": 5,
                "<mask>": 6,
                **{ch: i + 7 for i, ch in enumerate(characters)},
            }
        else:
            self._vocab_str_to_int = {
                "<cls>": 0,
                "<sep>": 1,
                "<s>": 2,
                "</s>": 3,
                "<pad>": 4,
                "<unk>": 5,
                "<mask>": 6
            }

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        tokenized = self.tokenization_function(text)
        tokenized = [self.bos_token] + tokenized + [self.eos_token]
        return tokenized

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["<unk>"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

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
        chars = set()
        for file in Path(files).iterdir():
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
            tokens = [self._convert_id_to_token(i) for i in ids]
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

    def get_config(self) -> Dict:
        return {
            "characters": self.characters,
            "tok_function_name": self.tokenization_function.name,
            "tok_function_config": self.tokenization_function.get_config(),
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict):
        cfg = {}
        cfg["characters"] = config["characters"]
        # load tokenization function
        tok_function_name = config["tok_function_name"]
        tok_function_cfg = config["tok_function_config"]
        tok_function = TOKENIZATION_FUNCTIONS[tok_function_name](**tok_function_cfg)
        cfg["tokenization_function"] = tok_function
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save(self, save_file: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_file)
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_file(cls, path, **kwargs) -> "CharacterBasedTokenizer":
        cfg_file = Path(path)
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)

    @property
    def special_tokens(self) -> List[str]:
        return list(self.special_tokens_map.values())

    @property
    def characters(self) -> List[str]:
        return [char for char in self._vocab_str_to_int if char not in self.special_tokens]