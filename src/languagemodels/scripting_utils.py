import os
from typing import List, Union

from tokenizers import Tokenizer
import torch
from transformers.tokenization_utils import PreTrainedTokenizer


def repackage_hidden(hidden_state):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    else:
        return tuple(repackage_hidden(t) for t in hidden_state)


def get_num_occurrences_in_tensor(value: int, t: torch.Tensor) -> int:
    word_ids, word_counts = torch.unique(t, return_counts=True)
    value_count = (word_counts == value).nonzero(as_tuple=True)
    if value_count.numel() == 0:
        return 0
    return value_count.item()


def save_tokenizer(tokenizer: Union[Tokenizer, PreTrainedTokenizer], \
    save_dir: Union[str, os.PathLike]) -> None:
    if isinstance(tokenizer, Tokenizer):
        tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    elif isinstance(tokenizer, PreTrainedTokenizer):
        tokenizer.save_pretrained(save_dir)


def ids_to_tokens(ids, tokenizer: Union[Tokenizer, PreTrainedTokenizer]) -> List[str]:
    if isinstance(tokenizer, Tokenizer):
        tokens = [tokenizer.id_to_token(id) for id in ids]
    elif isinstance(tokenizer, PreTrainedTokenizer):
        tokens = [tokenizer.convert_ids_to_tokens(id) for id in ids]
    return tokens