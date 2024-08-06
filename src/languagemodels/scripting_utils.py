import csv
from itertools import chain
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


CONLLU_COLS = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]


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


def tokenize_function(sequences, tokenizer, model_type):

    outputs = tokenizer(sequences["text"], padding="do_not_pad", truncation=False)

    # if the model uses the gpt-2 tokenizer, prepend the eos token to each sequence
    if model_type == "gpt2":
        eos_token_id = tokenizer.encode(tokenizer.eos_token)[0] # 50256
        outputs["input_ids"] = [input_ids + [eos_token_id] for input_ids in outputs["input_ids"]]
        outputs["attention_mask"] = [[1] + attention_mask for attention_mask in outputs["attention_mask"]]

    return outputs

def prefix_eos_token(batch: Dict, eos_token_id: int) -> Dict:

    # determine batch size
    actual_batch_size = batch["input_ids"].shape[0]
    
    # create prefix tensors
    eos_tensor = torch.tensor([[eos_token_id] for _ in range(actual_batch_size)])
    sequence_id_tensor = torch.tensor([[seq_ids[0]] for seq_ids in batch["sequence_ids"]])
    attention_mask_tensor = torch.tensor([[1] for _ in range(actual_batch_size)])

    # add prefixes to beginning of the sequences
    batch["input_ids"] = torch.cat([eos_tensor, batch["input_ids"]], dim=-1)
    batch["labels"] = batch["input_ids"].detach().clone()
    batch["attention_mask"] = torch.cat([attention_mask_tensor, batch["attention_mask"]], dim=-1)
    batch["sequence_ids"] = torch.cat([sequence_id_tensor, batch["sequence_ids"]], dim=-1)

    return batch


def preprocess_function_eval(sequences, tokenizer, model_max_length, stride, prefix_eos_token=False):

    # sequences["sequence_ids"] = []

    # # add sequence ids
    # for i in range(len(sequences["input_ids"])):
    #     sequence_ids = list(np.repeat(i, len(sequences["input_ids"][i])))
    #     sequences["sequence_ids"].append(sequence_ids)

    # concatenated_examples = {
    #     k: list(chain(*sequences[k])) for k in sequences.keys()}
    # # concatenated_examples["text"] = "".join(
    # #     concatenated_examples["text"]).split()

    # actual_total_length = len(concatenated_examples["input_ids"])

    # # make sure data is divisible by block_size
    # # as a result, we might ingore some tokens at the end of the sequence
    # if actual_total_length >= model_max_length:
    #     total_length = int(np.ceil(actual_total_length / model_max_length)) * model_max_length
    #     num_excess_tokens = actual_total_length - total_length
    #     to_pad = model_max_length - num_excess_tokens

    #     concatenated_examples["input_ids"] += [tokenizer.pad_token_id for _ in range(to_pad)]
    #     concatenated_examples["attention_mask"] += [0 for _ in range(to_pad)]
    #     concatenated_examples["sequence_ids"] += [concatenated_examples["sequence_ids"][-1] for _ in range(to_pad)]
    #     # concatenated_examples["text"] += [tokenizer.pad_token for _ in range(to_pad)]

    #     assert len(concatenated_examples["input_ids"]) % model_max_length == 0

    # # if a batch is smaller than the model's block size, pad to the full block size
    # elif actual_total_length < model_max_length:
    #     total_length = model_max_length
    #     to_pad = model_max_length - actual_total_length

    #     concatenated_examples["input_ids"] += [tokenizer.pad_token_id for _ in range(to_pad)]
    #     concatenated_examples["attention_mask"] += [0 for _ in range(to_pad)]
    #     concatenated_examples["sequence_ids"] += [concatenated_examples["sequence_ids"][-1] for _ in range(to_pad)]
    #     # concatenated_examples["text"] += [tokenizer.pad_token for _ in range(to_pad)]

    #     assert len(concatenated_examples["input_ids"]) % model_max_length == 0

    # # if data is already divisible by block size, use the length of the data instead
    # else:
    #     total_length = actual_total_length
    
    # # group sequences into blocks of length block_size
    # result = {
    #     k: [values[i: i + model_max_length]
    #         for i in range(0, total_length, stride)]
    #     for k, values in concatenated_examples.items()
    # }

    # TODO pad to max length in the batch, but at most to the models context window minus 1, and truncate sequences that are longer than that
    # TODO prefix eos token here (before truncation and padding)

    if prefix_eos_token:
        batch = {
            "input_ids": torch.tensor(sequences["input_ids"]),
            "attention_mask": torch.tensor(sequences["attention_mask"]),
        }

        eos_token_id = tokenizer.eos_token_id
        batch = prefix_eos_token(batch, eos_token_id)

        # Update sequences dictionary after prefixing EOS token
        sequences["input_ids"] = batch["input_ids"].tolist()
        sequences["attention_mask"] = batch["attention_mask"].tolist()

        max_length_in_batch = min(max(len(seq) for seq in sequences["input_ids"]), model_max_length - 1)

        sequences["input_ids"] = [seq[:max_length_in_batch] + [tokenizer.pad_token_id] * (max_length_in_batch - len(seq)) if len(seq) < max_length_in_batch else seq[:max_length_in_batch] for seq in sequences["input_ids"]]
        
        sequences["attention_mask"] = [mask[:max_length_in_batch] + [0] * (max_length_in_batch - len(mask)) if len(mask) < max_length_in_batch else mask[:max_length_in_batch] for mask in sequences["attention_mask"]]




    sequences["sequence_ids"] = [[sequence_id for _ in range(len(sequence))] for sequence_id, sequence in enumerate(sequences["input_ids"])]

    # copy inputs as labels in any case
    sequences["labels"] = sequences["input_ids"].copy()

    return sequences


def preprocess_function(sequences, tokenizer, model_max_length, stride):

    concatenated_examples = {
        k: list(chain(*sequences[k])) for k in sequences.keys()}
    concatenated_examples["text"] = "".join(
        concatenated_examples["text"]).split()

    actual_total_length = len(concatenated_examples["input_ids"])

    # make sure data is divisible by block_size
    # as a result, we might ingore some tokens at the end of the sequence
    if actual_total_length >= model_max_length:
        total_length = int(np.ceil(actual_total_length / model_max_length)) * model_max_length
        num_excess_tokens = actual_total_length - total_length
        to_pad = model_max_length - num_excess_tokens

        concatenated_examples["input_ids"] += [tokenizer.pad_token_id for _ in range(to_pad)]
        concatenated_examples["attention_mask"] += [0 for _ in range(to_pad)]
        concatenated_examples["text"] += [tokenizer.pad_token for _ in range(to_pad)]

        assert len(concatenated_examples["input_ids"]) % model_max_length == 0

    # if a batch is smaller than the model's block size, pad to the full block size
    elif actual_total_length < model_max_length:
        total_length = model_max_length
        to_pad = model_max_length - actual_total_length

        concatenated_examples["input_ids"] += [tokenizer.pad_token_id for _ in range(to_pad)]
        concatenated_examples["attention_mask"] += [0 for _ in range(to_pad)]
        concatenated_examples["text"] += [tokenizer.pad_token for _ in range(to_pad)]

        assert len(concatenated_examples["input_ids"]) % model_max_length == 0

    # if data is already divisible by block size, use the length of the data instead
    else:
        total_length = actual_total_length
        
    # group sequences into blocks of length block_size
    result = {
        k: [values[i: i + model_max_length]
            for i in range(0, actual_total_length, stride)]
        for k, values in concatenated_examples.items()
    }

    # copy inputs as labels in any case
    result["labels"] = result["input_ids"].copy()

    return result


def preprocess_function_sliding(sequences, tokenizer, T):

    concatenated_examples = {
        k: list(chain(*sequences[k])) for k in sequences.keys()}
    concatenated_examples["text"] = "".join(
        concatenated_examples["text"]).split()
    
    actual_total_length = len(concatenated_examples["input_ids"])

    # group sequences into blocks of length T
    result = {
        k: [values[i: i + T]
            for i in range(0, actual_total_length-T)]
        for k, values in concatenated_examples.items()
    }

    # copy inputs as labels in any case
    result["labels"] = result["input_ids"].copy()

    return result


def compute_batch_surprisal(batch_input_ids, batch_mask, batch_logits, sequence_ids, tokenizer):

    out_dict = {"surprisal": [], "tokens": [], "token_ids": [], "sequence_ids": []}

    for inner_batch, (input_ids, mask, logits, sequence_ids) in enumerate(zip(batch_input_ids, batch_mask, batch_logits, sequence_ids)):

        output_ids = input_ids[1:]
        output_mask = mask[1:]
        indices = torch.arange(0, output_ids.shape[0])
        sequence_ids = sequence_ids[1:] # first sequence id can be ignored

        assert len(sequence_ids) == len(output_ids)

        # ignore padded positions
        surprisals = -1*torch.log2(F.softmax(logits, dim=-1)).squeeze(0)[indices, output_ids]
        surprisals = [s for s, m in zip(surprisals.cpu().detach().numpy().tolist(), output_mask) if m > 0]
        tokens = [t for t, m in zip(tokenizer.convert_ids_to_tokens(output_ids), output_mask) if m > 0]
        token_ids = [i for i, m in zip(output_ids.cpu().detach().numpy().tolist(), output_mask) if m > 0]
        sequence_ids = [seq_id for seq_id, mask_id in zip(sequence_ids, output_mask) if mask_id > 0]

        # ignore eos token positions
        non_eos_token_indices = [i for i in range(len(tokens)) if tokens[i] != tokenizer.eos_token]

        surprisals = [surprisals[i] for i in non_eos_token_indices]
        tokens = [tokens[i] for i in non_eos_token_indices]
        token_ids = [token_ids[i] for i in non_eos_token_indices]
        sequence_ids = [sequence_ids[i] for i in non_eos_token_indices]
        batch_mask[inner_batch] = torch.tensor([[mask[0]] + [1 if i in non_eos_token_indices else 0 for i in range(len(output_mask))]])

        out_dict["surprisal"].extend(surprisals)
        out_dict["tokens"].extend(tokens)
        out_dict["token_ids"].extend(token_ids)
        out_dict["sequence_ids"].extend(sequence_ids)

    assert len(out_dict["surprisal"]) == batch_mask.sum().sum() - batch_mask.shape[0], f"{len(out_dict['surprisal'])}!={batch_mask.sum().sum() - batch_mask.shape[0]}"

    return out_dict


def compute_cloze_surprisal(input_ids, logits):
    
    assert input_ids.shape == logits.shape[:1]

    cloze_surprisal = -1*torch.log2(F.softmax(logits, dim=-1)).squeeze(0)[input_ids]
    
    return cloze_surprisal


def load_conllu_file(path: Union[os.PathLike, str]) -> Tuple[List[str], List[str], List[pd.DataFrame]]:
    
    with open(path, "r") as f:
        content = f.read()
        lines = content.split("\n")
        comments = [line for line in lines if line.startswith("#")]
        ids = [int(s.replace("# sent_id = ", "")) for s in comments[1:][::2]]
        dfs = [
            pd.read_csv(pd.io.common.StringIO('\n'.join(dat)), 
                        comment="#", header=None, sep="\t", 
                        names=CONLLU_COLS, quoting=csv.QUOTE_NONE,
                        encoding="utf-8")
            for dat in map(str.splitlines, content.split('\n\n')) if dat != ""
        ]

        sents = [" ".join(df["FORM"].astype(str).tolist()) for df in dfs]

        dfs = [df for df in dfs if not df.empty] # there might be a more elegant solution than this

    return sents, ids, dfs


def save_conllu_file(path: Union[os.PathLike, str], dfs: List[pd.DataFrame]) -> None:
   
    with open(path, "a") as f:
        for i, df in enumerate(dfs):
            text_str = f"# text = {' '.join(df['FORM'].astype(str).tolist())}\n"
            id_str = f"# sent_id = {i}\n"
            f.write(text_str)
            f.write(id_str)
            df.to_csv(f, sep="\t", header=False, index=False)
            f.write("\n")


def get_word_surprisal(surprisal: List[float], tokens: List[str], words: List[str], tokenizer, subword_prefix: str) -> List[pd.DataFrame]:
    
    expected_words = len(words)
    word_surprisal = []
    curr_word_ix = 0
    curr_word_surp = []
    curr_toks = ""

    # expected number of words = number of tokens that start with "Ġ" minus the number of EOS tokens
    for j in range(len(tokens)):
        # necessary for diacritics in Dundee
        # cleaned_tok = tokens[j].replace(subword_prefix, "", 1).encode("latin-1").decode("utf-8")
        cleaned_tok = tokens[j].replace(subword_prefix, "", 1)

        # ignore EOS tokens
        if cleaned_tok == tokenizer.eos_token:
            continue

        # for word-level surprisal
        curr_word_surp.append(surprisal[j])
        curr_toks += cleaned_tok

        # print(words)
        if j+1 == len(tokens) or j-1 == tokenizer.eos_token or tokens[j+1].startswith(subword_prefix) or tokens[j+1] == tokenizer.eos_token:
            # if curr_toks != words[curr_word_ix]:
            #     print("****************8")
            #     print(j, curr_word_ix, curr_toks, words[curr_word_ix])
            #     print(tokens[j-5:j+1])
            #     print(words[curr_word_ix-5:curr_word_ix+1])
            word_surprisal.append(round(sum(curr_word_surp),4))
            # print(curr_toks, sum(curr_word_surp))
            curr_word_surp = []
            curr_toks = ""
            curr_word_ix += 1

    assert expected_words == len(word_surprisal), f"Expected to find {expected_words} words, got {len(word_surprisal)}!"
    
    return words, word_surprisal


def apply_lossy_context(sequences, lossy_context_token: str, tokenizer):
    import random
    random.seed(0)

    lossy_sequences = {k: [] for k in sequences.keys()}

    lossy_context_token_id = tokenizer.convert_tokens_to_ids([lossy_context_token])[0]

    lossy_sequences = {k: [] for k in sequences}

    for input_ids, attention_mask, token_type_ids, text in zip(sequences["input_ids"], \
            sequences["attention_mask"], sequences["token_type_ids"], sequences["text"]):
        lossy_index = random.randint(1,len(input_ids))

        lossy_sequences["input_ids"].append(input_ids[:lossy_index]), \
            lossy_sequences["input_ids"].append([lossy_context_token_id] + input_ids[lossy_index:])
        lossy_sequences["attention_mask"].append(attention_mask[:lossy_index]), \
            lossy_sequences["attention_mask"].append([1] + attention_mask[lossy_index:])
        lossy_sequences["token_type_ids"].append(token_type_ids[:lossy_index]), \
                lossy_sequences["token_type_ids"].append([1] + token_type_ids[lossy_index:])
        text = text.split()
        lossy_sequences["text"].append(" ".join(text[:lossy_index-1])), \
            lossy_sequences["text"].append(" ".join(text[lossy_index-1:]))
    
    return lossy_sequences
