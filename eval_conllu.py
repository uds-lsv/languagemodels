import itertools
import os

from datasets import Dataset, DatasetDict
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, default_data_collator
from transformers.utils import logging

from languagemodels.argparser_factory import ArgumentParserFactory
from languagemodels.lm_factory import LMFactory
from languagemodels.scripting_utils import (
    tokenize_function, 
    preprocess_function_eval,
    prefix_eos_token,
    compute_batch_surprisal,
    repackage_hidden,
    load_conllu_file,
    save_conllu_file,
    get_word_surprisal
)


def main():

    parser = ArgumentParserFactory.get_argparser("eval-conllu")
    base_args, conllu_args, eval_args, = parser.parse_args_into_dataclasses()

    # determine surprisal column format
    if conllu_args.tag is not None:
        surp_col_name = f"srp_{conllu_args.tag}"
    else:
        surp_col_name = "srp"

    # initialize logger
    verbosity = logging.log_levels[eval_args.log_level]
    logging.set_verbosity(verbosity)
    # logger = logging.get_logger("transformers")

    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name_or_path)

    # set padding token if not set already
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    model, config = LMFactory.from_pretrained(base_args.auto_model_class, \
                                              eval_args.model_name_or_path)
    model.to(eval_args.device)
    model.eval()

    # if no eval file is specified, we instead iterate over all the files in
    if eval_args.eval_file_name is not None:
        conllu_file = os.path.join(eval_args.input_files_path, eval_args.eval_file_name)
        assert os.path.isfile(conllu_file), \
            f"File {eval_args.eval_file_name} does not exist at {eval_args.input_files_path} !"
        assert conllu_file.endswith(".conllu"), "Eval file has to be .conllu !"
        conllu_files = [conllu_file]
        
    else:
        conllu_files = [os.path.join(eval_args.input_files_path, conllu_file) \
                        for conllu_file in os.listdir(eval_args.input_files_path) \
                            if conllu_file.endswith(".conllu")]
        assert len(conllu_files) > 0, f"No conllu files found in {eval_args.eval_file_name} !"

    for conllu_file in tqdm(conllu_files):

        sents, ids, dfs = load_conllu_file(conllu_file)
        dataset_dict = {"test": Dataset.from_dict({"text": sents})}
        raw_dataset = DatasetDict(dataset_dict)

        if not os.path.exists(eval_args.output_dir):
            os.makedirs(eval_args.output_dir)

        output_file_path = os.path.join(eval_args.output_dir, conllu_file.split("/")[-1])

        # determine length of input sequences:
        if base_args.block_size is not None:
            block_size = base_args.block_size
        elif hasattr(config, "block_size"):
            block_size = config.block_size
        elif hasattr(config, "max_length"):
            block_size = config.max_length
        else:
            raise ValueError(
                "If config doesn't have a 'max_length' or 'block_size' attribute,"
                "this has to be provided via the '--block_size' CL argument"
            )
        
        tokenized_datasets = raw_dataset.map(
            tokenize_function,
            batched=True,
            batch_size=eval_args.batch_size,
            num_proc=1,  # no parallel processing for now
            fn_kwargs={
                "tokenizer": tokenizer,
                "model_type": config.model_type
            },
            desc="Running tokenizer on datasets"
        )

        lm_datasets = tokenized_datasets.map(
            preprocess_function_eval,
            batched=True,
            batch_size=1000,  # currently we group groups of 1000 samples together. For each of these groups we might delete some tokens
            num_proc=1,  # no parallel processing for now
            fn_kwargs={
                "tokenizer": tokenizer, 
                "model_max_length": block_size,
                "stride": block_size
            },
            desc=f"Grouping datasets into chunks of size {block_size}"
        )

        test_dataset = lm_datasets["test"].with_format(
            type="torch", columns=["input_ids", "labels", "attention_mask", "sequence_ids", ])

        test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, \
                                     batch_size=eval_args.batch_size, drop_last=False)

        doc_tokens, doc_surprisal = [], []

        with open(output_file_path, "a") as f:

            for batch in tqdm(test_dataloader, \
                              desc=f"file={conllu_file.split('/')[-1]}, bs={eval_args.batch_size}"):
                
                # if config.model_type in ["gpt2", "opt"]:
                #     # prepend eos_token_id to every sentence in the batch, where a sentence is a tensor of token_ids
                #     eos_token_id = tokenizer.encode(tokenizer.eos_token)[0] # 50256
                #     actual_batch_size = batch["input_ids"].shape[0]
                #     eos_tensor = torch.tensor([[eos_token_id] for _ in range(actual_batch_size)])
                #     sequence_id_tensor = torch.tensor([[seq_ids[0]] for seq_ids in batch["sequence_ids"]])
                #     attention_mask_tensor = torch.tensor([[1] for _ in range(actual_batch_size)])
                #     batch["input_ids"] = torch.cat([eos_tensor, batch["input_ids"]], dim=-1)
                
                #     batch["labels"] = batch["input_ids"].detach().clone()
                #     batch["attention_mask"] = torch.cat([attention_mask_tensor, batch["attention_mask"]], dim=-1)
                #     batch["sequence_ids"] = torch.cat([sequence_id_tensor, batch["sequence_ids"]], dim=-1)
                batch = prefix_eos_token(batch, tokenizer.eos_token_id)

                # delete batch ids for forward pass
                sequence_id_tensor = batch.pop("sequence_ids")
                
                # put data on device
                batch = {k: v.to(eval_args.device) for k, v in batch.items()}
                    
                with torch.no_grad():
                    # forward pass
                    if config.model_type == "rnn-lm":
                        outputs = model(
                            input_ids=batch["input_ids"], 
                            labels=batch["labels"],
                            pad_id=tokenizer.pad_token_id,
                            return_dict=True
                        )
                        repackage_hidden(outputs["final_hidden_state"])
                    else:
                        outputs = model(**batch, return_dict=True)

                    batch_surprisal = compute_batch_surprisal(
                        batch["input_ids"], 
                        batch["attention_mask"], 
                        outputs["logits"], 
                        sequence_id_tensor,
                        tokenizer
                    )

                    doc_tokens.extend(batch_surprisal["tokens"])
                    doc_surprisal.extend(batch_surprisal["surprisal"])


            if eval_args.sum_subword_surprisal:
                words = list(itertools.chain.from_iterable([s.split() for s in sents]))
                doc_words, doc_word_surprisal = \
                    get_word_surprisal(surprisal=doc_surprisal, tokens=doc_tokens, tokenizer=tokenizer, words=words, subword_prefix=eval_args.subword_prefix)
            else:
                doc_words, doc_word_surprisal = doc_tokens, doc_surprisal
            
            # annotate misc column with surprisal
            start_idx = 0


            for df in dfs:

                split_idx = len(df)
                surprisal_df = doc_word_surprisal[start_idx:start_idx+split_idx]
                
                # save surprisal annotation in misc column
                if conllu_args.annotation_style == "misc":
                    new_misc_col = [str(misc) + f"|{surp_col_name}={np.round(surp, 4)}" \
                                    for misc, surp in zip(df["MISC"].tolist(), surprisal_df)]
                    df["MISC"] = new_misc_col
                
                # save surprisal annotation in separate column
                elif conllu_args.annotation_style == "column":
                    df[surp_col_name] = surprisal_df

                start_idx += split_idx

            save_conllu_file(output_file_path, dfs)


if __name__ == "__main__":
    main()
