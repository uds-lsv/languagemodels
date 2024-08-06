import os
import json

from datasets import load_dataset, Dataset, DatasetDict
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
    repackage_hidden
)


def main():

    parser = ArgumentParserFactory.get_argparser("eval")
    base_args, eval_args, = parser.parse_args_into_dataclasses()

    # initialize logger
    verbosity = logging.log_levels[eval_args.log_level]
    logging.set_verbosity(verbosity)
    logger = logging.get_logger("transformers")

    tokenizer = AutoTokenizer.from_pretrained(eval_args.tokenizer_name_or_path)

    # If the tokenizer doesn't provide a pad token, set it to be the eos token
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model, config = LMFactory.from_pretrained(base_args.auto_model_class, eval_args.model_name_or_path)
    model.to(eval_args.device)
    model.eval()

    # print('MODEL PARAMETERS', model.num_parameters(exclude_embeddings=True))

    # file
    if eval_args.eval_file_name:
        dataset_files = {
            "test": eval_args.eval_file_name
        }
        raw_dataset = load_dataset(
            path=eval_args.input_files_path, 
            data_files=dataset_files
        )

    # json string from web interface
    elif eval_args.eval_string:
        data_dict = json.loads(eval_args.eval_string)
        dataset_dict = {"test": Dataset.from_dict(data_dict)}
        raw_dataset = DatasetDict(dataset_dict)

    
    if not os.path.exists(eval_args.output_dir):
        os.makedirs(eval_args.output_dir)

    output_file_path = os.path.join(eval_args.output_dir, eval_args.output_file_name)

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
        batch_size=100,
        num_proc=1,  # no parallel processing for now
        fn_kwargs={"tokenizer": tokenizer, "model_type": eval_args.model_name_or_path},
        desc="Running tokenizer on datasets"
    )

    actual_sequences_size = len(tokenized_datasets["test"]["input_ids"])

    lm_datasets = tokenized_datasets.map(
        preprocess_function_eval,
        batched=True,
        batch_size=actual_sequences_size, # process the whole test set at once to retain sequence ids
        fn_kwargs={"tokenizer": tokenizer, "model_max_length": block_size, "stride": block_size},
        desc=f"Grouping datasets into chunks of size {block_size}" 
    )

    test_dataset = lm_datasets["test"].with_format(
        type="torch", columns=["input_ids", "labels", "attention_mask", "sequence_ids"])
    
    # preview data
    n = np.min((len(test_dataset), 10))
    if n > 1:
        for i in range(n):
            logger.info(test_dataset[i])

    test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=1, drop_last=False)

    with open(output_file_path, "a") as f:
        f.write("sentence_id\ttoken\tsurprisal\ttoken_id\n")

        for batch in tqdm(test_dataloader):

            if eval_args.prepend_token:
                batch = prefix_eos_token(batch, tokenizer.eos_token_id)

            sequence_ids = batch["sequence_ids"]
            del batch["sequence_ids"] # remove sequence_ids from batch for forward pass
            
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
                    sequence_ids, 
                    tokenizer
                )

            # save results to tsv, for each batch
            for seq_id, tok, id, srp in zip(batch_surprisal["sequence_ids"], batch_surprisal["tokens"], batch_surprisal["surprisal"], batch_surprisal["token_ids"]):
                out_str = f"{seq_id}\t{tok}\t{id}\t{srp}\n"
                f.write(out_str)


if __name__ == "__main__":
    main()
