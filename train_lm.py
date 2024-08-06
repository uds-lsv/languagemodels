import os

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.data.data_collator import default_data_collator
from transformers.utils import logging
import wandb

from languagemodels import (
    ArgumentParserFactory, 
    LMFactory,
    TrainerFactory
)
from languagemodels.scripting_utils import (
    apply_lossy_context,
    tokenize_function, 
    preprocess_function
)
from languagemodels.utils import create_dir, get_timestamp


def main():

    parser = ArgumentParserFactory.get_argparser("train")
    base_args, train_args, input_args, wandb_args = parser.parse_args_into_dataclasses()

    # initialize logger
    verbosity = logging.log_levels[train_args.log_level]
    logging.set_verbosity(verbosity)
    logger = logging.get_logger("transformers")

    # create a unique dir for the current run
    TIMESTAMP = get_timestamp()
    dir_name = f"run_{TIMESTAMP}"
    train_args.output_dir = create_dir(train_args.output_dir, dir_name)

    tokenizer = AutoTokenizer.from_pretrained(input_args.tokenizer_name_or_path)
    # If the tokenizer doesn't provide a pad token, set it to be the eos token
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if input_args.config_name_or_path is not None:
        model, config = LMFactory.from_config(model_type=base_args.auto_model_class, config_name_or_path=input_args.config_name_or_path)
    elif input_args.model_name_or_path is not None:
        model, config = LMFactory.from_pretrained(model_type=base_args.auto_model_class, model_name_or_path=input_args.model_name_or_path)
    else:
        raise Exception("Either a pretrained model or a valid config file must be provided!")

    # wandb setup
    if os.environ["WANDB_DISABLED"] == "false":
        train_args.wandb_output_dir = create_dir(train_args.output_dir, "wandb")

        wandb.init(
            project=wandb_args.wandb_project_name,
            name=train_args.run_name,
            group=wandb_args.wandb_group_name,
            dir=train_args.wandb_output_dir,
            config=config
        )
    
    data_files = {
        "train": input_args.train_file_name,
        "validation": input_args.validation_file_name,
    }

    if train_args.do_predict:
        data_files["test"] = input_args.test_file_name

    raw_datasets = load_dataset(
        path=input_args.input_files_path,
        data_files=data_files
    )

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

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        batch_size=10000,
        num_proc=1,  # no parallel processing for now
        fn_kwargs={"tokenizer": tokenizer, "model_type": config.model_type},
        desc="Running tokenizer on datasets"
    )

    if input_args.lossy_context:

        assert input_args.lossy_context_token in tokenizer.get_vocab(), \
            f"Lossy context token {input_args.lossy_context_token} not in tokenizer!"
        
        pre_split_train_size = len(tokenized_datasets["train"]["input_ids"])

        tokenized_datasets = tokenized_datasets.map(
            apply_lossy_context,
            batched=True,
            batch_size=2,
            num_proc=1,  # no parallel processing for now
            fn_kwargs={"lossy_context_token": input_args.lossy_context_token, "tokenizer": tokenizer},
            desc="Running tokenizer on datasets"
        )

        post_split_train_size = len(tokenized_datasets["train"]["input_ids"])

        assert post_split_train_size == 2*pre_split_train_size


    preprocessed_datasets = tokenized_datasets.map(
        preprocess_function,
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

    train_dataset = preprocessed_datasets["train"].with_format(
            type="torch", columns=["input_ids", "labels", "attention_mask", ])
    validation_dataset = preprocessed_datasets["validation"].with_format(
            type="torch", columns=["input_ids", "labels", "attention_mask", ])
    
    # preview data
    n = np.min((len(train_dataset), 10))
    if n > 1:
        for i in range(n):
            logger.info(train_dataset[i])

    # initialize callbacks
    callbacks = []

    if input_args.use_early_stopping:
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=input_args.early_stopping_patience)
        callbacks.append(early_stopping_callback)

    trainer_args = dict(
        args=train_args,
        model=model.to(train_args.device),
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    trainer = TrainerFactory.get_trainer(config.model_type, **trainer_args)

    if train_args.do_train:
        trainer.train()
    if train_args.do_eval:
        trainer.evaluate()
    if train_args.do_predict:
        test_dataset = preprocessed_datasets["validation"].with_format(
            type="torch", columns=["input_ids", "labels", "attention_mask", ])
        trainer.predict(test_dataset=test_dataset)

    model.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    main()
