import argparse
import time
from itertools import chain
import os
import logging

from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import default_data_collator

from languagemodels import LMFactory, TokenizerFactory
from logging_utils import WandbLogger

from utils import set_seed, get_timestamp, create_dir


def parse_args():
    parser = argparse.ArgumentParser(description="languagemodels")

    # data args
    parser.add_argument('--train_file', type=str, help="path to training data")
    parser.add_argument('--validation_file', type=str,
                        help="path to validation data")

    # config args
    parser.add_argument('--config_name_or_path', type=str,
                        help="name or path of the config file to use")

    # model args
    parser.add_argument('--model_type', type=str,
                        help="type of the LM to use")
    parser.add_argument('--model_name_or_path', type=str,
                        help="name or path of the LM to use")

    # tokenizer args
    parser.add_argument('--tokenizer_path', type=str,
                        help="path of the pre-trained tokenizer to use")
    parser.add_argument('--tokenizer_type', type=str, default="pretrained-tokenizer",
                        help="path of the pre-trained tokenizer to use")
    parser.add_argument('--tokenizer_name', type=str,
                        help="name the pre-trained tokenizer to use (from the huggingface hub)")
    # TODO(mm): one of these has to be != None

    # optimizer args
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="learning rate")

    # training args
    parser.add_argument('--batch_size', type=int, default=100,
                        help="batch size")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="log every logging_steps steps")
    parser.add_argument('--eval_steps', type=int, default=100,
                        help="evaluate every logging_steps steps")
    parser.add_argument('--max_steps', type=int, default=1000,
                        help="maximum number of steps to train")

    # wandb args
    parser.add_argument('--wandb_project_name', type=str,
                        help="name of the wandb project used for logging stuff")
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help="wandb name for the current run. If None, a random name will be choosen.")
    parser.add_argument('--wandb_group_name', type=str, default="no-group",
                        help="wandb group name for the current run.")

    # additional args
    parser.add_argument('--output_dir', type=str, required=True,
                        help="where to save the trained model and other outputs")
    parser.add_argument('--device', type=str, default='cpu',
                        help="device to use for compute, examples: cpu|cuda|cuda:0")
    parser.add_argument('--seed', type=int, default=123, help="seed")

    args = parser.parse_args()

    # TODO(mm): use assertions for some args

    if os.environ["WANDB_DISABLED"] == "false":  # we are using wandb for logging
        assert args.wandb_project_name is not None, "when using wandb, you need to specify a project name"

    return args


def repackage_hidden(hidden_state):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(hidden_state, torch.Tensor):
        return hidden_state.detach()
    else:
        return tuple(repackage_hidden(t) for t in hidden_state)


def evaluate(model, validation_dataset, args):
    model.eval()
    # create validation dataloader
    validation_dataloader = DataLoader(
        validation_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=args.batch_size)

    # TODO(mm): implement

    model.train()  # put model back in training mode
    results = {}
    return results


def main():
    args = parse_args()

    # -------------------- setup --------------------

    set_seed(args.seed)
    TIMESTAMP = get_timestamp()

    # create a unique dir for the current run
    dir_name = f"run_{TIMESTAMP}"
    args.output_dir = create_dir(args.output_dir, dir_name)
    if os.environ["WANDB_DISABLED"] == "false":
        args.wandb_output_dir = create_dir(args.output_dir, "wandb")

    # -------------------- end setup --------------------

    # -------------------- data loading --------------------

    # load training and evaluation data
    data_files = {
        "train": args.train_file,
        "validation": args.validation_file
    }
    dataset_args = {}
    extension = args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        # dataset_args["keep_linebreaks"] = False

    raw_dataset = load_dataset(
        extension, data_files=data_files, **dataset_args)

    # -------------------- end data loading --------------------

    # -------------------- model and tokenizer --------------------

    # load tokenizer
    # tokenizer = Tokenizer.from_pretrained(
    #     args.tokenizer_name) if args.tokenizer_name is not None else Tokenizer.from_file(args.tokenizer_path)
    
    tokenizer = TokenizerFactory.get_tokenizer(tokenizer_type=args.tokenizer_type, tokenizer_name=args.tokenizer_name, \
        tokenizer_path=args.tokenizer_path)

    # load model
    model, model_config = LMFactory.get_lm(model_type=args.model_type, config_name_or_path=args.config_name_or_path,
                                           pre_trained=True if args.model_name_or_path is not None else False, model_name_or_path=args.model_name_or_path)
    model.to(args.device)

    # -------------------- end model and tokenizer --------------------

    # -------------------- process data --------------------

    # tokenize and encode data
    def tokenize_function(sequences):
        # we tokenize in batch mode, hence sequences will be batch_size many sequences
        # the text field holds the input text
        # map expects a dictionary
        output_dict = {"input_ids": [],
                       "tokens": [], "attention_mask": [], "special_tokens_mask": [], "word_ids": []}
        outputs = tokenizer.encode_batch(
            sequences["text"])  # returns an Encoding object (https://huggingface.co/docs/tokenizers/v0.13.0/en/api/encoding#tokenizers.Encoding)

        for output in outputs:
            output_dict["input_ids"].append(output.ids)
            output_dict["tokens"].append(output.tokens)
            output_dict["word_ids"].append(output.word_ids)
            output_dict["attention_mask"].append(output.attention_mask)
            output_dict["special_tokens_mask"].append(
                output.special_tokens_mask)

        return output_dict

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        num_proc=1,  # no parallel processing for now
        desc="Running tokenizer on datasets"
    )

    # group data into chunks of length block_size (block_size will be model specific, e.g. BigramLM)
    block_size = model.block_size
    # the stride is an important assumption we make on the format of our input data
    stride = 1 if args.model_type == "bigram-lm" else block_size

    def group_sequences(sequences):
        # concatenate all sequences into a single sequence
        concatenated_examples = {
            k: list(chain(*sequences[k])) for k in sequences.keys()}
        concatenated_examples["text"] = "".join(
            concatenated_examples["text"]).split()

        actual_total_length = len(concatenated_examples["input_ids"])

        # make sure data is divisible by block_size
        # as a result, we might ingore some tokens at the end of the sequence
        # TODO(mm): pad the last sequence instead
        if actual_total_length >= block_size:
            total_length = (actual_total_length // block_size) * block_size
            print(
                f"Deleting {actual_total_length - total_length} tokens")

        # group sequences into blocks of length block_size
        # depending on the stridge, these blocks might be overlapping or not
        # e.g. assume our sequence is <s> A B C D E F G </s> and we have a block_size of 2 and a stride of 1
        # --> [<s> A] [A B] [B C] [C D] [D E] [F G] [G </s>]
        result = {
            k: [values[i: i + block_size]
                for i in range(0, total_length, stride)]
            for k, values in concatenated_examples.items()
        }

        # all models will shift the labels, so here it's fine to simply copy the inputs
        # for the bigram LM this is a bit of a waste of memory but we ignore this for now
        result["labels"] = result["input_ids"].copy()

        return result

    lm_datasets = tokenized_datasets.map(
        group_sequences,
        batched=True,
        batch_size=1000,  # currently we group groups of 1000 samples together. For each of these groups we might delete some tokens
        num_proc=1,  # no parallel processing for now
        desc=f"Grouping datasets into chunks of size {block_size}"
    )

    # format dataset. we keep only the integer columns as we have to convert them to tensors
    train_dataset = lm_datasets["train"].with_format(
        type="torch", columns=["input_ids", "labels", "attention_mask", ])
    validation_dataset = lm_datasets["validation"].with_format(
        type="torch", columns=["input_ids", "labels", "attention_mask", ])

    # print some examples from the training data
    for idx in range(1):
        print(train_dataset[idx])

    # -------------------- end process data --------------------

    # -------------------- optimizer --------------------

    # initialize optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.99), eps=1e-8
    )

    # -------------------- end optimizer --------------------

    # -------------------- training loop --------------------

    # setup wandb logging
    if os.environ["WANDB_DISABLED"] == "false":
        config = {**model_config.to_dict(), **vars(args)}
        wandb_logger = WandbLogger(
            args.wandb_project_name, args.wandb_output_dir, args.wandb_run_name, args.wandb_group_name, config)

    epoch = 1
    current_step = 0
    predicted_tokens = 0
    continue_training = True
    hidden_state = None
    model.train()  # put model in training mode
    while continue_training:
        # evaluate model before training
        eval_results = evaluate(model, validation_dataset, args)

        # ---> beginning of 1 epoch
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size, drop_last=True)

        print(f"--- Starting epoch {epoch} ---")
        for batch in train_dataloader:
            start_time = time.time()

            # put data on device
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # run forward pass and compute loss
            model.zero_grad(set_to_none=True)
            logits, loss, final_hidden_state = model.forward(
                input_ids=batch["input_ids"], labels=batch["labels"], hidden_state=hidden_state)

            if final_hidden_state is not None:
                # detach the hidden state to avoid backpropagating through all batches
                hidden_state = repackage_hidden(final_hidden_state)

            predicted_tokens += args.batch_size * (model.block_size - 1)

            # compute gradients and update weights
            loss.backward()
            optimizer.step()
            current_step += 1

            end_time = time.time()
            delta = (end_time - start_time) * 1000  # ms

            if current_step % args.logging_steps == 0:
                ppl = torch.exp(loss)
                
                # log stuff
                logs = {"global_step": current_step, "epoch": epoch, "loss": loss.item(), "ppl": ppl.item(), "predicted_tokens": predicted_tokens}
                wandb_logger.log(logs, step=current_step)
                
                print(
                    f"step: {current_step:>8} | loss: {loss.item():.4f} | ppl: {ppl.item():.4f} | predicted tokens: {predicted_tokens:>10} | step time: {delta:.2f}ms")

            if current_step % args.eval_steps == 0:
                eval_results = evaluate(model, validation_dataset, args)

            if current_step == args.max_steps:
                continue_training = False  # stop training
                break

        # ---> end of 1 epoch
        if continue_training:
            print(f"--- End of epoch {epoch} ---")
            epoch += 1

    # ---> end of training
    print(f"\n+++ End of training +++\n")

    # -------------------- end training loop --------------------

    # -------------------- saving and clean-up --------------------

    # save model, tokenizer, args
    if args.output_dir is not None:
        # make sure output dir exists
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_model(args.output_dir)

        # TODO(mm): save tokenizer, config, and args as well

    # -------------------- end saving and clean-up --------------------


if __name__ == '__main__':
    main()
