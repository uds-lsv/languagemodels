# languagemodels

A simple toolkit to train and evaluate language models.

## Citing languagemodels

To cite our languagemodels toolkit, use the following bibtex:

```bibtex
@Misc{peft,
  title =        {languagemodels: A simple toolkit to train and evaluate language models},
  author =       {Julius Steuer, Marius Mosbach, Dietrich Klakow},
  howpublished = {\url{https://github.com/uds-lsv/languagemodels}},
  year =         {2023}
}
```

To cite the LSV LM-GUI, use the following bibtex:

```bibtex
@Misc{peft,
  title =        {LM-GUI: A graphical user interface for n-gram language models},
  author =       {Adam Kusmirek, Clayton Greenberg, Youssef Oualil, Dietrich Klakow},
  howpublished = {\url{https://github.com/uds-lsv/languagemodels}},
  year =         {2023}
}
```

## Setup

### Python venv

- Create a new virtual environment: `python3 -m venv ./languagemodels-venv`
- Activate the virtual environment: `source languagemodels-venv/bin/activate`

### Python miniconda

- Create a new virtual environment: `conda create --name languagemodels python=3.7`
- Activate the virtual environment: `conda activate languagemodels`

### Install languagemodels package

- Upgrade pip: `pip install --upgrade pip`
- Install package & requirements: `pip install -e .`

### Pytorch

- Install the appropriate version of Pytorch before installing the package: https://pytorch.org/get-started/locally/

### Weights & Biases (wandb)

In order to log your runs with wandb, source the `setup.sh` script in the `examples/scripts` folder at the beginning of your bash scripts. You will need to enter your user and team name as well as the API key to the script first:

```bash
export WANDB_API_KEY=<API key>
export WANDB_USERNAME=<username>
export WANDB_ENTITY=<username>
```

Wandb will log the runs to the folders indicated in the folders exported at the start of `setup.sh`. You have to create these folders before first using wandb (i.e. manually).

```bash
export CACHE_BASE_DIR=/data/users/$USER/logs/your_project/logfiles
export OUTPUT_DIR=/data/users/$USER/logs/your_project/logfiles
```

### Running scripts

- Once the virtual environment is activated and the package has been installed, Python scripts from `languagemodels/` can be run from the command line using `python3 <name of script> <arguments>`, e.g.: `python3 eval_lm -h`.

- Sample scripts for training a tokenizer (`train_tokenizer.sh`), model (`train_lm.sh`) and using it to annotate surprisal on 
the test (`eval_lm.sh`) set can be found in `examples/scripts/wikitext-103`. These can be run using `sh <path to script>`, e.g. `sh examples/scripts/wikitext-103/eval_lm.sh`.

## Scripting

### Training a Tokenizer

| CL Argument         | Description               | Example |
| :------------------ | ------------------------- | ------- |
| `--tokenizer_type` | Type of the tokenizer. | See supported tokenizers section. |
| `--input_files_dir` | Path to a directory with input files (can be $\geq 1$ file) | `/data/corpora/wikitext-103-v1` |
| `--input_files_type` | File ending of the input files; will be used for processing. E.g. if the input files are in CoNLLU format, you can define a function that extracts the word column from each file. | `txt` |
| `--vocab_size` | The tokenizer's vocabulary size. | |
| `--block_size` | The tokenizer's 'maximal' output size. Will be used with the `truncation=True` parameter of the tokenizer. | Should be $\geq d_{model}$. |
| `--output_dir` | Directory in which the tokenizer files will be saved. | `/data/users/$USER/tokenizers/your_project/` |
| `--eos_token` | The tokenizer's EOS (end-of-sequence) token. Don't change this if you don't want to do something specific. Will be prefixed to every sequence. | `</s>` |
| `--unk_token` | The tokenizer's UNK (unknown) token. Don't change this if you don't want to do something specific. | `<unk>` |
| `--pad_token` | The tokenizer's PAD (padding) token. Don't change this if you don't want to do something specific. | `<pad>` |

All other CL arguments are identical to the HuggingFace transformers [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments).

### Supported Tokenizers

| Tokenizer                    | Description               | Pre-Tokenizer |
| :--------------------------- | ------------------------  | ------------- |
| `unigram` | HuggingFace implementation of the Unigram algorithm. | Metaspace |
| `bpe` | HuggingFace implementation of byte pair encodings. | ByteLevel |
| `word-level` | HuggingFace implementation of a word-level tokenizer. | Whitespace |

### Training a Model

| CL Argument         | Description               | Example |
| :------------------ | ------------------------- | ------- |
| `--input_files_path` | Path to train, validation and test files. | `/data/corpora/wikitext-103-v1/` |
| `--train_file_name` | File to train data with one sequence per line. Preferably in `/data/corpora/`. | `train.txt` |
| `--validation_file_name` | File to validation data with one sequence per line. Preferably in `/data/corpora/`. | `validation.txt` |
| `--config_name_or_path` | Either a path to the config JSON file. Must be (compatible with) a config class inheriting from `transformers.PreTrainedConfig` | See `examples/configs/lstm/basic_lstm.json` |
| `--model_type` | Which auto model class to use. Will be processed by `languagemodels.lm_factory` to load the config and model classes. | `causal` for a causal language model (e.g. GPT-2), `masked` for a masked language model (e.g. BERT), or `seq2seq` for a sequence-to-sequence model (e.g. T5). |
| `--model_name_or_path` | Path to a trained model (e.g. in case of custom models) or name of a HuggingFace model. | `gpt2` in combination with `--model_type=causal`. |
| `--tokenizer_name_or_path` | Either a path to a pretrained tokenizer or the name of a HuggingFace tokenizer. | `bert-base-uncased` |
| `--seed` | Random seed for PyTorch, Cuda, Numpy etc. | 42 |
| `--wandb_project_name` | Identifier of the wandb project. | `languagemodels` for the b4 languagemodels project. |
| `--wandb_group_name` | Used to group runs below the project level. Typically this should identify the experiment you are running, with the run names differentiating (e.g.) between models and random seeds. | |
| `--use_early_stopping` | Whether to use the `EarlyStoppingCallback` class or not. |  |
| `--early_stopping_patience` | Number of evaluations to wait without improvement in eval loss before stopping training. |  |
| `--auto_model_class` | The automodel class to use. | `--auto_model_class causal` for `AutoModelForCausalLM`. |

### Supported Models

Apart from models from the HuggingFace hub, the following custom models are supported:

| CL Argument         | Description               | Example |
| :---------------- | ------------------------- | --------- |
| `bigram-lm` | Special implementation of a bigram language model. |
| `rnn-lm` | Vanilla RNN, LSTM, GRU |
| `opt-with-alibi` | Our custom OPT model with ALiBI instead of positional embeddings |

### Evaluation

| CL Argument         | Description               | Example |
| :------------------ | ------------------------- | ------- |
| `--input_files_path` | Path to the file with the sequences to be evaluated. | `/data/corpora/wikitext-103-v1/` |
| `--eval_file_name` | File to evaluate data with one sequence per line. Preferably in `/data/corpora/`. | `test.txt` |
| `--eval_string` | This can be used to score a single sequence or multiple sequences given as a string of text in JSON-format. Primarily for use with the web interface. | `'{"text": ["This a sequence.", "This is another sequence."]}'` |
| `--model_name_or_path` | Path to a trained model (e.g. in case of custom models) or name of a HuggingFace model. | `gpt2` |
| `--batch_size` | The maximum number of sequences evaluated per batch. | 8 |
| `--device` | Device used for training. | `cuda`: Nvidia (or AMD) GPU; `cpu`: CPU. |
| `--tokenizer_name_or_path` | Either a path to a pretrained tokenizer or the name of a HuggingFace tokenizer. | `bert-base-uncased` |
| `--output_dir` | Evaluation results will be saved here. | `trained_ models/` |
| `--output_file_name` | Name of the output file. Should have the `.tsv` file ending. | `eval_results.tsv` |
| `--block_size` | Size to which the eval sequences will be reshaped, s.t. batches are of the shape $d_{batch} \times d_{block}$ |  |
| `--prepend_token` | Whether or not to prepend an eos token to each batch of sequences. This is necessary for some tokenizers, like those used by the pre-trained gpt2 model. | | 

## Example workflow

### 1. Connect to lsv contact server

-  Use the lsv account
- ssh user@contact.lsv.uni-saarland.de
 
### 2. Connect to student’s workstation

- Ids of workstations: 56, 26, 64, 77, 71, 76, 24, 31, 68, 66
- Go to student’s room 0.03 to check which ones are running
- Pattern: wsXXlx , where XX is one of the ids above
- `ssh wsXXls` (password is the lsv password)

### 3. Connect to GitLab

- Set up an ssh key for GitLab, see [GitLab documentation](https://docs.gitlab.com/ee/user/ssh.html)
- Clone the [b4/languagemodels repo](https://repos.lsv.uni-saarland.de/b4/languagemodels)
- Make sure to checkout the right branch

### 4. Create environment (Miniconda) and install project

- Install Miniconda:
  - Navigate to your home directory /home/<username>
  - Download the installation file from [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers )
  - Run:  `wget <linktofile>`
  - Install miniconda by running: `bash <filename>`
  - Upon successful installation, run `source .bashrc` to activate conda
- Create the environment:
  - `conda create -n <myenv> python=[version]`
  - (note: you don’t need to have the version installed, conda will do that for you)
  - (note 2: we’ve found that Python 3.8.13 works the best, i.e. no depreciation messages) 
- Activate the environment: conda activate <myenv>
- Install the project: `pip install -e . --user`

### 5. Docker

- Use the Dockerfile inside the repo:
  - run: `docker build -f [path_to_docker_file] --build-arg USER_UID=$UID --build-arg USER_NAME=$(id -un) -t docker.lsv.uni-saarland.de/[lsv_user_name]/[image_name]:[tag_name] .`
  - (make sure to change the `lsv_user_name`, `image_name`, `tag_name`)
  - (note: the -t flag is used to be able to identify our docker image in the registry)
- Push to LSV Docker registry:
  - `docker push docker.lsv.uni-saarland.de/[lsv_user_name]/[image_name]:[tag_name]`

NOTE:

To check images in the registry: `docker image ls`

To delete an image: `docker image rm [image id]`

### 6. The .sh file

- Create a .sh file:
  - This specifies which of the bin scripts you want to run, i.e. `train_lm`, `train_tokenizer` or `eval_lm`
  - For examples see `examples/scripts/`
  - Make sure to specify paths to training data, output data, config files etc.
  - This will call the binary that we want to execute, setting its arguments (e.g. hyperparameters)
  - Source `setup.sh` for use of wandb before calling the binary

###  7. The submit file

- Create a submit file:
  - Using the project’s docker file/image
  - Specify which .sh you want to run, e.g. train_lm.sh (executable)
- Cluster breakdown:
  - cl14, cl15 - smaller LSTMs (11GB VRAM)
  - cl16 - larger models (32GB VRAM)
  - cl18+ - needs a good reason to be used; e.g. >4 GPUs, or >32GB VRAM

### 8. Connect to submit server 

- `ssh submit`
- From here you can submit jobs via thesubmit file and HTCondor:
  - `condor_submit <path/to/.sub>`
  - submit file needs to end with .sub
- Monitor jobs with `condor_q`
  - If the job is idle, wait
  - If job is on hold:
    - `condor_q -hold [job_id]` (this will display the errors causing the job to be on hold)
  - Make sure to create a directory for your username under `/data/users` if not present
    - create path `/data/users/[username]/logs/languagemodels/logfiles`
    - Here, logfiles including a .err file will be stored, which contains important information about specific jobs. Format: \[job\_id\]\_\[datetime\].err
- Collect the output from the specified output directory once the job is done

### 9. Example submit file

- See `examples`

## Contributing

| Contributor    | Email                        |
|:---------------|:-----------------------------|
| Marius Mosbach | mmosbach@lsv.uni-saarland.de |
| Julius Steuer  | jsteuer@lsv.uni-saarland.de  |
| AriaRay Brown  | arbrown@lsv.uni-saarland.de |
