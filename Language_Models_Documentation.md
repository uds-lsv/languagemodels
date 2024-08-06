
# Language Models Documentation

## Models and Configurations

### BigramLM

#### `BigramLMConfig`

A configuration class for the Bigram language model. Inherits from `PretrainedConfig`.

- **Attributes:**
  - `vocab_size`: int, default=2 - The size of the vocabulary.
  - `max_length`: int, default=2 - The maximum length of the input sequences.

#### `BigramLM`

A simple Bigram Language Model.

- **Attributes:**
  - `config_class`: `BigramLMConfig` - Configuration class for the model.
  - `vocab_size`: int - The size of the vocabulary.
  - `max_length`: int - The maximum length of the input sequences.
  - `logits`: `torch.nn.Parameter` - The logits for bigram predictions.

- **Methods:**
  - `forward(input_ids, labels=None, **kwargs)`: Computes the forward pass.

### DummyLM

#### `DummyLMConfig`

A configuration class for the Dummy language model. Inherits from `PretrainedConfig`.

#### `DummyLM`

A dummy model for testing purposes.

- **Attributes:**
  - `config_class`: `DummyLMConfig` - Configuration class for the model.

- **Methods:**
  - `load_model(cls, config, pre_trained=False, model_name_or_path=None)`: Loads the model.
  - `forward(input_ids, labels=None, **kwargs)`: Computes the forward pass.

### OPTWithALiBi

#### `OPTWithALiBiAttention`

An implementation of attention with ALiBi (Attention with Linear Biases).

#### `OPTWithALiBiConfig`

Configuration class for the model with ALiBi. Inherits from `OPTConfig`.

#### `OPTWithALiBiDecoderLayer`

A decoder layer for the OPT model with ALiBi.

#### `OPTWithALiBiDecoder`

A full decoder stack for the OPT model with ALiBi.

#### `OPTWithALiBiModel`

The full OPT model with ALiBi support.

#### `OPTWithALiBiForCausalLM`

A class for causal language modeling with OPT and ALiBi.

#### `OPTWithAliBiForSequenceClassification`

A class for sequence classification with OPT and ALiBi.

### RnnLM

#### `RnnLMConfig`

Configuration class for the RNN language model. Inherits from `PretrainedConfig`.

- **Attributes:**
  - `vocab_size`: int, default=10000 - The size of the vocabulary.
  - `block_size`: int, default=128 - The maximum length of the input sequences.
  - `embedding_dim`: int, default=256 - The dimension of the embeddings.
  - `hidden_dim`: int, default=256 - The dimension of the hidden states.
  - `num_layers`: int, default=4 - The number of layers.
  - `cell_type`: str, default="lstm" - The type of RNN cell.
  - `add_bias`: bool, default=True - Whether to add a bias term.
  - `embedding_dropout`: float, default=0.1 - The dropout rate for embeddings.
  - `dropout`: float, default=0.1 - The dropout rate.

#### `RnnLM`

A Recurrent Neural Network language model.

- **Attributes:**
  - `config_class`: `RnnLMConfig` - Configuration class for the model.
  - `wte`: `nn.Embedding` - The embedding layer.
  - `encoder`: `nn.Module` - The encoder module, could be RNN, GRU, or LSTM.
  - `lm_head`: `nn.Linear` - The output layer.

- **Methods:**
  - `_init_hidden()`: Initializes the hidden states.
  - `_expand_hidden(batch_size)`: Expands the initial hidden state to match the batch size.
  - `forward(input_ids, labels=None, hidden_state=None, pad_id=-100, reduction="mean", return_dict=True, **kwargs)`: Computes the forward pass.

## Tokenization

### `CharTokenizationFunction`

An abstract base class for character-level tokenization functions.

- **Attributes:**
  - `name`: str - The name of the tokenization function.

- **Methods:**
  - `__call__(text: str) -> List[str]`: Tokenizes the input text.
  - `get_config() -> Dict[str, str]`: Returns the configuration of the tokenization function.
  - `from_config(config: Dict[str, str]) -> "CharTokenizationFunction"`: Creates an instance from a configuration.

### `RegexTokenizationFunction`

A tokenization function using regular expressions.

- **Attributes:**
  - `pattern`: `re.Pattern` - The compiled regex pattern.

- **Methods:**
  - `__call__(text: str) -> List[str]`: Tokenizes the input text using the regex pattern.
  - `get_config() -> Dict[str, str]`: Returns the regex pattern as configuration.

### `IpaTokenizationFunction`

A tokenization function using IPA (International Phonetic Alphabet).

- **Methods:**
  - `__call__(text: str) -> List[str]`: Tokenizes the input text using IPA.
  - `get_config() -> Dict[str, str]`: Returns an empty configuration.

### `CharLevelTokenizer`

A character-level tokenizer.

- **Attributes:**
  - `vocab_files_names`: `Dict[str, str]` - The vocabulary files names.
  - `_vocab_int_to_str`: `Dict[int, str]` - Mapping from token IDs to strings.
  - `_vocab_str_to_int`: `Dict[str, int]` - Mapping from strings to token IDs.

- **Methods:**
  - `vocab_size -> int`: Returns the size of the vocabulary.
  - `_tokenize(text: str) -> List[str]`: Tokenizes the input text.
  - `convert_tokens_to_ids(tokens: Union[str, List[str]]) -> int`: Converts tokens to their corresponding IDs.
  - `convert_ids_to_tokens(indices: Union[int, List[int]]) -> str`: Converts IDs to their corresponding tokens.
  - `tokens_to_string(tokens)`: Converts a list of tokens to a string.
  - `build_inputs_with_special_tokens(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]`: Builds inputs with special tokens.
  - `train(files)`: Trains the tokenizer on a list of files.
  - `encode_batch(input, add_special_tokens=False, padding='max_length')`: Encodes a batch of inputs.
  - `_add_items_to_encodings(encodings)`: Adds items to encodings.
  - `get_special_tokens_mask(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]`: Returns a mask for special tokens.
  - `create_token_type_ids_from_sequences(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]`: Creates token type IDs from sequences.
  - `save_vocabulary(save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]`: Saves the vocabulary.
  - `set_tokenization_function(tokenization_function: CharTokenizationFunction)`: Sets the tokenization function.

### `SaGeTokenizer`

A tokenizer derived from the HuggingFace `BertTokenizer`, specialized for the SaGe model.

- **Attributes:**
  - `vocab_files_names`: `Dict[str, str]` - The vocabulary files names.
  - `vocab`: `Dict[str, int]` - The vocabulary.
  - `ids_to_tokens`: `Dict[int, str]` - Mapping from token IDs to tokens.
  - `do_basic_tokenize`: bool - Whether to do basic tokenization.
  - `basic_tokenizer`: `BasicTokenizer` - The basic tokenizer.
  - `wordpiece_tokenizer`: `WordpieceTokenizer` - The WordPiece tokenizer.

- **Methods:**
  - `do_lower_case`: Property indicating if lowercasing is applied.
  - `vocab_size`: Property returning the size of the vocabulary.
  - `get_vocab() -> Dict[str, int]`: Returns the vocabulary.
  - `_tokenize(text: str)`: Tokenizes the input text.
  - `_convert_token_to_id(token: str) -> int`: Converts a token to an ID.
  - `_convert_id_to_token(index: int) -> str`: Converts an ID to a token.
  - `convert_tokens_to_string(tokens: List[str]) -> str`: Converts a list of tokens to a string.
  - `get_special_tokens_mask(token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]`: Returns a mask for special tokens.
  - `save_vocabulary(save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]`: Saves the vocabulary.

### `WordpieceTokenizer`

A tokenizer for WordPiece tokenization.

- **Attributes:**
  - `vocab`: `Dict[str, int]` - The vocabulary.
  - `unk_token`: str - The unknown token.
  - `max_input_chars_per_word`: int - The maximum number of input characters per word.

- **Methods:**
  - `tokenize(text: str) -> List[str]`: Tokenizes the input text into WordPiece tokens.

## Trainer

### `RnnLMTrainer`

A trainer class for RNN language models.

- **Methods:**
  - `compute_loss(model, inputs, return_outputs=False)`: Computes the loss for the model.

## Argument Classes

### BaseArgs

- **block_size** (`Optional[int]`): The input size of the model. Default is `None`.
- **auto_model_class** (`Literal["causal", "masked", "seq2seq"]`): Specifies which automodel class to use. Default is `"causal"`.

### TrainingInputArgs

- **input_files_path** (`str`): Path to the train, validation, and test files. Default files names are `train.txt`, `validation.txt`, and `test.txt`.
- **train_file_name** (`str`): Filename of the train data file. Default is `train.txt`.
- **validation_file_name** (`str`): Filename of the validation data file. Default is `validation.txt`.
- **test_file_name** (`str`): Filename of the test data file. Default is `test.txt`.
- **tokenizer_name_or_path** (`str`): Name of a HuggingFace tokenizer or path to a self-trained tokenizer.
- **model_type** (`str`): Specifies the type of AutoModel class to use.
- **model_name_or_path** (`Optional[str]`): Name of a HuggingFace model or path to a self-trained model.
- **config_name_or_path** (`Optional[str]`): Path to a model config file.
- **use_early_stopping** (`bool`): Whether to stop training if validation loss does not improve for a certain number of steps. Default is `False`.
- **early_stopping_patience** (`int`): Number of eval steps to wait before stopping training if using early stopping. Default is `1`.
- **lossy_context** (`bool`): Whether to break sequences randomly before concatenation. Default is `False`.
- **lossy_context_token** (`str`): Token used for sequence breaking with lossy context. Default is `<b>`.

### WandbArgs

- **wandb_group_name** (`str`): Name of the wandb group.
- **wandb_project_name** (`str`): Name of the wandb project.

### EvalArgs

- **input_files_path** (`str`): Path to the file with the evaluation data.
- **eval_file_name** (`Optional[str]`): Filename of the evaluation data file.
- **eval_string** (`Optional[str]`): JSON string with strings to be annotated for surprisal.
- **tokenizer_name_or_path** (`str`): Name of a HuggingFace tokenizer or path to a self-trained tokenizer.
- **model_type** (`str`): Specifies the type of AutoModel class to use.
- **model_name_or_path** (`str`): Name of a HuggingFace model or path to a self-trained model.
- **batch_size** (`int`): Batch size for evaluation. Default is `8`.
- **device** (`Optional[str]`): Device to use for evaluation. Default is `"cpu"`.
- **output_dir** (`str`): Path to save the evaluation results.
- **output_file_name** (`str`): Filename used to save evaluation results. Default is `eval_results.tsv`.
- **log_level** (`Literal["info", "warning", "error", "debug"]`): Log level for evaluation. Default is `"warning"`.
- **sum_subword_surprisal** (`bool`): Sum surprisals based on `--subword_prefix`. Default is `False`.
- **subword_prefix** (`str`): Prefix of words for subword tokenization. Default is `"Ġ"`.
- **prepend_token** (`bool`): Prepend an EOS token to each batch of sequences. Default is `False`.

### TokenizerArgs

- **input_files_path** (`str`): Path to the directory containing the training files for the tokenizer.
- **input_files_type** (`str`): File type of input files, e.g., '.txt'. Default is `"txt"`.
- **tokenizer_type** (`Literal["bpe", "unigram", "word-level"]`): Algorithm of the tokenizer. Default is `None`.
- **vocab_size** (`int`): Vocabulary size of the tokenizer. Default is `None`.
- **output_dir** (`str`): Where to save the tokenizer.
- **pad_token** (`str`): Padding token. Default is `<pad>`.
- **eos_token** (`str`): End of sequence token. Default is `</s>`.
- **unk_token** (`str`): Unknown token. Default is `<unk>`.
- **lossy_context** (`bool`): Whether to add the lossy context token to the tokenizer. Default is `False`.
- **lossy_context_token** (`str`): Sequence breaking token used with lossy context. Default is `<b>`.

### CoNLLUArgs

- **annotation_style** (`Literal["misc", "column"]`): How annotation should take place. Default is `"misc"`.
- **tag** (`str`): Tag for the surprisal column.

## Utility Functions

### `get_args(parser_type: str) -> List[Type[dataclass]]`
Returns a list of argument data classes based on the parser type.

### `get_automodel(model_type: str) -> Type[AutoModelForCausalLM | AutoModelForMaskedLM | AutoModelForSeq2SeqLM]`
Returns the appropriate AutoModel class based on the model type.

### `repackage_hidden(hidden_state: torch.Tensor) -> torch.Tensor`
Detaches hidden states from their history to avoid memory issues.

### `get_num_occurrences_in_tensor(value: int, t: torch.Tensor) -> int`
Returns the number of times a value occurs in a tensor.

### `tokenize_function(sequences: dict, tokenizer: AutoTokenizer, model_type: str) -> dict`
Tokenizes the given sequences.

### `prefix_eos_token(batch: Dict, eos_token_id: int) -> Dict`
Prefixes each batch of sequences with an EOS token.

### `preprocess_function_eval(sequences: dict, tokenizer: AutoTokenizer, model_max_length: int, stride: int, prefix_eos_token: bool = False) -> dict`
Processes sequences for evaluation.

### `preprocess_function(sequences: dict, tokenizer: AutoTokenizer, model_max_length: int, stride: int) -> dict`
Processes sequences with optional padding and truncation.

### `preprocess_function_sliding(sequences: dict, tokenizer: AutoTokenizer, T: int) -> dict`
Processes sequences using a sliding window approach.

### `compute_batch_surprisal(batch_input_ids: torch.Tensor, batch_mask: torch.Tensor, batch_logits: torch.Tensor, sequence_ids: list, tokenizer: AutoTokenizer) -> dict`
Computes the surprisal for each token in the batch.

### `compute_cloze_surprisal(input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor`
Computes the surprisal for a cloze task.

### `load_conllu_file(path: Union[os.PathLike, str]) -> Tuple[List[str], List[str], List[pd.DataFrame]]`
Loads a CoNLL-U formatted file.

### `save_conllu_file(path: Union[os.PathLike, str], dfs: List[pd.DataFrame]) -> None`
Saves data frames in CoNLL-U format.

### `get_word_surprisal(surprisal: List[float], tokens: List[str], words: List[str], tokenizer: AutoTokenizer, subword_prefix: str) -> List[float]`
Calculates word-level surprisal.

### `apply_lossy_context(sequences: dict, lossy_context_token: str, tokenizer: AutoTokenizer) -> dict`
Applies a lossy context to the sequences.

### `set_seed(seed: int = 123) -> None`
Sets the seed for reproducibility.

### `get_timestamp() -> str`
Returns the current timestamp.

### `create_dir(path_prefix: str, dir_name: str) -> str`
Creates a directory and returns its path.

## Additional Classes

### `ArgumentParserFactory`

- **Methods**
  - `get_argparser(cls, parser_type: str)`: Returns the argument parser for the specified type.

### `LMFactory`

- **Methods**
  - `from_pretrained(cls, model_type: str, model_name_or_path: Optional[str] = None)`: Loads a model from a pre-trained model or path.
  - `from_config(cls, model_type: str, config_name_or_path: Optional[str] = None)`: Loads a model from a configuration file.

### `TrainerFactory`

- **Methods**
  - `get_trainer(cls, model_type: str, **trainer_args)`: Returns a trainer instance based on the model type.
