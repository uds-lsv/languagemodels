from pathlib import Path
import pytest
import shutil

from languagemodels.tokenization import (
    RegexTokenizationFunction,
    IpaTokenizationFunction,
    CharacterBasedTokenizer
)

MODEL_MAXLEN = 15

@pytest.fixture
def tokenizer_save_path(scope='session'):
    path = Path("tests/tmp/saved_tokenizers")
    if not path.exists():
        path.mkdir(parents=True)
    return  path / "test_tokenizer.json"

@pytest.fixture
def train_data_path_regex(scope='session'):
    path = Path("tests/data/train_data_regex_tokenizer")
    assert path.exists()
    return path

@pytest.fixture
def train_data_path_ipa(scope='session'):
    path = Path("tests/data/train_data_ipa_tokenizer")
    assert path.exists()
    return path

@pytest.fixture
def regex_tok_function():
    return RegexTokenizationFunction(r"\w'?")

@pytest.fixture
def ipa_tok_function():
    return IpaTokenizationFunction()

@pytest.fixture
def regex_tokenizer(regex_tok_function, train_data_path_regex):
    tokenizer = CharacterBasedTokenizer(model_max_length=15, \
        tokenization_function=regex_tok_function)
    tokenizer.train(train_data_path_regex)
    return tokenizer

@pytest.fixture
def ipa_tokenizer(ipa_tok_function, train_data_path_ipa):
    tokenizer = CharacterBasedTokenizer(model_max_length=15, \
        tokenization_function=ipa_tok_function)
    tokenizer.train(train_data_path_ipa)
    print(tokenizer.characters)
    return tokenizer

@pytest.mark.parametrize("text, expected_mask", [
    (
        "ol'l'i", 
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ),
    (
        "polkkov'n'ikku",
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    )
    
])
def test_regex_tokenizer(regex_tokenizer, text, expected_mask):
    tokenized = regex_tokenizer.encode_batch([text])
    for output in tokenized:
        assert len(output.input_ids) == MODEL_MAXLEN
        assert output.attention_mask == expected_mask

@pytest.mark.parametrize("text", [
    "ˈnoɹθ", "ˌwɪnd", "wəz", "əˈblaɪʒ", "tɪ", "kənˈfɛs", "ðət"
])
def test_save_load_ipa_tokenizer(ipa_tokenizer, tokenizer_save_path, text):
    tokenized_pre = ipa_tokenizer.encode_batch([text])
    ipa_tokenizer.save(tokenizer_save_path)
    del ipa_tokenizer
    ipa_tokenizer = CharacterBasedTokenizer.from_file(tokenizer_save_path)
    tokenized_post = ipa_tokenizer.encode_batch([text])
    for tok_pre, tok_post in zip(tokenized_pre, tokenized_post):
        assert tok_pre.input_ids == tok_post.input_ids
        assert tok_pre.attention_mask == tok_post.attention_mask

@pytest.mark.parametrize("text", [
    "ol'l'i", "polkkov'n'ikku"
])
def test_save_load_regex_tokenizer(regex_tokenizer, tokenizer_save_path, text):
    tokenized_pre = regex_tokenizer.encode_batch([text])
    regex_tokenizer.save(tokenizer_save_path)
    del regex_tokenizer
    regex_tokenizer = CharacterBasedTokenizer.from_file(tokenizer_save_path)
    tokenized_post = regex_tokenizer.encode_batch([text])
    for tok_pre, tok_post in zip(tokenized_pre, tokenized_post):
        assert tok_pre.input_ids == tok_post.input_ids
        assert tok_pre.attention_mask == tok_post.attention_mask

    #cleanup
    shutil.rmtree(Path("tests/tmp/"))
