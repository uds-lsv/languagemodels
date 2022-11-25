import os
from pathlib import Path
import pytest
import shutil

from languagemodels import TokenizerFactory
from languagemodels.tokenization import (
    RegexTokenizationFunction,
    IpaTokenizationFunction,
    CharLevelTokenizer
)

MODEL_MAXLEN = 15

@pytest.fixture(scope='session')
def base_path():
    return os.path.dirname(__file__)

@pytest.fixture(scope='session')
def tokenizer_save_path(base_path):
    path = os.path.join(base_path, "tmp/saved_tokenizers")
    if not os.path.exists(path):
        os.makedirs(path)
    return  os.path.join(path, "test_tokenizer.json")

@pytest.fixture(scope='session')
def train_data_path_regex(base_path):
    path = os.path.join(base_path, "data/train_data_regex_tokenizer")
    assert os.path.exists(path)
    return path

@pytest.fixture(scope='session')
def train_data_path_ipa(base_path):
    path = os.path.join(base_path, "data/train_data_ipa_tokenizer")
    assert os.path.exists(path)
    return path

@pytest.fixture
def regex_tok_function():
    return RegexTokenizationFunction(r"\w'?")

@pytest.fixture
def ipa_tok_function():
    return IpaTokenizationFunction()

@pytest.fixture
def regex_tokenizer(regex_tok_function, train_data_path_regex):
    tokenizer = CharLevelTokenizer(model_max_length=15, \
        tokenization_function=regex_tok_function)
    tokenizer.train(train_data_path_regex)
    return tokenizer

@pytest.fixture
def ipa_tokenizer(ipa_tok_function, train_data_path_ipa):
    tokenizer = CharLevelTokenizer(model_max_length=15, \
        tokenization_function=ipa_tok_function)
    tokenizer.train(train_data_path_ipa)
    return tokenizer

def test_regex_tokenizer(regex_tokenizer):
    text = ["kaa", "see", "see", "tahab", "rohkem", "sooea", "saada", "see", "ei", "tule"]
    tokenized = regex_tokenizer.encode_batch(text)
    for output in tokenized["input_ids"]:
        assert len(output) == MODEL_MAXLEN

def test_save_load_ipa_tokenizer(ipa_tokenizer, tokenizer_save_path):
    text = ["ˈnoɹθ", "ˌwɪnd", "wəz", "əˈblaɪʒ", "tɪ", "kənˈfɛs", "ðət"]
    tokenized_pre = ipa_tokenizer.encode_batch(text)
    ipa_tokenizer.save(tokenizer_save_path)
    new_ipa_tokenizer = CharLevelTokenizer.from_file(tokenizer_save_path)
    tokenized_post = new_ipa_tokenizer.encode_batch(text)
    for tok_pre, tok_post in zip(tokenized_pre["input_ids"], tokenized_post["input_ids"]):
        assert tok_pre == tok_post
    for tok_pre, tok_post in zip(tokenized_pre["attention_mask"], tokenized_post["attention_mask"]):
        assert tok_pre == tok_post

def test_save_load_regex_tokenizer(regex_tokenizer, tokenizer_save_path):
    text = ["ol'l'i", "polkkov'n'ikku"]
    tokenized_pre = regex_tokenizer.encode_batch(text)
    regex_tokenizer.save(tokenizer_save_path)
    new_regex_tokenizer = CharLevelTokenizer.from_file(tokenizer_save_path)
    tokenized_post = new_regex_tokenizer.encode_batch(text)
    for tok_pre, tok_post in zip(tokenized_pre["input_ids"], tokenized_post["input_ids"]):
        assert tok_pre == tok_post
    for tok_pre, tok_post in zip(tokenized_pre["attention_mask"], tokenized_post["attention_mask"]):
        assert tok_pre == tok_post


def test_tokenizer_factory(regex_tokenizer, ipa_tokenizer, tokenizer_save_path):
    # regex tokenizer
    chars_pre = regex_tokenizer.characters
    regex_tokenizer.save(tokenizer_save_path)
    new_regex_tokenizer = TokenizerFactory.get_tokenizer(tokenizer_type='char-tokenizer', \
        tokenizer_path=tokenizer_save_path)
    chars_post = new_regex_tokenizer.characters
    assert chars_post == chars_pre

    chars_pre = ipa_tokenizer.characters
    ipa_tokenizer.save(tokenizer_save_path)
    new_ipa_tokenizer = TokenizerFactory.get_tokenizer(tokenizer_type='char-tokenizer', \
        tokenizer_path=tokenizer_save_path)
    chars_post = new_ipa_tokenizer.characters
    assert chars_post == chars_pre


    #cleanup
    shutil.rmtree(Path("tests/tmp/"))
