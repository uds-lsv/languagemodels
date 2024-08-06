from languagemodels import TokenizerFactory


def test_tokenizer_factory_tok_name():
    tokenizer_name = 'bert-base-uncased'
    tokenizer = TokenizerFactory.get_tokenizer(
        tokenizer_type='pretrained-tokenizer',
        tokenizer_name=tokenizer_name)
    encoded = tokenizer.encode("To be or not to be, that is the question")
    assert encoded.ids == [101, 2000, 2022, 2030, 2025, 2000, 2022, 1010, 2008, 2003, 1996, 3160, 102]
    assert encoded.tokens[1:-1] == ['to', 'be', 'or', 'not', 'to', 'be', ',', 'that', 'is', 'the', 'question']