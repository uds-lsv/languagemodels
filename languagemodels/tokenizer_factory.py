from tokenizers import Tokenizer

from languagemodels.tokenization import CharacterBasedTokenizer


TOKENIZERS = {
    "char-tokenizer": CharacterBasedTokenizer,
    "pretrained-tokenizer": Tokenizer
}

class TokenizerFactory():

    @classmethod
    def get_tokenizer(cls, tokenizer_type="pretrained-tokenizer", tokenizer_name=None, tokenizer_path=None, pre_trained=False, \
        **kwargs) -> Tokenizer:
        
        tokenizer_cls = TOKENIZERS[tokenizer_type]

        if tokenizer_name is not None:
            tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
        else:
            assert ".json" in tokenizer_path
            tokenizer = tokenizer_cls.from_file(tokenizer_path)

        assert tokenizer

        return tokenizer
