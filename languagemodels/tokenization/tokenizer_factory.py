from tokenizers import Tokenizer

from languagemodels.tokenization import CharLevelTokenizer


TOKENIZERS = {
    "char-tokenizer": CharLevelTokenizer,
    "pretrained-tokenizer": Tokenizer
}

class TokenizerFactory():

    @classmethod
    def get_tokenizer(cls, tokenizer_type, tokenizer_name=None, tokenizer_path=None, \
        **kwargs) -> Tokenizer:
        
        assert tokenizer_type in TOKENIZERS
        
        tokenizer_cls = TOKENIZERS[tokenizer_type]

        if tokenizer_name is not None:
            tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
        else:
            assert ".json" in tokenizer_path
            tokenizer = tokenizer_cls.from_file(tokenizer_path)

        assert tokenizer

        return tokenizer
