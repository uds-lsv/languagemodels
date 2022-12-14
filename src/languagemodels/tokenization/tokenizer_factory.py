from tokenizers import Tokenizer

from languagemodels.tokenization import (
    CharLevelTokenizer, 
    RegexTokenizationFunction, 
    IpaTokenizationFunction
)

TOKENIZERS = {
    "char-tokenizer": (CharLevelTokenizer, RegexTokenizationFunction(r"\w")),
    "ged-tokenizer": (CharLevelTokenizer, RegexTokenizationFunction(r"\w'?")),
    "ipa-tokenizer": (CharLevelTokenizer, IpaTokenizationFunction()),
    "pretrained-tokenizer": (Tokenizer, None)
}

class TokenizerFactory():

    @classmethod
    def get_tokenizer(cls, tokenizer_type, tokenizer_name=None, tokenizer_name_or_path=None, \
        **kwargs) -> Tokenizer:

        assert tokenizer_type in TOKENIZERS

        tokenizer_cls, tokenization_fun = TOKENIZERS[tokenizer_type]

        if tokenizer_name is not None:
            tokenizer = tokenizer_cls.from_pretrained(tokenizer_name)
        elif tokenizer_name_or_path is not None:
            # Tokenizers extending PreTrainedTokenizer only 
            # have the from_pretrained method. 
            if tokenizer_cls == CharLevelTokenizer:
                tokenizer = tokenizer_cls.from_pretrained(tokenizer_name_or_path)
                tokenizer.set_tokenization_function(tokenization_fun)
            # Tokenizers from the tokenizers library have a 
            # their own method to load from a file
            elif tokenizer_cls == Tokenizer:
                tokenizer = tokenizer_cls.from_file(tokenizer_name_or_path)
            else:
                raise ValueError(f"Tokenizer {tokenizer_type} cannot be mapped to an appropriate class!")

        assert tokenizer

        return tokenizer
