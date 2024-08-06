from languagemodels.tokenization.sage_tokenizer.sage_tokenizer import SaGeTokenizer

# register auto classes
from transformers import AutoTokenizer
AutoTokenizer.register("sage", SaGeTokenizer)
