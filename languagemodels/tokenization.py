from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def train_word_level_tokenizer(vocab_size, files):
    """Train a word-level tokenizer on a list of files

    Args:
        vocab_size (int): vocabulary size of the tokenizer
        files (List[str]): list of files on which the tokenizer will be trained 

    Returns:
        Tokenizer: a tokenizer
    """
    tokenizer = Tokenizer(model=WordLevel(unk_token="<unk>"))

    # these are default and can be overwritten
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>"]
    )

    # TODO(mm): look into Normalizers and additional Pre-tokenizers and provide good defaults

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[(t, tokenizer.token_to_id(t)) for t in ["<s>", "</s>"]]
    )

    return tokenizer
