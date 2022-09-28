from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


def train_word_level_tokenizer(vocab_size, files):
    tokenizer = Tokenizer(model=WordLevel(unk_token="<unk>"))

    # these are default and can be overwritten
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>"]
    )

    # TODO(mm): look into Normaliuzers and additional Pre-tokenizers to provide good defaults

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[(t, tokenizer.token_to_id(t)) for t in ["<s>", "</s>"]]
    )

    return tokenizer
