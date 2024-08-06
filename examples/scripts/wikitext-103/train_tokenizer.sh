PROJECT_DIR="/nethome/jsteuer/git/lsv/languagemodels"
DATA_DIR="/data/corpora/wikitext-103-v1/"
OUTPUT_DIR="./tokenizers"

python $PROJECT_DIR/train_tokenizer.py \
    --input_files_path $DATA_DIR  \
    --input_files_type xt \
    --vocab_size 10000 \
    --output_dir $OUTPUT_DIR/wikitext-103/bpe_10000 \
    --tokenizer_type bpe