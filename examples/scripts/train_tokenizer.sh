DATA_DIR="/datasets/wikitext-103-raw"
OUTPUT_DIR="/languagemodels/out"

train_tokenizer \
    --input_files_dir $DATA_DIR/ \
    --input_files_type txt \
    --vocab_size 30000 \
    --output_dir $DATA_DIR \
    --block_size 64