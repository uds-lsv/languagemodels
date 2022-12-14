DATA_DIR="/datasets/wikitext-103-raw"
CONFIG_PATH="/languagemodels/examples/configs/gru/basic_gru.json"
OUTPUT_DIR="/logfiles"

train_tokenizer \
    --input_files_dir $DATA_DIR/ \
    --input_files_type txt \
    --vocab_size 30000 \
    --output_dir $DATA_DIR