export CUDA_VISIBLE_DEVICES=0

source /languagemodels/examples/scripts/setup.sh


DATA_DIR="/datasets/wikitext-103-raw"
CONFIG_PATH="/languagemodels/examples/configs/gru/basic_gru.json"
OUTPUT_DIR="/logfiles"

train_lm \
    --wandb_project_name languagemodels \
    --train_file $DATA_DIR/wiki.valid.txt \
    --validation $DATA_DIR/wiki.valid.txt \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --tokenizer_name bert-base-uncased \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --max_steps 10000 \
    --logging_steps 100 \
    --device cuda \
    --output_dir $OUTPUT_DIR \
    --seed 123