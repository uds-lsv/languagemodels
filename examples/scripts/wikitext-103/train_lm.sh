pip install -e . --user

source examples/scripts/setup.sh

PROJECT_DIR="/nethome/jsteuer/git/lsv/languagemodels"
DATA_DIR="/data/corpora/wikitext-103-v1/"
TOKENIZER_DIR="./tokenizers"
CONFIG_DIR="./examples/configs/wikitext-103"
OUTPUT_DIR="./models/wikitext-103/"

python $PROJECT_DIR/train_lm.py \
    --input_files_path $DATA_DIR \
    --train_file_name train.txt \
    --validation_file_name validation.txt \
    --tokenizer_name_or_path $TOKENIZER_DIR/wikitext-103/bpe_10000/ \
    --model_type "rnn-lm" \
    --config_name_or_path $CONFIG_DIR/model_config.json \
    --output_dir $OUTPUT_DIR \
    --run_name new_train_script \
    --wandb_group_name wikitext-103 \
    --wandb_project_name languagemodels \
    --learning_rate 1e-3 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 128 \
    --warmup_ratio 0.1 \
    --evaluation_strategy steps \
    --greater_is_better False \
    --load_best_model_at_end \
    --metric_for_best_model eval_loss \
    --log_level info \
    --do_train \
    --block_size 128 \
    --save_total_limit 5 \
    --use_early_stopping True \
    --early_stopping_patience 3