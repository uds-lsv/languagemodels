export CUDA_VISIBLE_DEVICES=3

export PYTHONPATH=$PYTHONPATH:/nethome/mmosbach/projects/languagemodels

DATA_DIR="/nethome/mmosbach/projects/languagemodels/examples/datasets/wikitext-103-raw"
CONFIG_PATH="/nethome/mmosbach/projects/languagemodels/examples/configs/gru/basic_gru.json"

python /nethome/mmosbach/projects/languagemodels/languagemodels/train_lm.py \
    --train_file $DATA_DIR/wiki.valid.txt \
    --validation $DATA_DIR/wiki.valid.txt \
    --model_type rnn-lm \
    --tokenizer_path $DATA_DIR/tokenizer-wiki.json \
    --config_name_or_path $CONFIG_PATH \
    --learning_rate 1e-3 \
    --batch_size 16 \
    --max_steps 100000 \
    --device cuda \
    --seed 123