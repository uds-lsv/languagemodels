export CUDA_VISIBLE_DEVICES=3

export PYTHONPATH=$PYTHONPATH:/nethome/mmosbach/projects/languagemodels

DATA_DIR="/nethome/mmosbach/projects/languagemodels/examples/datasets/wikitext-103-raw"

python /nethome/mmosbach/projects/languagemodels/languagemodels/train_lm.py \
    --train_file $DATA_DIR/wiki.valid.txt \
    --validation $DATA_DIR/wiki.valid.txt \
    --model_type bigram-lm \
    --tokenizer_path $DATA_DIR/tokenizer-wiki.json \
    --learning_rate 1e-2 \
    --batch_size 16 \
    --max_steps 100 \
    --device cuda \
    --seed 123