pip install -e . --user

PROJECT_DIR="/nethome/jsteuer/git/lsv/languagemodels"
DATA_DIR="/data/corpora/wikitext-103-v1/"
MODEL_DIR="./trained_models/wikitext-103/"
TOKENIZER_DIR="./tokenizers"

python $PROJECT_DIR/eval_lm.py \
    --input_files_path $DATA_DIR \
    --eval_file_name test.txt \
    --tokenizer_name_or_path $TOKENIZER_DIR/wikitext-103/bpe_10000 \
    --model_name_or_path $MODEL_DIR \
    --output_dir $MODEL_DIR/results \
    --batch_size 8 \
    --device cpu