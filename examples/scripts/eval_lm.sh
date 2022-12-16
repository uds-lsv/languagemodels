export CUDA_VISIBLE_DEVICES=0

DATA_DIR="/languagemodels/examples/datasets/wikitext-103-raw"
OUTPUT_DIR="/languagemodels/eval_results/wikitext-103-raw"
SAVED_MODELS_DIR="/languagemodels/trained_models"

eval_lm \
    --test_file $DATA_DIR/wiki.test.txt \
    --model_dir $SAVED_MODELS_DIR/wikitext_model \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --batch_size 32 \
    --save_word_ids