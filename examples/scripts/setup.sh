# setup basic paths
export CACHE_BASE_DIR=/data/users/jsteuer/logs/languagemodels/logfiles/cache
export OUTPUT_DIR=/data/users/jsteuer/logs/languagemodels/logfiles

# setup wandb
export WANDB_DISABLED=false
export WANDB_API_KEY=2a0328272bb779eaba239722b3dff84531e7cea8
export WANDB_USERNAME=b4
export WANDB_ENTITY=b4
export WANDB_CACHE_DIR=$CACHE_BASE_DIR/wandb
export WANDB_CONFIG_DIR=$WANDB_CACHE_DIR

# set variables for transformers, datasets, evaluate
export TOKENIZERS_PARALLELISM=true
# export HF_DATASETS_CACHE=$CACHE_BASE_DIR/hf_datasets
# export HF_EVALUATE_CACHE=$CACHE_BASE_DIR/hf_evaluate
# export HF_MODULES_CACHE=$CACHE_BASE_DIR/hf_modules
# export HF_MODELS_CACHE=$CACHE_BASE_DIR/hf_lms

# create cash dirs if they don't exist yet
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR
# mkdir -p $HF_DATASETS_CACHE
# mkdir -p $HF_EVALUATE_CACHE
# mkdir -p $HF_MODULES_CACHE
# mkdir -p $HF_MODELS_CACHE

# set path to python dir
# export PYTHON_BIN="/opt/conda/bin"
# export PYTHONPATH=$PYTHONPATH:/languagemodels
