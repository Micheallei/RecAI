RAW_DATA_DIR="$HOME/RecAI/RecLM-emb/data/steam/raw_data"
EXE_DIR="$HOME/RecAI/RecLM-emb"
OUT_DIR="$EXE_DIR/data/steam/train"
CACHE_DIR="$EXE_DIR/data/steam/cache"
MODEL_PATH_OR_NAME="xxxx"
SEED=2023
QUERY_MAX_LEN=512
PASSAGE_MAX_LEN=280
SENTENCE_POOLING_METHOD="mean"

cd $EXE_DIR
CONFIG_FILE=./shell/infer.yaml

accelerate launch --config_file $CONFIG_FILE infer_selection.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --exist_user_prompt_path $OUT_DIR/user2item.jsonl \
    --user_embedding_prompt_path $CACHE_DIR/candidate_u2i.jsonl \
    --answer_file $OUT_DIR/selected_user2item.jsonl \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 1024 \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized \
    --has_template