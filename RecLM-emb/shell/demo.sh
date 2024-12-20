# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

RAW_DATA_DIR=$HOME/RecAI/RecLM-emb/data/steam/raw_data
EXE_DIR=$HOME/RecAI/RecLM-emb

cd $EXE_DIR

# export OPENAI_API_VERSION="xxx";
# export OPENAI_API_BASE="xxx"

accelerate launch --config_file ./shell/infer_case.yaml demo.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --user_embedding_prompt_path $EXE_DIR/output/demo/user_embedding_prompt.jsonl \
    --model_path_or_name "xxxx" \
    --seed 2024 \
    --query_max_len 512 \
    --passage_max_len 280 \
    --per_device_eval_batch_size 1 \
    --sentence_pooling_method "mean" \
    --normlized \
    --has_template