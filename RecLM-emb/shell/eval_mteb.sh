EXE_DIR="$HOME/RecAI/RecLM-emb"

cd $EXE_DIR

python utils/eval_mteb.py \
    --model_names_or_path "$EXE_DIR/output/merged_model" \
    --output_path $EXE_DIR/output/mteb_results \
    --sentence_pooling_method mean \
    --normlized  \
    --has_template