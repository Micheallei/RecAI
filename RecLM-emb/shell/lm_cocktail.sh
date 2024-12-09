EXE_DIR="$HOME/RecAI/RecLM-emb"

cd $EXE_DIR

python utils/lm_cocktail.py \
    --model_names_or_paths "xxx,xxx" \
    --model_type encoder \
    --weights 0.7,0.3 \
    --output_path $EXE_DIR/output/merged_model \
    --sentence_pooling_method mean \
    --normlized