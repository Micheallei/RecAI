EXE_DIR="$HOME/RecAI/RecLM-emb"

cd $EXE_DIR

export OPENAI_API_VERSION="xxx";
export OPENAI_API_BASE="xxx"

python preprocess/rewrite_item.py \
    --in_meta_data $EXE_DIR/data/steam/raw_data/metadata.json \
    --out_meta_data $EXE_DIR/data/steam/raw_data/rewrite_meta_text.json \
    --model_name_or_path gpt-4o \
    --start_idx 0 \
    --end_idx 3000 \