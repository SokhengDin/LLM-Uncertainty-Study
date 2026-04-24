#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SAMPLES=100
PROMPT="base"
ICL="icl1"
ALPHA=0.1
OUT_DIR="outputs_kaggle"

DATASETS=(
    "mmlu_10k"
    "cosmosqa_10k"
    "hellaswag_10k"
    "halu_dialogue"
    "halu_summarization"
)

MODELS=(
    "qwen3-0.6b"
    "qwen2.5-3b"
    "qwen2.5-7b"
    "olmo-1b"
    "olmo-7b"
)

echo "============================================================"
echo "  Benchmark: ${#MODELS[@]} models | ${#DATASETS[@]} tasks | n=${SAMPLES}"
echo "  Logits:  ${OUT_DIR}/"
echo "  Figures: figures/"
echo "============================================================"

# ── Step 1: CP evaluation on existing logits 
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "  [evaluate] model=${MODEL}"
    python main.py \
        --model           "$MODEL" \
        --data_names      "${DATASETS[@]}" \
        --prompt_methods  "$PROMPT" \
        --icl_methods     "$ICL" \
        --max_samples     "$SAMPLES" \
        --alpha           "$ALPHA" \
        --logits_data_dir "$OUT_DIR" \
        --output_dir      "$OUT_DIR"
done

# ── Step 2: ECE calculation ───────────────────────────────────
echo ""
echo "  Computing ECE..."
python utils/ece.py \
    --out_dir  "$OUT_DIR" \
    --data_dir data \
    --fig_dir  figures \
    --samples  "$SAMPLES"

# ── Step 3: Figures ───────────────────────────────────────────
echo ""
echo "  Generating figures..."
python utils/plot_results.py \
    --samples     "$SAMPLES" \
    --prompt      "$PROMPT" \
    --icl         "$ICL" \
    --results_dir "$OUT_DIR" \
    --figures_dir figures

echo ""
echo "============================================================"
echo "  Done. Results in ${OUT_DIR}/  Figures in figures/"
echo "============================================================"
