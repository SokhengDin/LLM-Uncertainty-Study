#!/usr/bin/env bash
# ============================================================
#  Multi-model, multi-task benchmark runner
#  Usage: bash run_benchmark.sh
#  n=50 → n_cal=27, quantile level=0.926 (non-trivial CP)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SAMPLES=50
PROMPT="base"
ICL="icl1"
ALPHA=0.1

# All 5 datasets
DATASETS=(
    "mmlu_10k"
    "cosmosqa_10k"
    "hellaswag_10k"
    "halu_dialogue"
    "halu_summarization"
)

# Models available in Ollama
MODELS=(
    "qwen3.5:2b"
    "qwen3.5:4b"
    "qwen3.5:9b"
    "gemma4:e4b"
    "llama3.1:latest"
)

echo "============================================================"
echo "  Benchmark: ${#MODELS[@]} models | ${#DATASETS[@]} tasks | n=${SAMPLES}"
echo "============================================================"

# ── Step 1: Generate logits for all models x all datasets ────
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  [generate] model=${MODEL}"
    echo "============================================================"
    for DATA in "${DATASETS[@]}"; do
        echo "  --> dataset=${DATA}"
        python utils/generate_logits.py \
            --model         "$MODEL" \
            --file          "${DATA}.json" \
            --prompt_method "$PROMPT" \
            --few_shot      1 \
            --max_samples   "$SAMPLES" \
            --output_dir    outputs_base
    done
done

echo ""
echo "============================================================"
echo "  All logits generated. Running CP evaluation..."
echo "============================================================"

# ── Step 2: Run CP evaluation for all models (all datasets at once) ──
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  [evaluate] model=${MODEL}"
    echo "------------------------------------------------------------"
    python main.py \
        --model          "$MODEL" \
        --data_names     "${DATASETS[@]}" \
        --prompt_methods "$PROMPT" \
        --icl_methods    "$ICL" \
        --max_samples    "$SAMPLES" \
        --alpha          "$ALPHA"
done

echo ""
echo "============================================================"
echo "  All done! Generating figures..."
echo "============================================================"

# ── Step 3: Generate figures ─────────────────────────────────
.venv/bin/python3.12 plot_results.py \
    --samples "$SAMPLES" \
    --prompt  "$PROMPT" \
    --icl     "$ICL"

echo ""
echo "============================================================"
echo "  Figures saved in figures/"
echo "============================================================"
