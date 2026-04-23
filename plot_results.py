#!/usr/bin/env python3
"""
Generate benchmark figures from all_results.json files.
Produces:
  figures/fig_by_model.pdf  — grouped by model, one bar group per task
  figures/fig_by_task.pdf   — grouped by task, one bar group per model
  figures/fig_heatmap.pdf   — heatmap of Acc across models x tasks
"""

import json
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUTPUTS_DIR = "outputs_base"
FIGURES_DIR = "figures"

# Overridden at runtime by --results_dir / --figures_dir args

MODELS_ORDER = [
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "gemma4:e4b",
    "llama3.1:latest",
]
MODEL_LABELS = {
    "qwen3.5:2b":      "Qwen3.5\n2B",
    "qwen3.5:4b":      "Qwen3.5\n4B",
    "qwen3.5:9b":      "Qwen3.5\n9B",
    "gemma4:e4b":      "Gemma4\nE4B",
    "llama3.1:latest": "Llama3.1\n8B",
}

DATASETS_ORDER = [
    "mmlu_10k",
    "cosmosqa_10k",
    "hellaswag_10k",
    "halu_dialogue",
    "halu_summarization",
]
DATASET_LABELS = {
    "mmlu_10k":          "QA\n(MMLU)",
    "cosmosqa_10k":      "RC\n(CosmosQA)",
    "hellaswag_10k":     "CI\n(HellaSwag)",
    "halu_dialogue":     "DRS\n(HaluDial)",
    "halu_summarization":"DS\n(HaluSum)",
}

COLORS = {
    "Acc": "#4472C4",
    "CR":  "#ED7D31",
    "SS":  "#70AD47",
}


def load_all(key):
    """Load results for all models. key = 'base_icl1' etc."""
    results = {}
    for m in MODELS_ORDER:
        path = os.path.join(OUTPUTS_DIR, f"{m}_all_results.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            results[m] = json.load(f)
    return results


def extract(results, model, dataset, key):
    try:
        d = results[model][dataset]
        acc = 100 * d["Acc"][key]
        cr  = 100 * np.mean([d["LAC_coverage"][key], d["APS_coverage"][key]])
        ss  =       np.mean([d["LAC_set_size"][key], d["APS_set_size"][key]])
        return acc, cr, ss
    except (KeyError, TypeError):
        return None, None, None


def fig_by_model(results, key, samples):
    """One figure per metric (Acc, CR, SS), x=models, hue=task."""
    n_models  = len(MODELS_ORDER)
    n_tasks   = len(DATASETS_ORDER)
    task_colors = plt.cm.tab10(np.linspace(0, 0.9, n_tasks))

    for metric_idx, (metric, ylabel, ylim) in enumerate([
        ("Acc", "Accuracy (%)", (0, 100)),
        ("CR",  "Coverage Rate (%)", (0, 110)),
        ("SS",  "Set Size", (0, 7)),
    ]):
        fig, ax = plt.subplots(figsize=(10, 4))
        w  = 0.14
        xs = np.arange(n_models)

        for ti, dataset in enumerate(DATASETS_ORDER):
            vals = []
            for model in MODELS_ORDER:
                acc, cr, ss = extract(results, model, dataset, key)
                v = {"Acc": acc, "CR": cr, "SS": ss}[metric]
                vals.append(v if v is not None else 0)

            offset = (ti - n_tasks / 2 + 0.5) * w
            bars = ax.bar(xs + offset, vals, w,
                          label=DATASET_LABELS[dataset].replace("\n", " "),
                          color=task_colors[ti], alpha=0.88)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                        f"{h:.0f}" if metric != "SS" else f"{h:.2f}",
                        ha="center", va="bottom", fontsize=5.5, rotation=90)

        if metric == "CR":
            ax.axhline(90, color="red", ls="--", lw=1.2, label="90% target")

        ax.set_xticks(xs)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS_ORDER], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_title(f"{metric} across models and tasks  (n={samples}, base, 1-shot, α=0.1)",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        plt.tight_layout()
        path = os.path.join(FIGURES_DIR, f"fig_{metric.lower()}_by_model.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")


def fig_by_task(results, key, samples):
    """Combined 3-metric bar chart, x=tasks, one panel per model."""
    n_models = len(MODELS_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(14, 4), sharey=False)

    for mi, model in enumerate(MODELS_ORDER):
        ax1 = axes[mi]
        ax2 = ax1.twinx()
        xs  = np.arange(len(DATASETS_ORDER))
        w   = 0.26

        accs = [extract(results, model, d, key)[0] or 0 for d in DATASETS_ORDER]
        crs  = [extract(results, model, d, key)[1] or 0 for d in DATASETS_ORDER]
        sss  = [extract(results, model, d, key)[2] or 0 for d in DATASETS_ORDER]

        ax1.bar(xs - w, accs, w, color=COLORS["Acc"], alpha=0.9, label="Acc (%)")
        ax1.bar(xs,     crs,  w, color=COLORS["CR"],  alpha=0.9, label="CR (%)")
        ax1.axhline(90, color="red", ls="--", lw=0.8)
        ax2.bar(xs + w, sss,  w, color=COLORS["SS"],  alpha=0.9, label="SS")

        ax1.set_ylim(0, 115)
        ax2.set_ylim(0, 8)
        ax1.set_xticks(xs)
        ax1.set_xticklabels([DATASET_LABELS[d].split("\n")[0] for d in DATASETS_ORDER],
                             fontsize=7, rotation=20, ha="right")
        ax1.set_title(MODEL_LABELS[model].replace("\n", " "), fontsize=9, fontweight="bold")
        if mi == 0:
            ax1.set_ylabel("Acc / CR (%)", fontsize=8)
        if mi == n_models - 1:
            ax2.set_ylabel("SS", fontsize=8)
        else:
            ax2.set_yticklabels([])

    # shared legend
    from matplotlib.patches import Patch
    legend_els = [
        Patch(color=COLORS["Acc"], label="Acc (%)"),
        Patch(color=COLORS["CR"],  label="CR (%)"),
        Patch(color=COLORS["SS"],  label="SS"),
    ]
    fig.legend(handles=legend_els, loc="upper center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Per-model results across 5 tasks  (n={samples}, base, 1-shot, α=0.1)",
                 fontsize=10, y=1.06)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig_by_task.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig_heatmap(results, key, metric, samples):
    """Heatmap: rows=models, cols=tasks."""
    data = np.zeros((len(MODELS_ORDER), len(DATASETS_ORDER)))
    for mi, model in enumerate(MODELS_ORDER):
        for di, dataset in enumerate(DATASETS_ORDER):
            acc, cr, ss = extract(results, model, dataset, key)
            data[mi, di] = {"Acc": acc, "CR": cr, "SS": ss}[metric] or 0

    fig, ax = plt.subplots(figsize=(8, 3.5))
    cmap = "YlGn" if metric in ("Acc", "CR") else "YlOrRd_r"
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=data[data > 0].min() * 0.95, vmax=data.max() * 1.02)
    plt.colorbar(im, ax=ax, fraction=0.03)

    ax.set_xticks(range(len(DATASETS_ORDER)))
    ax.set_xticklabels([DATASET_LABELS[d].replace("\n", " ") for d in DATASETS_ORDER], fontsize=9)
    ax.set_yticks(range(len(MODELS_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODELS_ORDER], fontsize=9)

    for mi in range(len(MODELS_ORDER)):
        for di in range(len(DATASETS_ORDER)):
            v = data[mi, di]
            fmt = f"{v:.0f}" if metric in ("Acc", "CR") else f"{v:.2f}"
            ax.text(di, mi, fmt, ha="center", va="center", fontsize=8,
                    color="black" if v < data.max() * 0.85 else "white")

    ax.set_title(f"{metric} heatmap — models × tasks  (n={samples}, base, 1-shot, α=0.1)",
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"fig_heatmap_{metric.lower()}.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def main():
    global OUTPUTS_DIR, FIGURES_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",     type=int, default=50)
    parser.add_argument("--prompt",      type=str, default="base")
    parser.add_argument("--icl",         type=str, default="icl1")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    args = parser.parse_args()

    if args.results_dir:
        OUTPUTS_DIR = args.results_dir
    if args.figures_dir:
        FIGURES_DIR = args.figures_dir

    key = f"{args.prompt}_{args.icl}"
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading results...")
    results = load_all(key)
    if not results:
        print("No results found. Run run_benchmark.sh first.")
        return

    print("Generating figures...")
    fig_by_model(results, key, args.samples)
    fig_by_task(results, key, args.samples)
    for metric in ("Acc", "CR", "SS"):
        fig_heatmap(results, key, metric, args.samples)

    print("\nDone! Files in figures/:")
    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith(".pdf"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
