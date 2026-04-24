import json
import pickle
import random
import numpy as np

OPTIONS = ["A", "B", "C", "D", "E", "F"]


def _softmax(x):
    e = np.exp(np.array(x, dtype=float) - np.max(x))
    return e / e.sum()



def load_test_split(short, ds, out_dir, data_dir, samples=100, seed=42):
    import os
    pkl_path  = os.path.join(out_dir, f"{short}_{ds}_base_icl1_sample{samples}.pkl")
    data_path = os.path.join(data_dir, f"{ds}.json")
    if not os.path.exists(pkl_path) or not os.path.exists(data_path):
        return None, None

    logits_all = pickle.load(open(pkl_path, "rb"))
    data_all   = json.load(open(data_path))


    rest = data_all[10:]
    random.seed(seed)
    if len(rest) > samples:
        rest = random.sample(rest, samples)
    demo_ids = {1, 3, 5, 7, 9}
    rest = [d for d in rest if d["id"] not in demo_ids]

    logits_by_id = {str(r["id"]): r for r in logits_all}
    paired = [(d, logits_by_id[str(d["id"])]) for d in rest if str(d["id"]) in logits_by_id]

    random.seed(seed)
    random.shuffle(paired)
    n_cal = len(paired) // 2
    test_data, test_logits = zip(*paired[n_cal:])
    return list(test_data), list(test_logits)



def compute_ece(test_data, test_logits, n_bins=10):
    confs, corrects = [], []
    for item, row in zip(test_data, test_logits):
        probs = _softmax(row["logits_options"])
        confs.append(float(np.max(probs)))
        corrects.append(int(OPTIONS[int(np.argmax(probs))] == item["answer"]))

    confs    = np.array(confs)
    corrects = np.array(corrects)
    n        = len(confs)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins  = []
    ece = mce = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confs >= lo) & (confs <= hi)
        n_b  = int(mask.sum())
        if n_b == 0:
            bins.append({"lo": lo, "hi": hi, "n": 0,
                         "acc": None, "conf": None, "gap": None})
            continue
        acc_b  = float(corrects[mask].mean())
        conf_b = float(confs[mask].mean())
        gap    = abs(acc_b - conf_b)
        ece   += (n_b / n) * gap
        mce    = max(mce, gap)
        bins.append({"lo": lo, "hi": hi, "n": n_b,
                     "acc": acc_b, "conf": conf_b, "gap": gap})

    return {
        "ece":         round(float(ece), 6),
        "mce":         round(float(mce), 6),
        "n":           n,
        "avg_conf":    round(float(confs.mean()), 6),
        "acc":         round(float(corrects.mean()), 6),
        "bins":        bins,
        "confidences": confs.tolist(),
        "corrects":    corrects.tolist(),
    }



def compute_ece_all(models, datasets, out_dir, data_dir, samples=100, n_bins=10):
    results = {}
    for short in models:
        results[short] = {}
        for ds in datasets:
            td, tl = load_test_split(short, ds, out_dir, data_dir, samples)
            if td is None:
                continue
            results[short][ds] = compute_ece(td, tl, n_bins=n_bins)
    return results



def reliability_diagram(ece_result, ax=None, title=""):

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    bins   = [b for b in ece_result["bins"] if b["n"] > 0]
    mids   = [0.5 * (b["lo"] + b["hi"]) for b in bins]
    accs   = [b["acc"]  for b in bins]
    confs  = [b["conf"] for b in bins]
    w      = bins[0]["hi"] - bins[0]["lo"] if bins else 0.1

    ax.bar(mids, accs, width=w * 0.9, alpha=0.7, color="steelblue", label="Accuracy")
    ax.plot([0, 1], [0, 1], "r--", lw=1.2, label="Perfect calibration")
    ax.scatter(confs, accs, color="black", s=20, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max softmax)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={ece_result['ece']:.4f}")
    ax.legend(fontsize=8)
    return ax


if __name__ == "__main__":
    import argparse, json, os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    default="outputs_kaggle")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--fig_dir",    default="figures")
    parser.add_argument("--samples",    type=int, default=100)
    parser.add_argument("--n_bins",     type=int, default=10)
    args = parser.parse_args()

    models   = ["qwen3-0.6b", "qwen2.5-3b", "qwen2.5-7b", "olmo-1b", "olmo-7b"]
    datasets = ["mmlu_10k", "hellaswag_10k", "cosmosqa_10k", "halu_dialogue", "halu_summarization"]
    ds_labels = {
        "mmlu_10k": "MMLU", "hellaswag_10k": "HellaSwag", "cosmosqa_10k": "CosmosQA",
        "halu_dialogue": "HaluDial", "halu_summarization": "HaluSum",
    }
    os.makedirs(args.fig_dir, exist_ok=True)

    results = compute_ece_all(models, datasets, args.out_dir, args.data_dir, args.samples, args.n_bins)

    # ── Print table 
    print(f"\nECE Table (n_bins={args.n_bins})")
    print(f"{'Model':<14}", end="")
    for ds in datasets:
        print(f"  {ds_labels[ds]:>10}", end="")
    print()
    print("-" * (14 + 13 * len(datasets)))
    for m in models:
        print(f"{m:<14}", end="")
        for ds in datasets:
            v = results.get(m, {}).get(ds, {}).get("ece", None)
            print(f"  {'N/A':>10}" if v is None else f"  {v:>10.4f}", end="")
        print()

    # ── Heatmap 
    import numpy as np
    ece_mat = np.full((len(models), len(datasets)), np.nan)
    for i, m in enumerate(models):
        for j, ds in enumerate(datasets):
            v = results.get(m, {}).get(ds, {}).get("ece", None)
            if v is not None:
                ece_mat[i, j] = v

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(ece_mat, cmap="RdYlGn_r", vmin=0.1, vmax=0.55, aspect="auto")
    plt.colorbar(im, ax=ax, label="ECE")
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels([ds_labels[d] for d in datasets], rotation=15, ha="right")
    ax.set_yticks(range(len(models)));   ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(datasets)):
            if not np.isnan(ece_mat[i, j]):
                ax.text(j, i, f"{ece_mat[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("ECE Heatmap (5×5)")
    plt.tight_layout()
    out = os.path.join(args.fig_dir, "fig_ece_heatmap.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"\nSaved {out}")