"""
Expected Calibration Error (ECE) for MCQA with softmax logits.

ECE measures how well model confidence (max softmax probability) matches
empirical accuracy. A perfectly calibrated model satisfies:
    P(correct | confidence = p) = p  for all p.

Standard equal-width bin estimator (Guo et al., ICML 2017):
    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

The split logic (load_test_split) replicates exactly the cal/test partition
used in kaggle_benchmark.ipynb so ECE is computed on the same n=50 test
samples that produced the Acc/SS/CR results.

References:
    Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
"""

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
