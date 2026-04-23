"""
Mondrian Conformal Prediction stratified by entropy bins.

Matches the interface of LAC_CP / APS_CP in conformal_prediction.py so it
plugs into the existing main.py / evaluate pipeline without changes.

Key idea: instead of one global q̂, compute a separate q̂_b per difficulty
bin (defined by entropy quantiles). This gives conditional coverage per bin
rather than only marginal coverage across all questions.

Reference: Vovk et al. (2005), "Algorithmic Learning in a Random World", §3.
"""

import numpy as np
from .conformal_prediction import softmax, OPTIONS


# ── Score helpers 

def _lac_score(probs, true_idx):
    """LAC nonconformity score: 1 - p(true label)."""
    return 1.0 - probs[true_idx]


def _aps_score(probs, true_idx):
    """APS nonconformity score: cumulative prob down to true label in rank order."""
    pi = np.argsort(probs)[::-1]
    cum = np.take_along_axis(probs, pi, axis=0).cumsum()
    cum_r = np.take_along_axis(cum, pi.argsort(), axis=0)
    return cum_r[true_idx]


# ── Entropy

def _entropy(probs):
    """Shannon entropy H = -∑ p log p (nats). Clips to avoid log(0)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p))


# ── Bin assignment 

def _assign_bin(H, boundaries):
    """Return bin index b such that H ∈ [boundaries[b], boundaries[b+1])."""
    B = len(boundaries) - 1
    for b in range(B - 1):
        if H < boundaries[b + 1]:
            return b
    return B - 1  # last bin catches everything ≥ h_{B-1}


# ── Phase 1: Calibration 

def _calibrate_mondrian(cal_logits_data, cal_raw_data, score_fn, B, alpha):
    """
    Compute per-bin thresholds from calibration data.

    Returns:
        boundaries : (B+1,) array — entropy quantile boundaries
        q_hats     : (B,) array  — per-bin quantile thresholds
    """
    entropies = []
    scores = []

    for row, item in zip(cal_logits_data, cal_raw_data):
        probs = softmax(row["logits_options"])
        entropies.append(_entropy(probs))
        scores.append(score_fn(probs, OPTIONS.index(item["answer"])))

    entropies = np.array(entropies)
    scores = np.array(scores)

    # Entropy quantile boundaries — equal-frequency bins
    boundaries = np.quantile(entropies, np.linspace(0, 1, B + 1))
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    q_hats = np.empty(B)
    for b in range(B):
        mask = np.array([_assign_bin(h, boundaries) == b for h in entropies])
        S_b = scores[mask]
        n_b = mask.sum()
        if n_b == 0:
            q_hats[b] = 1.0
        else:
            level = min(np.ceil((n_b + 1) * (1 - alpha)) / n_b, 1.0)
            q_hats[b] = np.quantile(S_b, level, method="higher")

    return boundaries, q_hats


# ── Phase 2: Prediction set construction 

def _predict_lac(probs, q_hat):
    ps = [OPTIONS[i] for i, p in enumerate(probs) if p >= 1 - q_hat]
    return ps if ps else [OPTIONS[np.argmax(probs)]]


def _predict_aps(probs, q_hat):
    pi = np.argsort(probs)[::-1]
    cum = np.take_along_axis(probs, pi, axis=0).cumsum()
    ps = [OPTIONS[pi[ii]] for ii, s in enumerate(cum) if s <= q_hat]
    return ps if ps else [OPTIONS[pi[0]]]


# ── Public API — matches LAC_CP / APS_CP interface 

def mondrian_LAC_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods,
                    alpha=0.1, B=3):
    """
    Mondrian CP with LAC scores, stratified by B entropy bins.

    Returns pred_sets_all with the same structure as LAC_CP:
        {key: {str(id): [options]}}
    """
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = m + "_" + fs
            cal_logits  = logits_data_all[key]["cal"]
            test_logits = logits_data_all[key]["test"]

            boundaries, q_hats = _calibrate_mondrian(
                cal_logits, cal_raw_data, _lac_score, B, alpha)

            pred_sets = {}
            for row in test_logits:
                probs = softmax(row["logits_options"])
                b = _assign_bin(_entropy(probs), boundaries)
                pred_sets[str(row["id"])] = _predict_lac(probs, q_hats[b])

            pred_sets_all[key] = pred_sets
    return pred_sets_all


def mondrian_APS_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods,
                    alpha=0.1, B=3):
    """
    Mondrian CP with APS scores, stratified by B entropy bins.

    Returns pred_sets_all with the same structure as APS_CP:
        {key: {str(id): [options]}}
    """
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key = m + "_" + fs
            cal_logits  = logits_data_all[key]["cal"]
            test_logits = logits_data_all[key]["test"]

            boundaries, q_hats = _calibrate_mondrian(
                cal_logits, cal_raw_data, _aps_score, B, alpha)

            pred_sets = {}
            for row in test_logits:
                probs = softmax(row["logits_options"])
                b = _assign_bin(_entropy(probs), boundaries)
                pred_sets[str(row["id"])] = _predict_aps(probs, q_hats[b])

            pred_sets_all[key] = pred_sets
    return pred_sets_all


# ── Standalone evaluation (for Kaggle notebook use) 

def evaluate_mondrian(cal_data, cal_logits, test_data, test_logits,
                      score="lac", B=3, alpha=0.1):
    """
    Self-contained calibrate + evaluate. Returns a results dict.

    cal_data / test_data : list of {"answer": "A"/"B"/...}
    cal_logits / test_logits : list of {"logits_options": [6 floats]}
    score : "lac" or "aps"
    """
    score_fn = _lac_score if score == "lac" else _aps_score
    predict_fn = _predict_lac if score == "lac" else _predict_aps

    # --- calibrate ---
    entropies_cal = []
    scores_cal = []
    for row, item in zip(cal_logits, cal_data):
        probs = softmax(row["logits_options"])
        entropies_cal.append(_entropy(probs))
        scores_cal.append(score_fn(probs, OPTIONS.index(item["answer"])))

    entropies_cal = np.array(entropies_cal)
    scores_cal = np.array(scores_cal)

    boundaries = np.quantile(entropies_cal, np.linspace(0, 1, B + 1))
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf

    q_hats = np.empty(B)
    bin_sizes = []
    for b in range(B):
        mask = np.array([_assign_bin(h, boundaries) == b for h in entropies_cal])
        S_b = scores_cal[mask]
        n_b = mask.sum()
        bin_sizes.append(int(n_b))
        if n_b == 0:
            q_hats[b] = 1.0
        else:
            level = min(np.ceil((n_b + 1) * (1 - alpha)) / n_b, 1.0)
            q_hats[b] = np.quantile(S_b, level, method="higher")

    print(f"Bin sizes (cal): {bin_sizes}")
    print(f"q̂ per bin:       {[round(float(q), 3) for q in q_hats]}")

    # --- evaluate ---
    correct, covered, sizes, bins = [], [], [], []

    for row, item in zip(test_logits, test_data):
        probs = softmax(row["logits_options"])
        H = _entropy(probs)
        b = _assign_bin(H, boundaries)
        ps = predict_fn(probs, q_hats[b])

        correct.append(int(OPTIONS[np.argmax(probs)] == item["answer"]))
        covered.append(int(item["answer"] in ps))
        sizes.append(len(ps))
        bins.append(b)

    correct = np.array(correct)
    covered = np.array(covered)
    sizes   = np.array(sizes)
    bins    = np.array(bins)

    overall = {
        "Acc": correct.mean() * 100,
        "CR":  covered.mean() * 100,
        "SS":  sizes.mean(),
    }
    print(f"\nOverall  — Acc={overall['Acc']:.2f}%  CR={overall['CR']:.2f}%  SS={overall['SS']:.2f}")

    per_bin = []
    for b in range(B):
        m = bins == b
        if m.sum() == 0:
            per_bin.append(None)
            continue
        pb = {
            "n":   int(m.sum()),
            "Acc": correct[m].mean() * 100,
            "CR":  covered[m].mean() * 100,
            "SS":  sizes[m].mean(),
        }
        per_bin.append(pb)
        print(f"Bin {b} (n={pb['n']:3d}) — "
              f"Acc={pb['Acc']:.2f}%  CR={pb['CR']:.2f}%  SS={pb['SS']:.2f}")

    return {
        "overall": overall,
        "per_bin": per_bin,
        "boundaries": boundaries,
        "q_hats": q_hats.tolist(),
        "B": B,
    }
