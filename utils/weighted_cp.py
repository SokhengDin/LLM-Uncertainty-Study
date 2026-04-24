import json
import pickle
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

OPTIONS = ["A", "B", "C", "D", "E", "F"]



def _softmax(x):
    e = np.exp(np.array(x, dtype=float) - np.max(x))
    return e / e.sum()


def _lac_score(probs, answer):
    """LAC nonconformity score: 1 − p(true label)."""
    return 1.0 - probs[OPTIONS.index(answer)]


def _weighted_quantile(scores, weights, alpha):
    """
    Weighted empirical quantile at level 1-alpha.

    Implements inf{q : sum_i w_i * 1[s_i <= q] >= 1-alpha}
    where weights are the normalised p_i^w (NOT including the test point).
    The test-point weight p_{n+1}^w is folded in by inflating alpha:
        effective threshold at 1 - alpha * (1 + 1/n) approximately,
        or exactly by adding an infinity score with weight p_{n+1}^w.

    We use the exact formulation: augment scores with +inf and weight
    p_{n+1}^w, then take the weighted (1-alpha) quantile.
    """
    scores  = np.array(scores, dtype=float)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()           # normalise to sum=1 (cal part)

    # augment with the test point (score = +inf, weight = mean of remaining)
    # Following Tibshirani 2019 exactly: p_{n+1}^w = w(x_{n+1}) / (sum_cal + w(x_{n+1}))
    # Here we treat w(x_{n+1}) as the average cal weight (unknown at cal time).
    # For a cleaner simulation we set w_{n+1} = 1 (neutral).
    w_test     = 1.0
    w_total    = weights.sum() * len(weights) + w_test   # unnorm sum + test
    p_cal      = weights / (weights.sum() + w_test / len(weights))
    p_test     = (w_test / len(weights)) / (weights.sum() + w_test / len(weights))

    aug_scores  = np.append(scores, np.inf)
    aug_weights = np.append(weights, w_test / len(weights))
    aug_weights = aug_weights / aug_weights.sum()

    order  = np.argsort(aug_scores)
    s_sort = aug_scores[order]
    w_sort = aug_weights[order]
    cumw   = np.cumsum(w_sort)
    idx    = np.searchsorted(cumw, 1.0 - alpha)
    return float(s_sort[min(idx, len(s_sort) - 1)])



def load_all_pairs(short, ds, out_dir, data_dir, samples=100, seed=42):
    """
    Load ALL (data, logits) pairs after removing demo examples.
    Returns list of (data_item, logits_row) — NOT split into cal/test yet.
    """
    import os
    pkl_path  = os.path.join(out_dir, f"{short}_{ds}_base_icl1_sample{samples}.pkl")
    data_path = os.path.join(data_dir, f"{ds}.json")
    if not os.path.exists(pkl_path) or not os.path.exists(data_path):
        return None

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
    return paired



def estimate_weights(cal_features, test_features):
    """
    Estimate likelihood-ratio weights w(x) = P_test(x) / P_cal(x) via
    logistic regression (Sugiyama et al. 2007 / Cheng & Bartlett 2007).

    Label cal=0, test=1; fit P(test | x); then
        w(x) = P(test|x) / P(cal|x) * (n_cal / n_test)   [ratio of priors cancels]
    Returns w for cal points.
    """
    n_cal  = len(cal_features)
    n_test = len(test_features)
    X = np.vstack([cal_features, test_features])
    y = np.array([0] * n_cal + [1] * n_test)

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_s, y)

    # P(test | x) for each cal point
    p_test_given_x = clf.predict_proba(X_s[:n_cal])[:, 1]
    p_cal_given_x  = 1.0 - p_test_given_x + 1e-9

    # likelihood ratio (prior ratio cancels when n_cal == n_test)
    prior_ratio = n_cal / n_test
    weights = (p_test_given_x / p_cal_given_x) * prior_ratio
    return weights



def standard_cp(cal_pairs, test_pairs, alpha=0.1):
    """
    Standard split conformal prediction with LAC score.
    Returns (coverage_rate, avg_set_size, prediction_sets).
    """
    cal_scores = []
    for item, row in cal_pairs:
        probs = _softmax(row["logits_options"])
        cal_scores.append(_lac_score(probs, item["answer"]))

    n     = len(cal_scores)
    q_lvl = np.ceil((n + 1) * (1 - alpha)) / n
    qhat  = np.quantile(cal_scores, min(q_lvl, 1.0), method="higher")

    pred_sets, covered = [], []
    for item, row in test_pairs:
        probs = _softmax(row["logits_options"])
        ps = [OPTIONS[i] for i, p in enumerate(probs) if p >= 1 - qhat]
        if not ps:
            ps = [OPTIONS[int(np.argmax(probs))]]
        pred_sets.append(ps)
        covered.append(int(item["answer"] in ps))

    return {
        "qhat":     float(qhat),
        "coverage": float(np.mean(covered)),
        "avg_size": float(np.mean([len(ps) for ps in pred_sets])),
        "pred_sets": pred_sets,
        "covered":   covered,
    }



def weighted_cp(cal_pairs, test_pairs, alpha=0.1):
    """
    Weighted split conformal prediction (Tibshirani et al. 2019).

    Features: softmax probability vectors (dim = number of options).
    Weights estimated via logistic regression distinguishing cal vs test.
    """
    cal_features  = np.array([_softmax(r["logits_options"]) for _, r in cal_pairs])
    test_features = np.array([_softmax(r["logits_options"]) for _, r in test_pairs])

    weights = estimate_weights(cal_features, test_features)
    weights = np.clip(weights, 1e-3, 1e3)     # numerical stability

    cal_scores = []
    for item, row in cal_pairs:
        probs = _softmax(row["logits_options"])
        cal_scores.append(_lac_score(probs, item["answer"]))

    qhat_w = _weighted_quantile(cal_scores, weights, alpha)

    pred_sets, covered = [], []
    for item, row in test_pairs:
        probs = _softmax(row["logits_options"])
        ps = [OPTIONS[i] for i, p in enumerate(probs) if p >= 1 - qhat_w]
        if not ps:
            ps = [OPTIONS[int(np.argmax(probs))]]
        pred_sets.append(ps)
        covered.append(int(item["answer"] in ps))

    return {
        "qhat_w":   float(qhat_w),
        "coverage": float(np.mean(covered)),
        "avg_size": float(np.mean([len(ps) for ps in pred_sets])),
        "pred_sets": pred_sets,
        "covered":   covered,
        "weights":   weights.tolist(),
    }



def run_comparison(models, datasets, out_dir, data_dir, alpha=0.1, seed=42):
    """
    For each (model, dataset), simulate cross-prompt shift by splitting
    the 100 samples into two halves: cal half = 'prompt A', test half = 'prompt B'.
    Run both standard CP and weighted CP, report coverage and set size.

    Returns nested dict: results[model][dataset] = {standard: ..., weighted: ...}
    """
    results = {}
    for short in models:
        results[short] = {}
        for ds in datasets:
            paired = load_all_pairs(short, ds, out_dir, data_dir, seed=seed)
            if paired is None or len(paired) < 10:
                continue
            n_cal    = len(paired) // 2
            cal_pairs  = paired[:n_cal]
            test_pairs = paired[n_cal:]

            std = standard_cp(cal_pairs, test_pairs, alpha)
            wcp = weighted_cp(cal_pairs, test_pairs, alpha)

            results[short][ds] = {"standard": std, "weighted": wcp, "n_cal": n_cal, "n_test": len(test_pairs)}

    return results
