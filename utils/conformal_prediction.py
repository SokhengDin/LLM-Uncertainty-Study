import numpy as np

OPTIONS = ["A", "B", "C", "D", "E", "F"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _quantile_threshold(scores, alpha):
    n       = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(scores, q_level, method="higher")


def LAC_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    """Conformal prediction with the LAC (Least Ambiguous Classifier) score."""
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key              = m + "_" + fs
            cal_logits_data  = logits_data_all[key]["cal"]
            test_logits_data = logits_data_all[key]["test"]

            cal_scores = []
            for idx, row in enumerate(cal_logits_data):
                assert cal_raw_data[idx]["id"] == row["id"]
                probs = softmax(row["logits_options"])
                cal_scores.append(1 - probs[OPTIONS.index(cal_raw_data[idx]["answer"])])

            qhat      = _quantile_threshold(cal_scores, alpha)
            pred_sets = {}
            for row in test_logits_data:
                probs = softmax(row["logits_options"])
                ps    = [OPTIONS[i] for i, p in enumerate(probs) if p >= 1 - qhat]
                if not ps:
                    ps = [OPTIONS[np.argmax(probs)]]
                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets
    return pred_sets_all


def APS_CP(logits_data_all, cal_raw_data, prompt_methods, icl_methods, alpha=0.1):
    """Conformal prediction with the APS (Adaptive Prediction Sets) score."""
    pred_sets_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key              = m + "_" + fs
            cal_logits_data  = logits_data_all[key]["cal"]
            test_logits_data = logits_data_all[key]["test"]

            cal_scores = []
            for idx, row in enumerate(cal_logits_data):
                assert cal_raw_data[idx]["id"] == row["id"]
                probs     = softmax(row["logits_options"])
                pi        = np.argsort(probs)[::-1]
                cum_sum   = np.take_along_axis(probs, pi, axis=0).cumsum()
                cum_sum_r = np.take_along_axis(cum_sum, pi.argsort(), axis=0)
                cal_scores.append(cum_sum_r[OPTIONS.index(cal_raw_data[idx]["answer"])])

            qhat      = _quantile_threshold(cal_scores, alpha)
            pred_sets = {}
            for row in test_logits_data:
                probs   = softmax(row["logits_options"])
                pi      = np.argsort(probs)[::-1]
                cum_sum = np.take_along_axis(probs, pi, axis=0).cumsum()
                ps      = []
                for ii, s in enumerate(cum_sum):
                    if s > qhat:
                        break
                    ps.append(OPTIONS[pi[ii]])
                if not ps:
                    ps = [OPTIONS[pi[0]]]
                pred_sets[str(row["id"])] = ps

            pred_sets_all[key] = pred_sets
    return pred_sets_all
