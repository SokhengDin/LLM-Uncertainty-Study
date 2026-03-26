import numpy as np
from collections import Counter

OPTIONS = ["A", "B", "C", "D", "E", "F"]


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_accuracy(logits_data, raw_data):
    res, preds = [], []
    for idx, row in enumerate(raw_data):
        assert logits_data[idx]["id"] == row["id"]
        pred = OPTIONS[np.argmax(logits_data[idx]["logits_options"])]
        preds.append(pred)
        res.append(1 if pred == row["answer"] else 0)
    return sum(res) / len(res), preds


def cal_acc(logits_data_all, test_raw_data, prompt_methods, icl_methods):
    results_acc, E_ratios, F_ratios = {}, {}, {}
    for m in prompt_methods:
        for fs in icl_methods:
            key              = m + "_" + fs
            test_logits_data = logits_data_all[key]["test"]
            acc, preds       = get_accuracy(test_logits_data, test_raw_data)
            counts           = Counter(preds)
            results_acc[key] = acc
            E_ratios[key]    = counts.get("E", 0) / len(preds)
            F_ratios[key]    = counts.get("F", 0) / len(preds)
    return results_acc, E_ratios, F_ratios


def cal_coverage(pred_sets_all, test_id_to_answer, prompt_methods, icl_methods):
    coverage_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key       = m + "_" + fs
            pred_sets = pred_sets_all[key]
            cover     = [1 if test_id_to_answer[k] in v else 0 for k, v in pred_sets.items()]
            coverage_all[key] = sum(cover) / len(cover)
    return coverage_all


def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    set_sizes = {}
    for m in prompt_methods:
        for fs in icl_methods:
            key       = m + "_" + fs
            sizes     = [len(v) for v in pred_sets_all[key].values()]
            set_sizes[key] = sum(sizes) / len(sizes)
    return set_sizes


def cal_uacc(results_acc, set_sizes, num_options=6):
    return {k: v * np.sqrt(num_options) / set_sizes[k] for k, v in results_acc.items()}
