import numpy as np
from collections import Counter

options = ["A", "B", "C", "D", "E", "F"]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_accuracy(logits_data, raw_data):
    res = []
    preds = []
    for idx, row in enumerate(raw_data):
        truth_answer = row["answer"]
        pred = logits_data[idx]
        assert pred["id"] == row["id"]
        pred_answer = options[np.argmax(pred["logits_options"])]
        preds.append(pred_answer)
        if pred_answer == truth_answer:
            res.append(1)
        else:
            res.append(0)
    return sum(res) / len(res), preds

def cal_acc(logits_data_all, test_raw_data, prompt_methods, icl_methods):
    results_acc = {}
    E_ratios = {}
    F_ratios = {}
    for m in prompt_methods:
        for fs in icl_methods:
            test_logits_data = logits_data_all[m+"_"+fs]["test"]
            acc, preds = get_accuracy(test_logits_data, test_raw_data)
            results_acc[m+"_"+fs] = acc
            counts = Counter(preds)
            E_ratios[m+"_"+fs] = counts["E"] / len(preds) if "E" in counts else 0
            F_ratios[m+"_"+fs] = counts["F"] / len(preds) if "F" in counts else 0
    return results_acc, E_ratios, F_ratios

def cal_coverage(pred_sets_all, test_id_to_answer, prompt_methods, icl_methods):
    coverage_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            cover = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                if test_id_to_answer[k] in v:
                    cover.append(1)
                else:
                    cover.append(0)
            coverage_all[m+"_"+fs] = sum(cover) / len(cover)
    return coverage_all

def cal_set_size(pred_sets_all, prompt_methods, icl_methods):
    set_sizes = {}
    for m in prompt_methods:
        for fs in icl_methods:
            sz = []
            pred_sets = pred_sets_all[m+"_"+fs]
            for k, v in pred_sets.items():
                sz.append(len(v))
            set_sizes[m+"_"+fs] = sum(sz) / len(sz)
    return set_sizes

def cal_uacc(results_acc, set_sizes, num_options=6):
    results_uacc = {}
    for k, v in results_acc.items():
        results_uacc[k] = v * np.sqrt(num_options) / set_sizes[k]
    return results_uacc