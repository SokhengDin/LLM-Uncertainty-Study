import json
import os
import argparse
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from utils.conformal_prediction import LAC_CP, APS_CP
from utils.metrics import cal_acc, cal_coverage, cal_set_size, cal_uacc


IDS_TO_REMOVE = [1, 3, 5, 7, 9]  # few-shot demonstration indices


def get_raw_data(raw_data_dir, data_name, cal_ratio, valid_ids=None):
    raw_data = json.load(open(os.path.join(raw_data_dir, data_name + ".json"), "r"))
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in IDS_TO_REMOVE]
    if valid_ids is not None:
        raw_data = [item for item in raw_data if item["id"] in valid_ids]
    raw_data = sorted(raw_data, key=lambda x: x["id"])
    cal_raw_data, test_raw_data = train_test_split(raw_data, train_size=cal_ratio, random_state=42)
    print(f"{data_name}: total={len(raw_data)}, cal={len(cal_raw_data)}, test={len(test_raw_data)}")
    return cal_raw_data, test_raw_data


def get_logits_data(model_name, data_name, logits_data_dir, cal_ratio,
                    prompt_methods, icl_methods, max_samples=0):
    logits_data_all = {}
    first_logits    = None
    for m in prompt_methods:
        for fs in icl_methods:
            suffix     = f"_sample{max_samples}" if max_samples > 0 else ""
            logits_file = os.path.join(logits_data_dir,
                                       model_name + "_" + data_name + "_" + m + "_" + fs + suffix + ".pkl")
            with open(logits_file, "rb") as f:
                logits_data = pickle.load(f)
            logits_data = [item for idx, item in enumerate(logits_data) if idx not in IDS_TO_REMOVE]
            logits_data = sorted(logits_data, key=lambda x: x["id"])
            cal_logits, test_logits = train_test_split(logits_data, train_size=cal_ratio, random_state=42)
            if first_logits is None:
                first_logits = logits_data
            logits_data_all[m + "_" + fs] = {"cal": cal_logits, "test": test_logits}
    valid_ids = {item["id"] for item in first_logits}
    return logits_data_all, valid_ids


def convert_id_to_ans(test_raw_data):
    return {str(row["id"]): row["answer"] for row in test_raw_data}


def main(args):
    all_data_results = {}

    for data_name in args.data_names:
        print(f"\n{'='*50}")
        print(f"Processing {data_name}")
        print(f"{'='*50}")

        logits_data_all, valid_ids = get_logits_data(
            args.model, data_name, args.logits_data_dir,
            args.cal_ratio, args.prompt_methods, args.icl_methods, args.max_samples
        )
        cal_raw_data, test_raw_data = get_raw_data(
            args.raw_data_dir, data_name, args.cal_ratio, valid_ids
        )

        results_acc, E_ratios, F_ratios = cal_acc(
            logits_data_all, test_raw_data, args.prompt_methods, args.icl_methods
        )
        test_id_to_answer = convert_id_to_ans(test_raw_data)

        pred_sets_LAC   = LAC_CP(logits_data_all, cal_raw_data, args.prompt_methods, args.icl_methods, args.alpha)
        coverage_LAC    = cal_coverage(pred_sets_LAC, test_id_to_answer, args.prompt_methods, args.icl_methods)
        set_sizes_LAC   = cal_set_size(pred_sets_LAC, args.prompt_methods, args.icl_methods)
        uacc_LAC        = cal_uacc(results_acc, set_sizes_LAC)

        pred_sets_APS   = APS_CP(logits_data_all, cal_raw_data, args.prompt_methods, args.icl_methods, args.alpha)
        coverage_APS    = cal_coverage(pred_sets_APS, test_id_to_answer, args.prompt_methods, args.icl_methods)
        set_sizes_APS   = cal_set_size(pred_sets_APS, args.prompt_methods, args.icl_methods)
        uacc_APS        = cal_uacc(results_acc, set_sizes_APS)

        all_data_results[data_name] = {
            "Acc":          results_acc,
            "E_rate":       E_ratios,
            "F_rate":       F_ratios,
            "LAC_set_size": set_sizes_LAC,
            "APS_set_size": set_sizes_APS,
            "LAC_coverage": coverage_LAC,
            "APS_coverage": coverage_APS,
            "UAcc_LAC":     uacc_LAC,
            "UAcc_APS":     uacc_APS,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.model}_all_results.json")
    with open(output_file, "w") as f:
        json.dump(all_data_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")

    acc_avg, ss_avg, cr_avg = [], [], []
    for data_name in args.data_names:
        acc = 100 * np.mean(list(all_data_results[data_name]["Acc"].values()))
        ss  = np.mean([
            np.mean(list(all_data_results[data_name]["LAC_set_size"].values())),
            np.mean(list(all_data_results[data_name]["APS_set_size"].values())),
        ])
        cr  = 100 * np.mean([
            np.mean(list(all_data_results[data_name]["LAC_coverage"].values())),
            np.mean(list(all_data_results[data_name]["APS_coverage"].values())),
        ])
        acc_avg.append(acc)
        ss_avg.append(ss)
        cr_avg.append(cr)
        print(f"{data_name}  Acc={acc:.2f}%  SS={ss:.2f}  CR={cr:.2f}%")

    print(f"\nAverage  Acc={np.mean(acc_avg):.2f}%  SS={np.mean(ss_avg):.2f}  CR={np.mean(cr_avg):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str,   default="qwen3.5:4b")
    parser.add_argument("--raw_data_dir",   type=str,   default="data")
    parser.add_argument("--logits_data_dir",type=str,   default="outputs_base")
    parser.add_argument("--output_dir",     type=str,   default="outputs_base")
    parser.add_argument("--data_names",     nargs="*",
                        default=["mmlu_10k", "cosmosqa_10k", "hellaswag_10k",
                                 "halu_dialogue", "halu_summarization"])
    parser.add_argument("--prompt_methods", nargs="*",  default=["base", "shared", "task"])
    parser.add_argument("--icl_methods",    nargs="*",  default=["icl1"])
    parser.add_argument("--cal_ratio",      type=float, default=0.5)
    parser.add_argument("--alpha",          type=float, default=0.1)
    parser.add_argument("--max_samples",    type=int,   default=0)
    args = parser.parse_args()
    main(args)
