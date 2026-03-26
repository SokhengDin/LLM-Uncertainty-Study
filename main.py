import json
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from utils.conformal_prediction import LAC_CP, APS_CP
from utils.metrics import cal_acc, cal_coverage, cal_set_size, cal_uacc

ids_to_remove = [1, 3, 5, 7, 9]

def get_raw_data(raw_data_dir, data_name, cal_ratio):
    raw_data = json.load(open(os.path.join(raw_data_dir, data_name+".json"), "r"))
    raw_data = [item for idx, item in enumerate(raw_data) if idx not in ids_to_remove]
    cal_raw_data, test_raw_data = train_test_split(raw_data, train_size=cal_ratio, random_state=42)
    print(f"{data_name}: total={len(raw_data)}, cal={len(cal_raw_data)}, test={len(test_raw_data)}")
    return cal_raw_data, test_raw_data

def get_logits_data(model_name, data_name, cal_raw_data, test_raw_data, 
                    logits_data_dir, cal_ratio, prompt_methods, icl_methods):
    logits_data_all = {}
    for m in prompt_methods:
        for fs in icl_methods:
            logits_file = os.path.join(logits_data_dir, model_name+"_"+data_name+"_"+m+"_"+fs+".pkl")
            with open(logits_file, 'rb') as f:
                logits_data = pickle.load(f)
            logits_data = [item for idx, item in enumerate(logits_data) if idx not in ids_to_remove]
            cal_logits_data, test_logits_data = train_test_split(logits_data, train_size=cal_ratio, random_state=42)
            assert len(cal_logits_data) == len(cal_raw_data)
            assert len(test_logits_data) == len(test_raw_data)
            logits_data_all[m+"_"+fs] = {}
            logits_data_all[m+"_"+fs]["cal"] = cal_logits_data
            logits_data_all[m+"_"+fs]["test"] = test_logits_data
    return logits_data_all

def convert_id_to_ans(test_raw_data):
    test_id_to_answer = {}
    for row in test_raw_data:
        test_id_to_answer[str(row["id"])] = row["answer"]
    return test_id_to_answer

def main(args):
    all_data_results = {}
    
    for data_name in args.data_names:
        print(f"\n{'='*50}")
        print(f"Processing {data_name}")
        print(f"{'='*50}")
        
        cal_raw_data, test_raw_data = get_raw_data(args.raw_data_dir, data_name, args.cal_ratio)
        logits_data_all = get_logits_data(args.model, data_name, cal_raw_data, test_raw_data, 
                                          args.logits_data_dir, args.cal_ratio,
                                          args.prompt_methods, args.icl_methods)
        
        results_acc, E_ratios, F_ratios = cal_acc(logits_data_all, test_raw_data,
                                                  args.prompt_methods, args.icl_methods)
        test_id_to_answer = convert_id_to_ans(test_raw_data)
        
        # LAC
        pred_sets_all_LAC = LAC_CP(logits_data_all, cal_raw_data,
                                   args.prompt_methods, args.icl_methods,
                                   alpha=args.alpha)
        coverage_all_LAC = cal_coverage(pred_sets_all_LAC, test_id_to_answer,
                                        args.prompt_methods, args.icl_methods)
        set_sizes_LAC = cal_set_size(pred_sets_all_LAC, args.prompt_methods, args.icl_methods)
        results_uacc_LAC = cal_uacc(results_acc, set_sizes_LAC)
        
        # APS
        pred_sets_all_APS = APS_CP(logits_data_all, cal_raw_data,
                                   args.prompt_methods, args.icl_methods,
                                   alpha=args.alpha)
        coverage_all_APS = cal_coverage(pred_sets_all_APS, test_id_to_answer,
                                        args.prompt_methods, args.icl_methods)
        set_sizes_APS = cal_set_size(pred_sets_all_APS, args.prompt_methods, args.icl_methods)
        results_uacc_APS = cal_uacc(results_acc, set_sizes_APS)
        
        all_data_results[data_name] = {}
        all_data_results[data_name]["Acc"] = results_acc
        all_data_results[data_name]["E_rate"] = E_ratios
        all_data_results[data_name]["F_rate"] = F_ratios
        all_data_results[data_name]["LAC_set_size"] = set_sizes_LAC
        all_data_results[data_name]["APS_set_size"] = set_sizes_APS
        all_data_results[data_name]["LAC_coverage"] = coverage_all_LAC
        all_data_results[data_name]["APS_coverage"] = coverage_all_APS
        all_data_results[data_name]["UAcc_LAC"] = results_uacc_LAC
        all_data_results[data_name]["UAcc_APS"] = results_uacc_APS
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{args.model}_all_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_data_results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    acc_avg = []
    ss_avg = []
    cr_avg = []
    
    for data_name in args.data_names:
        acc = 100 * np.mean(list(all_data_results[data_name]["Acc"].values()))
        acc_avg.append(acc)
        print(f"{data_name} Accuracy: {acc:.2f}%")
        
        lac_ss = np.mean(list(all_data_results[data_name]["LAC_set_size"].values()))
        aps_ss = np.mean(list(all_data_results[data_name]["APS_set_size"].values()))
        ss = (lac_ss + aps_ss) / 2
        ss_avg.append(ss)
        print(f"{data_name} Set Size: {ss:.2f}")
        
        lac_cr = 100 * np.mean(list(all_data_results[data_name]["LAC_coverage"].values()))
        aps_cr = 100 * np.mean(list(all_data_results[data_name]["APS_coverage"].values()))
        cr = (lac_cr + aps_cr) / 2
        cr_avg.append(cr)
        print(f"{data_name} Coverage: {cr:.2f}%")
        print()
    
    print(f"Average Accuracy: {np.mean(acc_avg):.2f}%")
    print(f"Average Set Size: {np.mean(ss_avg):.2f}")
    print(f"Average Coverage: {np.mean(cr_avg):.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3.5:0.8b")
    parser.add_argument("--raw_data_dir", type=str, default="data")
    parser.add_argument("--logits_data_dir", type=str, default="outputs_base")
    parser.add_argument("--output_dir", type=str, default="outputs_base")
    parser.add_argument("--data_names", nargs='*', 
                        default=['mmlu_10k', 'cosmosqa_10k', 'hellaswag_10k', 'halu_dialogue', 'halu_summarization'])
    parser.add_argument("--prompt_methods", nargs='*', default=['base', 'shared', 'task'])
    parser.add_argument("--icl_methods", nargs='*', default=['icl1'])
    parser.add_argument("--cal_ratio", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    
    main(args)