#!/usr/bin/env python3
import json
import os
import glob
import numpy as np


OUTPUTS_DIR  = "outputs_base"
DATA_NAMES   = ["mmlu_10k"]          # extend if you add more datasets
PROMPT       = "base"
ICL          = "icl1"
KEY          = f"{PROMPT}_{ICL}"

# Friendly display names
MODEL_DISPLAY = {
    "qwen3.5:2b":      "Qwen3.5-2B",
    "qwen3.5:4b":      "Qwen3.5-4B",
    "qwen3.5:9b":      "Qwen3.5-9B",
    "gemma4:e4b":      "Gemma4-E4B",
    "llama3.1:latest": "Llama3.1-8B",
}

DATA_DISPLAY = {
    "mmlu_10k": "QA (MMLU)",
}


def load_result(model_name):
    path = os.path.join(OUTPUTS_DIR, f"{model_name}_all_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(result, data_name):
    d = result[data_name]
    acc = 100 * d["Acc"][KEY]
    ss  = np.mean([d["LAC_set_size"][KEY], d["APS_set_size"][KEY]])
    cr  = 100 * np.mean([d["LAC_coverage"][KEY], d["APS_coverage"][KEY]])
    return acc, ss, cr


def main():
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS  —  n=20 samples, base prompt, 1-shot ICL")
    print("=" * 70)

    header = f"{'Model':<18}" + "".join(
        f"  {'CR%':>6} {'Acc%':>6} {'SS':>5}" for _ in DATA_NAMES
    )
    subheader = " " * 18 + "".join(
        f"  {DATA_DISPLAY.get(d, d):>18}" for d in DATA_NAMES
    )
    print(subheader)
    print(header)
    print("-" * 70)

    rows = []
    for model_key, display in MODEL_DISPLAY.items():
        result = load_result(model_key)
        if result is None:
            print(f"{display:<18}  (no results yet)")
            continue
        line = f"{display:<18}"
        accs, sss, crs = [], [], []
        for data_name in DATA_NAMES:
            if data_name not in result:
                line += f"  {'N/A':>6} {'N/A':>6} {'N/A':>5}"
                continue
            acc, ss, cr = extract_metrics(result, data_name)
            accs.append(acc); sss.append(ss); crs.append(cr)
            line += f"  {cr:>6.2f} {acc:>6.2f} {ss:>5.2f}"
        if accs:
            rows.append((display, np.mean(crs), np.mean(accs), np.mean(sss)))
        print(line)

    print("-" * 70)

    # LaTeX table output
    print("\n\n" + "=" * 70)
    print("  LATEX TABLE")
    print("=" * 70)
    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \caption{Benchmark results: CR (\%), Acc (\%), SS — $n=20$ samples, base prompt, 1-shot ICL, $\alpha=0.1$.}")
    print(r"  \label{tab:benchmark_results}")
    print(r"  \begin{tabular}{lccc}")
    print(r"    \toprule")
    print(r"    \textbf{Model} & \textbf{CR (\%)} & \textbf{Acc (\%)} & \textbf{SS} \\")
    print(r"    \midrule")
    for display, cr, acc, ss in sorted(rows, key=lambda x: -x[2]):
        bold = r"\textbf{" if acc == max(r[2] for r in rows) else ""
        endb = r"}" if bold else ""
        print(f"    {bold}{display}{endb} & {cr:.2f} & {acc:.2f} & {ss:.2f} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
