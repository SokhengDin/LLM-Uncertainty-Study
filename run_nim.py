#!/usr/bin/env python3
import json, os, pickle, random, subprocess, sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

try:
    load_dotenv()
except Exception:
    pass  # python-dotenv not installed — fallback to os.environ / shell export

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("NVIDIA_API_KEY") or ""
if not API_KEY:
    raise SystemExit("ERROR: NVIDIA_API_KEY not set. Add it to .env or run:\n"
                     "  export NVIDIA_API_KEY=nvapi-...")
BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

MODELS = [
    "meta/llama-3.1-8b-instruct",          # 200 ✓
    "meta/llama-3.2-3b-instruct",          # 200 ✓
    "nvidia/llama-3.1-nemotron-nano-8b-v1", # 200 ✓
]

DATASETS = [
    "mmlu_10k",
    "cosmosqa_10k",
    "hellaswag_10k",
    "halu_dialogue",
    "halu_summarization",
]

SAMPLES  = 100   # n_cal ≈ 52 → qhat ≈ 0.923 (non-trivial CP)
ALPHA    = 0.1
PROMPT   = "base"
ICL      = "icl1"
OUT_DIR  = "outputs_nim"
FIG_DIR  = "figures_nim"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "utils"))

# ── Few-shot setup (mirrors generate_logits.py) ───────────────────────────────
FEW_SHOT_IDS = {
    "MMLU"               : [1, 3, 5, 7, 9],
    "HellaSwag"          : [1, 3, 5, 7, 9],
    "CosmosQA"           : [1, 3, 5, 7, 9],
    "Halu-OpenDialKG"    : [5, 7, 9],
    "Halu-CNN/DailyMail" : [9],
}
FEW_SHOT_RESERVE = 10


# ── NIM client (inline — no extra file needed) ────────────────────────────────
import requests as _req

CHOICES = ["A", "B", "C", "D", "E", "F"]

def get_choice_logits(prompt: str, model_id: str, retries: int = 5) -> np.ndarray:
    """Single API call → logprobs for A-F. Retries on timeout and 429."""
    import time
    for attempt in range(retries):
        try:
            resp = _req.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={
                    "model":        model_id,
                    "messages":     [{"role": "user", "content": prompt}],
                    "max_tokens":   1,
                    "temperature":  0,
                    "logprobs":     True,
                    "top_logprobs": 20,
                },
                timeout=60,
            )
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30)) + 5
                print(f"\n  Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            top = (resp.json()["choices"][0]
                   .get("logprobs", {})
                   .get("content", [{}])[0]
                   .get("top_logprobs", []))
            lp_map = {e["token"].strip(): e["logprob"] for e in top}
            return np.array([lp_map.get(c, -1e9) for c in CHOICES], dtype=np.float32)
        except (_req.exceptions.ReadTimeout, _req.exceptions.ConnectionError):
            wait = 10 * (attempt + 1)
            print(f"\n  Timeout (attempt {attempt+1}/{retries}), retrying in {wait}s...")
            time.sleep(wait)
    print(f"\n  ERROR: all {retries} attempts failed — skipping question")
    return np.full(len(CHOICES), -1e9, dtype=np.float32)


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_data(path, max_samples):
    data = json.load(open(path))
    few  = data[:FEW_SHOT_RESERVE]
    rest = data[FEW_SHOT_RESERVE:]
    if len(rest) > max_samples:
        random.seed(42)
        rest = random.sample(rest, max_samples)
    print(f"  {FEW_SHOT_RESERVE} few-shot + {len(rest)} test = {len(few+rest)} total")
    return few + rest


def get_fewshot(data):
    src = data[0]["source"]
    return [data[i] for i in FEW_SHOT_IDS[src]]


def fmt_example(ex, prompt, with_answer=False):
    src = ex["source"]
    if src == "MMLU":
        prompt += "Question: " + ex["question"] + "\nChoices:\n"
    elif src in ("CosmosQA", "HellaSwag"):
        prompt += "Context: "  + ex["context"]  + "\n"
        prompt += "Question: " + ex["question"] + "\nChoices:\n"
    elif src == "Halu-OpenDialKG":
        prompt += "Dialogue: " + ex["context"]  + "\n"
        prompt += "Question: " + ex["question"] + "\nChoices:\n"
    elif src == "Halu-CNN/DailyMail":
        prompt += "Document: " + ex["context"]  + "\n"
        prompt += "Question: " + ex["question"] + "\nChoices:\n"
    for k, v in ex["choices"].items():
        prompt += f"{k}. {v}\n"
    prompt += "Answer:"
    if with_answer:
        prompt += " " + ex["answer"] + "\n"
    return prompt


def build_prompts(data, fewshot):
    out = []
    for ex in data:
        p = ""
        for fs in fewshot:
            p = fmt_example(fs, p, with_answer=True)
        out.append({"id": ex["id"], "prompt": fmt_example(ex, p)})
    return out


def check_logprobs_support(model_id):
    """Returns True if model supports logprobs=True on NIM."""
    try:
        resp = _req.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": model_id, "messages": [{"role": "user", "content": "A"}],
                  "max_tokens": 1, "logprobs": True, "top_logprobs": 5},
            timeout=20,
        )
        if resp.status_code == 200:
            print(f"  ✓ {model_id} supports logprobs")
            return True
        print(f"  ✗ {model_id} → HTTP {resp.status_code} (skipping)")
        return False
    except Exception as e:
        print(f"  ✗ {model_id} → {e} (skipping)")
        return False


# ── Step 1: generate logits ───────────────────────────────────────────────────
def step_logits():
    os.makedirs(OUT_DIR, exist_ok=True)
    for model_id in MODELS:
        short = model_id.split("/")[-1]
        # Check logprobs support before processing any dataset
        first_ds_done = any(
            os.path.exists(f"{OUT_DIR}/{short}_{ds}_base_icl1_sample{SAMPLES}.pkl")
            for ds in DATASETS
        )
        if not first_ds_done and not check_logprobs_support(model_id):
            continue
        for ds in DATASETS:
            pkl = f"{OUT_DIR}/{short}_{ds}_base_icl1_sample{SAMPLES}.pkl"
            if os.path.exists(pkl):
                print(f"  SKIP  {short} | {ds}")
                continue
            print(f"\n--- {short} | {ds} ---")
            data_path = ROOT / "data" / f"{ds}.json"
            if not data_path.exists():
                print(f"  ERROR: {data_path} not found"); continue
            data    = load_data(str(data_path), SAMPLES)
            fewshot = get_fewshot(data)
            prompts = build_prompts(data, fewshot)
            outputs = []
            for ex in tqdm(prompts, desc=f"{short}|{ds}"):
                logits = get_choice_logits(ex["prompt"], model_id)
                outputs.append({"id": ex["id"], "logits_options": logits})
            with open(pkl, "wb") as f:
                pickle.dump(outputs, f)
            print(f"  Saved → {pkl}")


# ── Step 2: CP evaluation ─────────────────────────────────────────────────────
def step_eval():
    for model_id in MODELS:
        short = model_id.split("/")[-1]
        print(f"\n=== Evaluating {short} ===")
        subprocess.run([
            sys.executable, str(ROOT / "main.py"),
            "--model",           short,
            "--data_names",      *DATASETS,
            "--prompt_methods",  PROMPT,
            "--icl_methods",     ICL,
            "--max_samples",     str(SAMPLES),
            "--alpha",           str(ALPHA),
            "--logits_data_dir", OUT_DIR,
            "--output_dir",      OUT_DIR,
        ], check=False)


# ── Step 3: figures ───────────────────────────────────────────────────────────
def step_figures():
    os.makedirs(FIG_DIR, exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / "plot_results.py"),
        "--samples",     str(SAMPLES),
        "--prompt",      PROMPT,
        "--icl",         ICL,
        "--results_dir", OUT_DIR,
        "--figures_dir", FIG_DIR,
    ], check=False)


# ── Step 4: summary ───────────────────────────────────────────────────────────
def step_summary():
    key = f"{PROMPT}_{ICL}"
    col = 16
    sep = "=" * (30 + col * len(DATASETS))
    print(f"\n{sep}")
    print("RESULTS  (CR% / Acc% / SS avg LAC+APS)")
    print(sep)
    print(f"{'Model':<30}" + "".join(f"{d.split('_')[0]:>{col}}" for d in DATASETS))
    print("-" * (30 + col * len(DATASETS)))
    for model_id in MODELS:
        short = model_id.split("/")[-1]
        path  = f"{OUT_DIR}/{short}_all_results.json"
        if not os.path.exists(path):
            print(f"{short:<30}  (no results)"); continue
        res = json.load(open(path))
        row = f"{short:<30}"
        for d in DATASETS:
            if d not in res or key not in res[d].get("Acc", {}):
                row += f"{'N/A':>{col}}"; continue
            acc = 100 * res[d]["Acc"][key]
            cr  = 100 * np.mean([res[d]["LAC_coverage"][key], res[d]["APS_coverage"][key]])
            ss  =       np.mean([res[d]["LAC_set_size"][key],  res[d]["APS_set_size"][key]])
            row += f"{cr:.0f}/{acc:.0f}/{ss:.1f}".rjust(col)
        print(row)
    # n_cal math: 10 few-shot + 100 test = 110 loaded, -5 demo indices = 105, *0.5 = 52
    print(f"\nn={SAMPLES} → n_cal≈52 → qhat≈{np.ceil(53*0.9)/52:.3f}  (α={ALPHA})")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"NIM Benchmark: {len(MODELS)} models × {len(DATASETS)} tasks × n={SAMPLES}")
    print("Models:", [m.split("/")[-1] for m in MODELS])

    print("\n[1/4] Generating logits...")
    step_logits()

    print("\n[2/4] CP evaluation...")
    step_eval()

    print("\n[3/4] Figures...")
    step_figures()

    step_summary()
    print(f"\nFigures → {FIG_DIR}/\nDone!")
