# LLM Uncertainty Benchmark — Conformal Prediction

Reproduction and extension of **Ye et al. (NeurIPS 2024) "Benchmarking LLMs via Uncertainty Quantification"** using split conformal prediction on open-source models evaluated on 5 MCQA tasks.

---

## Description

We reproduce the conformal prediction benchmark of Ye et al. on 5 models × 5 datasets using direct HuggingFace logits (rather than continuation scoring), then extend the analysis with Expected Calibration Error (ECE). We also identify **prompt shift** the change in model output distribution when the instruction prefix changes as a covariate shift that invalidates the standard CP coverage guarantee, and propose weighted conformal prediction (Tibshirani et al., NeurIPS 2019) as a principled fix.

---

## Mathematical Background

### Split Conformal Prediction

Given a calibration set $\{(X_i, Y_i)\}_{i=1}^n$ and a nonconformity score $s(x, y)$, the split CP prediction set at level $\alpha$ is:

$$C_\alpha(X_{n+1}) = \{y \in \mathcal{Y} : s(X_{n+1}, y) \leq \hat{q}_\alpha\}$$

where the threshold is the empirical quantile:

$$\hat{q}_\alpha = \text{Quantile}\!\left(\{s_i\}_{i=1}^n,\ \frac{\lceil(n+1)(1-\alpha)\rceil}{n}\right)$$

Under exchangeability this satisfies $\mathbb{P}(Y_{n+1} \in C_\alpha(X_{n+1})) \geq 1 - \alpha$.

### Score Functions

**LAC** (Least Ambiguous Classifier, Sadinle et al. 2019):
$$s_{\text{LAC}}(x, y) = 1 - \hat{p}(y \mid x)$$

**APS** (Adaptive Prediction Sets, Romano et al. 2020):
$$s_{\text{APS}}(x, y) = \sum_{y' : \hat{p}(y' \mid x) \geq \hat{p}(y \mid x)} \hat{p}(y' \mid x)$$

### Expected Calibration Error

ECE measures how well model confidence (max softmax probability) matches empirical accuracy across $B$ equal-width bins (Guo et al., ICML 2017):

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$
---

## Project Structure

```
LLM-Uncertainty-Study/
├── data/                        # MCQA datasets (JSON)
│   ├── mmlu_10k.json
│   ├── hellaswag_10k.json
│   ├── cosmosqa_10k.json
│   ├── halu_dialogue.json
│   └── halu_summarization.json
│
├── outputs_kaggle/              # Logits from Kaggle GPU runs
│   └── {model}_{dataset}_base_icl1_sample100.pkl
│
├── figures/                     # Generated plots (PDF)
│
├── notebooks/
│   ├── kaggle_benchmark.ipynb   # Full GPU benchmark (HuggingFace + Kaggle T4)
│   └── conformal_prediction.ipynb
│
├── utils/
│   ├── conformal_prediction.py  # LAC and APS split CP
│   ├── ece.py                   # ECE computation + reliability diagrams
│   ├── generate_logits.py       # Logit extraction via Ollama (local)
│   ├── metrics.py               # Coverage rate, set size, accuracy
│   ├── plot_results.py          # Figure generation
│   ├── prompt.py                # Prompt templates (base, shared, task)
│   ├── ollama_client.py         # Local Ollama API client
│   └── collect_results.py       # Result aggregation
│
├── main.py                      # CP evaluation entry point
├── run_benchmark.sh             # Run CP eval + ECE + figures on kaggle outputs
└── requirements.txt
```

---

## Models & Datasets

| Model | Parameters | Source |
|-------|-----------|--------|
| Qwen3-0.6B | 0.6B | Qwen/Qwen3-0.6B |
| Qwen2.5-3B | 3B | Qwen/Qwen2.5-3B-Instruct |
| Qwen2.5-7B | 7B | Qwen/Qwen2.5-7B-Instruct |
| OLMo-1B | 1B | allenai/OLMo-1B-hf |
| OLMo-7B | 7B | allenai/OLMo-7B-Instruct-hf |

| Dataset | Task | Labels |
|---------|------|--------|
| MMLU | Question Answering | A–D |
| HellaSwag | Commonsense NLI | A–D |
| CosmosQA | Reading Comprehension | A–D |
| HaluDial | Dialogue Response Selection | A–F |
| HaluSum | Summarization QA | A–F |

---

## Setup

### With uv (recommended)

```bash
# Install uv if not already installed
curl -Ls https://astral.sh/uv/install.sh | sh

# Create virtualenv and install all dependencies
uv sync

# Activate
source .venv/bin/activate
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Run CP evaluation + ECE on existing Kaggle logits

```bash
bash run_benchmark.sh
```

This runs in 3 steps:
1. **CP evaluation** — LAC and APS on `outputs_kaggle/` logits via `main.py`
2. **ECE** — calibration analysis, prints table, saves `fig_ece_heatmap.pdf`
3. **Figures** — CR/SS/Accuracy plots saved to `figures/`

### Generate new logits on Kaggle GPU

Open `notebooks/kaggle_benchmark.ipynb` on Kaggle (GPU T4 x2), run all cells. Outputs saved to `outputs_kaggle/`.

---

## References

- Ye et al. (2024). *Benchmarking LLMs via Uncertainty Quantification*. NeurIPS.
- Tibshirani et al. (2019). *Conformal Prediction Under Covariate Shift*. NeurIPS.
- Sadinle et al. (2019). *Least Ambiguous Set-Valued Classifiers*. JASA.
- Romano et al. (2020). *Classification with Valid and Adaptive Coverage*. NeurIPS.
- Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Angelopoulos & Bates (2021). *Conformal Prediction: A Gentle Introduction*. FnTML.
