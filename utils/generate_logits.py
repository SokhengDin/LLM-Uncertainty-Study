import json
import os
import argparse
import pickle
import random

from tqdm import tqdm

import prompt as pt
from ollama_client import OllamaClient


FEW_SHOT_IDS = {
    "MMLU"               : [1, 3, 5, 7, 9],
    "HellaSwag"          : [1, 3, 5, 7, 9],
    "CosmosQA"           : [1, 3, 5, 7, 9],
    "Halu-OpenDialKG"    : [5, 7, 9],
    "Halu-CNN/DailyMail" : [9],
}

FEW_SHOT_RESERVE = 10  # indices 0-9 always kept for few-shot pool


def load_data(data_file, max_samples=None):
    data = json.load(open(data_file, "r"))
    if max_samples and max_samples > 0:
        few_shot_data  = data[:FEW_SHOT_RESERVE]
        remaining_data = data[FEW_SHOT_RESERVE:]
        if len(remaining_data) > max_samples:
            random.seed(42)
            remaining_data = random.sample(remaining_data, max_samples)
        data = few_shot_data + remaining_data
        print(f"Limited to {len(data)} total samples ({FEW_SHOT_RESERVE} few-shot + {len(remaining_data)} test)")
    return data


def get_fewshot_exps(data):
    src         = data[0]["source"]
    fewshot_ids = FEW_SHOT_IDS[src]
    exps        = []
    for idx in fewshot_ids:
        assert data[idx]["id"] == idx
        exps.append(data[idx])
    return exps


def format_example(example, prompt, with_answer=False):
    src = example["source"]
    if src == "MMLU":
        prompt += "Question: " + example["question"] + "\nChoices:\n"
    elif src in ("CosmosQA", "HellaSwag"):
        prompt += "Context: "  + example["context"]  + "\n"
        prompt += "Question: " + example["question"] + "\nChoices:\n"
    elif src == "Halu-OpenDialKG":
        prompt += "Dialogue: " + example["context"]  + "\n"
        prompt += "Question: " + example["question"] + "\nChoices:\n"
    elif src == "Halu-CNN/DailyMail":
        prompt += "Document: " + example["context"]  + "\n"
        prompt += "Question: " + example["question"] + "\nChoices:\n"
    else:
        raise NotImplementedError(f"Unsupported dataset: {src}")
    for k, v in example["choices"].items():
        prompt += k + ". " + str(v) + "\n"
    prompt += "Answer:"
    if with_answer:
        prompt += " " + example["answer"] + "\n"
    return prompt


def format_base_prompt(example, args, fewshot_exps=None):
    if args.few_shot == 0 and not args.cot:
        prompt = ""
    elif args.few_shot > 0 and not args.cot:
        prompt = ""
        for exp in fewshot_exps:
            prompt = format_example(exp, prompt, with_answer=True)
    elif args.few_shot == 0 and args.cot:
        prompt = pt.base_cot_prompt
    else:
        raise NotImplementedError("Unsupported method.")
    return {"id": example["id"], "prompt": format_example(example, prompt)}


def format_shared_prompt(example, args, fewshot_exps=None):
    if args.few_shot == 0 and not args.cot:
        prompt = pt.shared_zero_prompt
    elif args.few_shot > 0 and not args.cot:
        prompt = pt.shared_few_prompt
        for exp in fewshot_exps:
            prompt = format_example(exp, prompt, with_answer=True)
        prompt += "\nNow make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
    elif args.few_shot == 0 and args.cot:
        prompt = pt.shared_cot_prompt
    else:
        raise NotImplementedError("Unsupported method.")
    return {"id": example["id"], "prompt": format_example(example, prompt)}


def format_task_prompt(example, args, fewshot_exps=None):
    if args.few_shot == 0 and not args.cot:
        prompt = json.loads(pt.task_zero_prompt, strict=False)[example["source"]]
    elif args.few_shot > 0 and not args.cot:
        prompt = json.loads(pt.task_few_prompt, strict=False)[example["source"]]
        for exp in fewshot_exps:
            prompt = format_example(exp, prompt, with_answer=True)
        prompt += "\nNow make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
    elif args.few_shot == 0 and args.cot:
        prompt = json.loads(pt.task_cot_prompt, strict=False)[example["source"]]
    else:
        raise NotImplementedError("Unsupported method.")
    return {"id": example["id"], "prompt": format_example(example, prompt)}


def get_model_outputs(client, data, args):
    all_outputs = []
    desc = f"Generating logits ({args.prompt_method}, {args.few_shot}-shot)"
    for exp in tqdm(data, desc=desc):
        logits_options = client.get_choice_logits(exp["prompt"])
        all_outputs.append({"id": exp["id"], "logits_options": logits_options})
    return all_outputs


def main(args):
    all_data_files = (
        [args.file] if args.file != "xxx.json"
        else ["mmlu_10k.json", "cosmosqa_10k.json", "hellaswag_10k.json",
              "halu_dialogue.json", "halu_summarization.json"]
    )
    print(f"Data files: {all_data_files}")
    print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'ALL'}")

    client = OllamaClient(args.model)

    for file in all_data_files:
        print(f"\n{'='*50}")
        print(f"Processing {file}")
        print(f"{'='*50}")

        data         = load_data(os.path.join(args.data_path, file), args.max_samples)
        fewshot_exps = get_fewshot_exps(data) if args.few_shot > 0 else None

        format_fn = {
            "base":   format_base_prompt,
            "shared": format_shared_prompt,
            "task":   format_task_prompt,
        }[args.prompt_method]

        prompt_data = [format_fn(d, args, fewshot_exps=fewshot_exps) for d in data]
        print(f"Processing {len(prompt_data)} instances (including few-shot examples)")

        model_outputs = get_model_outputs(client, prompt_data, args)

        save_name  = args.model.split("/")[-1]
        save_name += "_" + file.split(".json")[0]
        save_name += "_" + args.prompt_method
        save_name += "_icl" + str(args.few_shot)
        if args.cot:
            save_name += "_cot"
        if args.max_samples > 0:
            save_name += f"_sample{args.max_samples}"

        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, save_name + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model_outputs, f)
        print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",         type=str,  default="qwen3.5:4b")
    parser.add_argument("--data_path",     type=str,  default="data")
    parser.add_argument("--file",          type=str,  default="xxx.json",   help="Dataset file (default runs all)")
    parser.add_argument("--prompt_method", type=str,  default="base",       help="base | shared | task")
    parser.add_argument("--output_dir",    type=str,  default="outputs_base")
    parser.add_argument("--few_shot",      type=int,  default=0)
    parser.add_argument("--cot",           action="store_true", default=False)
    parser.add_argument("--max_samples",   type=int,  default=200,          help="Max test samples (0 = all)")
    args = parser.parse_args()
    main(args)
