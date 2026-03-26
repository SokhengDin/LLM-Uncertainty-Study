import json
import os
import argparse
import pickle
from tqdm import tqdm
import prompt as pt
from ollama_client import OllamaClient

few_shot_exp_ids = {
    "MMLU": [1, 3, 5, 7, 9],
    "HellaSwag": [1, 3, 5, 7, 9],
    "CosmosQA": [1, 3, 5, 7, 9],
    "Halu-OpenDialKG": [5, 7, 9],
    "Halu-CNN/DailyMail": [9]
}

def load_data(data_file, max_samples=None):
    data = json.load(open(data_file, "r"))
    if max_samples and max_samples > 0:

        few_shot_count = max(few_shot_exp_ids.values(), key=lambda x: max(x) if x else [0])[0] + 1 if few_shot_exp_ids else 0
        few_shot_count = max(10, few_shot_count)  
        
        # Keep few-shot examples, sample from the rest
        few_shot_data = data[:few_shot_count]
        remaining_data = data[few_shot_count:]
        
        if len(remaining_data) > max_samples:
            import random
            random.seed(42)
            sampled_remaining = random.sample(remaining_data, max_samples)
            data = few_shot_data + sampled_remaining
            print(f"Limited to {len(data)} total samples ({len(few_shot_data)} few-shot + {max_samples} test samples)")
    return data

def get_fewshot_exps(data):
    src = data[0]["source"]
    fewshot_exps = []
    for idx in few_shot_exp_ids[src]:
        fewshot_exps.append(data[idx])
        assert data[idx]["id"] == idx
    return fewshot_exps

def format_example(example, prompt, with_answer=False):
    if example["source"] == "MMLU":
        prompt += "Question: " + example["question"] + "\nChoices:\n"
    elif example["source"] == "CosmosQA":
        prompt += "Context: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    elif example["source"] == "HellaSwag":
        prompt += "Context: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    elif example["source"] == "Halu-OpenDialKG":
        prompt += "Dialogue: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    elif example["source"] == "Halu-CNN/DailyMail":
        prompt += "Document: " + example["context"] + "\n" + "Question: " + example["question"] + "\nChoices:\n"
    else:
        raise NotImplementedError("Not supported dataset.")
    for k, v in example["choices"].items():
        prompt += k + ". " + str(v) + "\n"
    prompt += "Answer:"
    if with_answer:
        prompt += " " + example["answer"] + "\n"   
    return prompt

def format_base_prompt(example, args, fewshot_exps=None):
    exp = {}
    exp["id"] = example["id"]
    if args.few_shot == 0 and not args.cot:
        prompt = ""
    elif args.few_shot > 0 and not args.cot:
        prompt = ""
        for fs_exp in fewshot_exps:
            prompt = format_example(fs_exp, prompt, with_answer=True)
    elif args.few_shot == 0 and args.cot:
        prompt = pt.base_cot_prompt
    else:
        raise NotImplementedError("Not supported method.")
    prompt = format_example(example, prompt)
    exp["prompt"] = prompt
    return exp

def format_shared_prompt(example, args, fewshot_exps=None):
    exp = {}
    exp["id"] = example["id"]
    if args.few_shot == 0 and not args.cot:
        prompt = pt.shared_zero_prompt
    elif args.few_shot > 0 and not args.cot:
        prompt = pt.shared_few_prompt
        for fs_exp in fewshot_exps:
            prompt = format_example(fs_exp, prompt, with_answer=True)
        prompt += "\nNow make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
    elif args.few_shot == 0 and args.cot:
        prompt = pt.shared_cot_prompt
    else:
        raise NotImplementedError("Not supported method.")
    prompt = format_example(example, prompt)
    exp["prompt"] = prompt
    return exp

def format_task_prompt(example, args, fewshot_exps=None):
    exp = {}
    exp["id"] = example["id"]
    if args.few_shot == 0 and not args.cot:
        pt_dict = json.loads(pt.task_zero_prompt, strict=False)
        prompt = pt_dict[example["source"]]
    elif args.few_shot > 0 and not args.cot:
        pt_dict = json.loads(pt.task_few_prompt, strict=False)
        prompt = pt_dict[example["source"]]
        for fs_exp in fewshot_exps:
            prompt = format_example(fs_exp, prompt, with_answer=True)
        prompt += "\nNow make your best effort and select the correct answer for the following question. You only need to output the option.\n\n"
    elif args.few_shot == 0 and args.cot:
        pt_dict = json.loads(pt.task_cot_prompt, strict=False)
        prompt = pt_dict[example["source"]]
    else:
        raise NotImplementedError("Not supported method.")
    prompt = format_example(example, prompt)
    exp["prompt"] = prompt
    return exp

def get_model_outputs(client, data, args):
    all_outputs = []
    
    for idx, exp in enumerate(tqdm(data, desc=f"Generating logits ({args.prompt_method}, {args.few_shot}-shot)")):
        prompt = exp["prompt"]
        logits_options = client.get_choice_logits(prompt, args.num_samples)
        
        out = {}
        out["id"] = exp["id"]
        out["logits_options"] = logits_options
        all_outputs.append(out)
    
    return all_outputs

def main(args):
    if args.file != "xxx.json":
        all_data_files = [args.file]
    else:
        all_data_files = ['mmlu_10k.json', 'cosmosqa_10k.json', 'hellaswag_10k.json', 'halu_dialogue.json', 'halu_summarization.json']
    print(f"Data files: {all_data_files}")
    print(f"Max samples per file: {args.max_samples if args.max_samples > 0 else 'ALL'}")
    
    client = OllamaClient(args.model)
    
    for file in all_data_files:
        print(f"\n{'='*50}")
        print(f"Processing {file}")
        print(f"{'='*50}")
        
        data = load_data(os.path.join(args.data_path, file), args.max_samples)
        
        if args.few_shot > 0:
            fewshot_exps = get_fewshot_exps(data)
        else:
            fewshot_exps = None
        
        prompt_data = []
        for datum in data:
            if args.prompt_method == "base":
                prompt_data.append(format_base_prompt(datum, args, fewshot_exps=fewshot_exps))
            elif args.prompt_method == "shared":
                prompt_data.append(format_shared_prompt(datum, args, fewshot_exps=fewshot_exps))
            elif args.prompt_method == "task":
                prompt_data.append(format_task_prompt(datum, args, fewshot_exps=fewshot_exps))
        
        print(f"Processing {len(prompt_data)} instances (including few-shot examples)")
        model_outputs = get_model_outputs(client, prompt_data, args)
        
        save_file = args.model.split("/")[-1] + "_" + file.split(".json")[0] + "_" + args.prompt_method
        save_file += "_icl" + str(args.few_shot)
        if args.cot:
            save_file += "_cot"
        if args.max_samples > 0:
            save_file += f"_sample{args.max_samples}"
        save_file = os.path.join(args.output_dir, save_file)
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(save_file + ".pkl", "wb") as f:
            pickle.dump(model_outputs, f)
        print(f"Saved to {save_file}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="qwen3.5:0.8b")
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--file', type=str, default="xxx.json", help="Specify which dataset to use")
    parser.add_argument('--prompt_method', type=str, default="base", help="Select from 'base', 'shared', 'task'")
    parser.add_argument('--output_dir', type=str, default='outputs_base')
    parser.add_argument('--few_shot', type=int, default=0)
    parser.add_argument('--cot', action="store_true", default=False)
    parser.add_argument('--num_samples', type=int, default=50, help="Number of samples for probability estimation")
    parser.add_argument('--max_samples', type=int, default=200, help="Maximum number of test samples to process (0 for all)")
    args = parser.parse_args()
    
    main(args)