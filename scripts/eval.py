import torch
import pandas as pd
from peft import PeftConfig, PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import Dataset
from run import get_data, add
import re
from mojito.tokenizer import preprocess

def tokenize(prompt, tokenizer):
    result = tokenizer(prompt, truncation=True, return_tensors="pt")
    result["labels"] = result["input_ids"].clone()
    return result

def get_data(point, tokenizer):
    input_text = point['input']
    prompt = f"[INST]{input_text}[/INST]"
    prompt = re.sub(r'(<SMILES>[^<;]*);([^<]*</SMILES>)', r'\1</SMILES> <SMILES>\2', prompt)    
    try:
        prompt = preprocess(prompt)
        prompt = prompt.replace("SMILES>", "MOJITO>")
    except Exception as e:
        print(f"Error processing prompt: {prompt}")
    point["preprocessed_prompt"] = prompt
    return tokenize(prompt, tokenizer)

def extract_number(string):
    match = re.search(r"<NUMBER>(.*?)</NUMBER>", string)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def run():
    ckpt = "results/checkpoint-85000/"
    peft_cfg = PeftConfig.from_pretrained(ckpt)
    base = peft_cfg.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base)
    add(tokenizer)

    # Evaluate the model
    model = AutoModelForCausalLM.from_pretrained(base, device_map="cuda")
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ckpt) 

    tasks= [
        'property_prediction-esol',
        # 'property_prediction-lipo',
        # 'property_prediction-bbbp',
        # 'property_prediction-clintox',
        # 'property_prediction-hiv',
        # 'property_prediction-sider',
    ]

    dataset = load_dataset(
        'osunlp/SMolInstruct', 
        tasks=tasks, 
        trust_remote_code=True,
        split="train",
        use_first=100,
    )
    
    y = []
    y_hat = []

    for point in dataset:
        _y = extract_number(point['output'])
        point = get_data(point, tokenizer=tokenizer)

        point = {k: v.cuda() for k, v in point.items()}

        # Generate output using the model
        with torch.no_grad():
            output = model.generate(
                point["input_ids"],
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )[:, point["input_ids"].shape[-1]:]

        # Decode the generated output
        output = tokenizer.batch_decode(output, skip_special_tokens=True)
        _y_hat = extract_number(output[0])
        y.append(_y)
        y_hat.append(_y_hat)
        print(_y, _y_hat)
    
    import pdb; pdb.set_trace()
        


if __name__ == "__main__":
    run()
