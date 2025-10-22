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
    result = tokenizer(prompt, padding="max_length", max_length=1024, return_tensors="pt")
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

def run():
    ckpt = "results/"
    peft_cfg = PeftConfig.from_pretrained(ckpt)
    base = peft_cfg.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base)
    add(tokenizer)

    # Evaluate the model
    model = AutoModelForCausalLM.from_pretrained(base)
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, ckpt) 

    tasks= [
        'property_prediction-esol',
        'property_prediction-lipo',
        'property_prediction-bbbp',
        'property_prediction-clintox',
        'property_prediction-hiv',
        'property_prediction-sider',
    ]

    dataset = load_dataset(
        'osunlp/SMolInstruct', 
        tasks=tasks, 
        trust_remote_code=True,
        split="train",
        use_first=100,
    )

    point = dataset[0]
    point = get_data(point, tokenizer=tokenizer)

    # Generate output using the model
    with torch.no_grad():
        outputs = model.generate(
            point["input_ids"],
            max_new_tokens=500,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated output
    generated_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print("Generated Output:", generated_output)


if __name__ == "__main__":
    run()
