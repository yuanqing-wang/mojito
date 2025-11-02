import re
from mojito.tokenizer import preprocess, add
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def tokenize(prompt, tokenizer):
    result = tokenizer(prompt, padding="max_length", max_length=128, truncation=True)
    result["labels"] = result["input_ids"].copy() # .clone()
    return result

def get_data(point, tokenizer):
    input_text = point['input']
    output_text = point['output']
    prompt = f"[INST]{input_text}[/INST]{output_text}"
    prompt = re.sub(r'(<SMILES>[^<;]*);([^<]*</SMILES>)', r'\1</SMILES> <SMILES>\2', prompt)    
    try:
        prompt = preprocess(prompt)
        prompt = prompt.replace("SMILES>", "MOJITO>")
    except Exception as e:
        print(f"Error processing prompt: {prompt}")
    point["preprocessed_prompt"] = prompt
    return tokenize(prompt, tokenizer)


def run(args):
    model = AutoModelForCausalLM.from_pretrained(args.model)
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    add(tokenizer)    
    model.resize_token_embeddings(len(tokenizer))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # target_modules=["embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        target_modules=["embed_tokens", "q_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        
    tasks = [
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
        # use_first=100,
    )
    
    dataset = dataset.shuffle().map(
        lambda x: get_data(x, tokenizer),
    )
    
    
    # define the training arguments
    training_args = TrainingArguments(
        output_dir="./results-large",
        per_device_train_batch_size=16,
        num_train_epochs=100,
        save_total_limit=2,
        fp16=True,
    )
    
    tokenizer.save_pretrained(training_args.output_dir)
    
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    
    trainer.train()
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    run(args)
