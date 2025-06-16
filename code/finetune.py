# finetuning the normistral instruct 7b model
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
import transformers
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
import logging
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json, os
from argparse import ArgumentParser


def load_norquad(norquad_dir):
    # Load norquad dataset
    data_list = []  # Store all data samples in a single list

    for file_name, split in [('training_dataset_flattened.json', 'train'), 
                             ('validation_dataset_flattened.json', 'validation')]:
        file_path = os.path.join(norquad_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data['data']:
                for paragraph in entry['paragraphs']:
                    qas = [{'question': qa['question'], 'answers': qa['answers']} for qa in paragraph['qas']]
                    data_list.append({
                        'context': paragraph['context'],
                        'QAS': qas,
                        'split': split  # Add a split column
                    })
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({
        'context': [entry['context'] for entry in data_list],
        'QAS': [entry['QAS'] for entry in data_list],
        'split': [entry['split'] for entry in data_list]
    })
    
    return dataset


# helper function to print the number of trainable model parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


# helper function to format data instances 
def create_prompt_formats(sample):
    INTRO_BLURB = "Nedenfor er en instruksjon som beskriver en oppgave. Skriv et svar som fullfører forespørselen på en passende måte."
    INSTRUCTION_KEY = "### Instruct: Du må stille så mange forskjellige HV-spørsmål (hva, hvor, hvem, når, osv.) som mulig til teksten. Alle spørsmålene må ha et riktig svar i teksten som er kort og konsist, for eksempel en frase. Svaret må tas fra teksten akkurat som det står! Du kan IKKE omformulere svaret eller legge til dine egne ord.\n"
    RESPONSE_KEY = "### Output:"
    END_KEY = "### End"
    
    questions = [x['question'] for x in sample['QAS']]
    answers = [x['answers'][0]['text'] for x in sample['QAS']]  # Extract only the answer text

    QAS = ""
    for i in range(len(questions)):
        QAS += f"{i+1}. Spørsmål: {questions[i]}\nSvar:{answers[i]}\n\n"

    blurb = f"\n{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}"
    input_context = f"Teksten: {sample['context']}"
    response = f"{RESPONSE_KEY}\n{QAS}"
    end = f"{END_KEY}"
    
    parts = [part for part in [blurb, instruction, input_context, response, end] if part]

    formatted_prompt = "\n\n".join(parts)
    sample["text"] = formatted_prompt
    return sample


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logging.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logging.info(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizing a batch
    """
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )

# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int,seed, dataset):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    logging.info("Preprocessing dataset...")
    dataset = dataset.map(create_prompt_formats)#, batched=True)
    
    # Apply preprocessing to each batch of the dataset & and remove 'instruction', 'context', 'response', 'category' fields
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=['context', 'QAS'],
    )

    # Filter out samples that have input_ids exceeding max_length
    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset

def main():
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=3) 
    parser.add_argument("--lr", default=1e-04) 
    parser.add_argument("--rank", default=16) 
    parser.add_argument("--batchsize", default=4) 

    parser.add_argument("--datapath", default="/cluster/home/zoiab/thesis/NorQuAD/data/evaluation/all")
    parser.add_argument("--outpath", default="/cluster/projects/nn9851k/zoiab/fine-tune/")

    parser.add_argument("--seed", action="store", type=int, default=42)

    args = parser.parse_args()
    seed = args.seed
    logging.info(args)

    norquad_dir = args.datapath

    # confiure bits and bytes for efficient loading
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    # Load the model
    model_name='norallm/normistral-7b-warm-instruct'
    device_map = {"": 0}
    original_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          device_map=device_map,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          cache_dir="/cluster/projects/nn9851k/zoiab/cache")

    logging.info("Model loaded")

    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False,cache_dir="/cluster/projects/nn9851k/zoiab/cache")
    tokenizer.pad_token = tokenizer.eos_token

    ## Pre-process dataset
    max_length = get_max_length(original_model)
    logging.info(max_length)
    dataset = load_norquad(norquad_dir)
    logging.info("Dataset loaded")
    train_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset.filter(lambda x: x['split'] == 'train'))
    eval_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset.filter(lambda x: x['split'] == 'validation'))
    
    logging.info("Tokenizer loaded")

    original_model = prepare_model_for_kbit_training(original_model)

    config = LoraConfig(
        r=int(args.rank), #Rank
        lora_alpha=int(args.rank)*2,    # set alpha to 2*rank
        target_modules=[
            'q_proj',
            'k_proj',
            'v_proj',
            'dense'
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    original_model.gradient_checkpointing_enable()

    peft_model = get_peft_model(original_model, config)

    logging.info(print_number_of_trainable_model_parameters(peft_model))

    # Print the total number of parameters
    model_fp32 = AutoModelForCausalLM.from_pretrained("norallm/normistral-7b-warm-instruct")
    total_params = sum(p.numel() for p in model_fp32.parameters())
    logging.info(f"Total parameters (FP32 model): {total_params}")

    output_dir = args.outpath + f'peft-dialogue-summary-training-{str(int(time.time()))}'
    # output_dir = f'/cluster/projects/nn9851k/zoiab/fine-tune/peft-dialogue-summary-training-{str(int(time.time()))}'
    logging.info(f"Output directory: {output_dir}")

    peft_training_args = TrainingArguments(
        output_dir = output_dir,
        warmup_steps=1,
        per_device_train_batch_size=int(args.batchsize),
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=float(args.lr),
        num_train_epochs=int(args.epochs),
        optim="paged_adamw_8bit",
        logging_steps=50,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=50,
        do_eval=True,
        gradient_checkpointing=True,
        report_to="none",
        overwrite_output_dir = 'True',
        group_by_length=True,
    )

    peft_model.config.use_cache = False

    peft_trainer = transformers.Trainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=peft_training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    peft_trainer.train()

if __name__ == "__main__":
    main()