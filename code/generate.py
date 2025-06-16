# generate new data samples with fine-tuned model
import torch
from argparse import ArgumentParser
from datasets import Dataset
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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
from transformers import set_seed
import logging
import pandas as pd
import re, gzip, tarfile, time

seed = 42
set_seed(seed)


def load_norquad_contexts(norquad_dir):
    # load norquad to check that we don't generate with articles that are already in the dataset
    contexts = set()
    for file_name in ['training_dataset_flattened.json', 'test_dataset_flattened.json']:
        file_path = os.path.join(norquad_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data['data']:
                for paragraph in entry['paragraphs']:
                    contexts.add(paragraph['context'])
    return contexts


def load_dataset(file_path):
    # Load dataset and yield instances one at a time
    if file_path.endswith('newspaper_ocr_no.txt.gz'):
        with gzip.open(file_path,'rt', encoding='utf-8') as fin:
            for line in fin:
                yield line.strip()

    # load wiki
    elif file_path.endswith('wikipedia.tar.gz'):
        yield from load_wiki(file_path, 'nob.wikipedia.json', '/cluster/home/zoiab/thesis/NorQuAD/data/evaluation/all')
    
    # load norquad test set for evaluation with n-gram-based scores
    elif file_path.endswith('test_dataset_flattened.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data['data']:
                for paragraph in entry['paragraphs']:
                    yield paragraph['context']


def load_wiki(archive_path, target_file, norquad_dir):
    # Load NorQuAD contexts
    norquad_contexts = load_norquad_contexts(norquad_dir)
    data_path = '/cluster/projects/nn9851k/zoiab/data/'

    # check if the target file is already extracted
    if os.path.exists(data_path + target_file):
        pass
    else:
        # Extract the target file from the archive
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extract(target_file, path=data_path)

    # Load the JSON data from the extracted file and yield each article text
    with open(data_path + target_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
        for article in articles:
            text = article['text']
            words = re.findall(r'\w+', text)
            # set length for the articles to around 300-400 words as normal for NorQuAD wiki subset
            if len(words) >= 300 and len(words) > 400:
                truncated_text = ' '.join(words[:400])
                if truncated_text not in norquad_contexts:
                    yield truncated_text
            elif len(words) >= 300 and len(words) <= 400:
                if text not in norquad_contexts:
                    yield text


def extract_qas(text):
    qa_pairs = re.findall(r'(\d+)\.\s*Spørsmål:\s*(.*?)\n\s*Svar:\s*(.*)', text)
    questions = [q.strip() for _, q, _ in qa_pairs]
    answers = [a.strip() for _, _, a in qa_pairs]
    answers = [a.strip("<|im_end|>") for _, _, a in qa_pairs]
    answers = [re.sub(r'</s>.*', '', a) for a in answers]
    return questions, answers


def format_qas(questions, answers, article, text_id=1):
    # format generated question-answer pair as JSON for easier access
    qas_list = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
    output = {
        "text_id": text_id,
        "text": article,
        "qas": qas_list
    }
    return json.dumps(output, ensure_ascii=False)


def main():
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    )

    parser = ArgumentParser()

    parser.add_argument("--datapath", default="/cluster/projects/nn9851k/zoiab/newspaper_ocr_no.txt.gz") # or /cluster/projects/nn9851k/zoiab/2019_wikipedia.tar.gz
    parser.add_argument("--modeldir", default="/cluster/projects/nn9851k/zoiab/fine-tune/")
    parser.add_argument("--modelpath", default="peft-dialogue-summary-training-1743096269/")
    parser.add_argument("--outpath", default="")

    parser.add_argument("--checkpoint", default="checkpoint-476")
    parser.add_argument("--run_from", default=0)
    parser.add_argument("--num_instances", default=1000)

    parser.add_argument("--seed", action="store", type=int, default=42)

    args = parser.parse_args()
    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": 0}  # Keep this if using GPU 0
    # confiure bits and bytes for efficient loading
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )


    base_model_id = "norallm/normistral-7b-warm-instruct"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                                          device_map=device_map,
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True,
                                                          cache_dir="/cluster/projects/nn9851k/zoiab/cache")

    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True, use_fast=False)
    eval_tokenizer.pad_token = eval_tokenizer.eos_token

    MODEL_PATH = args.modeldir + args.modelpath + args.checkpoint
    ft_model = PeftModel.from_pretrained(base_model, MODEL_PATH, torch_dtype=torch.float16, is_trainable=False)

    data_subset = 'news' if args.datapath.endswith('newspaper_ocr_no.txt.gz') else 'wiki'

    if args.datapath.endswith('test_dataset_flattened.json'):
        filename = f'test_predictions{args.modelpath[-4:-1]}.jsonl'
    else:
        filename = f'{args.modeldir}{args.modelpath}gen{data_subset}_from{args.run_from}.jsonl'
    
    logging.info(f'Generation will be saved to {filename}')
    processing_times = []
    for i, line in enumerate(load_dataset(args.datapath)):
        start_time = time.time()
        if i < int(args.run_from):  # Skip until we reach the start point
            continue
        try:
            prompt = f"Instruct: Du må stille så mange forskjellige HV-spørsmål (hva, hvor, hvem, når, osv.) som mulig til teksten. Alle spørsmålene må ha et riktig svar i teksten som er kort og konsist, for eksempel en frase. Svaret må tas fra teksten akkurat som det står! Du kan IKKE omformulere svaret eller legge til dine egne ord. {line}\nOutput:"

            # Generate a new sample
            input_ids = eval_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

            instruct_model_outputs = ft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=450, num_beams=1))
            instruct_model_text_output = eval_tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)


            questions, answers = extract_qas(instruct_model_text_output)
            json_ready = format_qas(questions, answers, line, i+1)
            if len(questions) > 0:
                with open(filename, 'a', encoding='utf-8') as file:
                    file.write(json_ready + '\n')
            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            if i % 100 == 0:
                logging.info(f'{i} instances processed.')
                # get average processing time for 100 instances
                avg_processing_time = (sum(processing_times)) / len(processing_times)
                processing_times = []
                logging.info(f"Avg. Processing time for article: {avg_processing_time:.4f} seconds")
        
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            continue    # Skip problematic entries and continue processing       
        if i >= int(args.run_from) + int(args.num_instances):    # Stop after n lines
            break
        
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    main()
