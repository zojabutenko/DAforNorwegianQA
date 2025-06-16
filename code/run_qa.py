# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import time
import uuid
import random
import tempfile
import json
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from datasets import Dataset
from datasets import load_dataset
from datasets import concatenate_datasets

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    # AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from squad import SquadResult, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features, SquadExample
from early_stopping import EarlyStopping
from transformers.trainer_utils import is_main_process
# import wandb
from torch.optim import AdamW
from transformers import DebertaV2Tokenizer


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def ensure_unique_output_dir(args):
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        base_output_dir = args.output_dir.rstrip("/")  # Ensure clean path
        unique_id = int(time.time())  # Timestamp-based unique identifier
        args.output_dir = os.path.join(base_output_dir, f"run_{unique_id}")  # Create a subdirectory
        
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output directory: {args.output_dir}")

def merge_json_files(args):
    all_data = []

    # === Load original dataset ===
    original_file = os.path.join(args.data_dir, args.train_file)
    with open(original_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)["data"]
        all_data.extend(original_data)

    orig_qa_count = sum(
        len(paragraph["qas"]) for item in original_data for paragraph in item["paragraphs"]
    )

    flat_aug_qas = []

    # === Load augmentation from JSON files ===
    if args.augment_with_files:
        for aug_file in args.augment_with_files:
            with open(aug_file, "r", encoding="utf-8") as f:
                aug_data = json.load(f)["data"]
                for item in aug_data:
                    for paragraph in item["paragraphs"]:
                        context = paragraph["context"]
                        for qa in paragraph["qas"]:
                            if is_valid_qa_pair(context, qa):
                                flat_aug_qas.append({"context": context, "qa": qa})

    # === Load augmentation from Hugging Face dataset ===
    if args.augment_dataset_name:
        dataset = load_dataset(args.augment_dataset_name, 'no', split='train')
        for example in dataset:
            context = example["context"]
    
            answer_text = example["answers"]["text"]
            answer_start = example["answers"]["answer_start"]
    
            # Convert to list if not already
            if isinstance(answer_text, str):
                answer_text = [answer_text]
            if isinstance(answer_start, int):
                answer_start = [answer_start]
    
            # Only proceed if we have at least one valid answer
            if answer_text and answer_start:
                qa = {
                    "id": str(example.get("example_id", hash(example["question"] + context))),
                    "question": example["question"],
                    "answers": [
                        {"text": t, "answer_start": s}
                        for t, s in zip(answer_text, answer_start)
                        if isinstance(t, str) and isinstance(s, int)
                    ],
                    "is_impossible": False
                }
    
                if qa["answers"]:  # Make sure there's at least one valid answer
                    if is_valid_qa_pair(context, qa):
                        flat_aug_qas.append({"context": context, "qa": qa})

    logging.info(f"Original QA count: {orig_qa_count}, Augment QA total: {len(flat_aug_qas)}")

    # === Subsample or Ablation Replace ===
    if args.ablation > 0:
        # Ablation mode: replace part of original dataset
        total_target_size = orig_qa_count  # based on number of QA-pairs, not items
        replace_size = int(total_target_size * args.ablation)  # how many QA-pairs to replace

        logging.info(f"Ablation mode: replacing {replace_size} QA-pairs out of {total_target_size}.")

        if len(flat_aug_qas) < replace_size:
            logging.warning(
                f"Requested {replace_size} augmentation QA-pairs for ablation, but only {len(flat_aug_qas)} available."
            )
            replace_size = len(flat_aug_qas)

        random.seed(args.seed)
        random.shuffle(flat_aug_qas)
        selected_aug_qas = flat_aug_qas[:replace_size]

        # Randomly choose entries from original data to keep
        keep_size = total_target_size - replace_size
        original_qas_flat = []
        for item in original_data:
            for paragraph in item["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    original_qas_flat.append({"context": context, "qa": qa})

        if len(original_qas_flat) < keep_size:
            logging.warning(
                f"Requested to keep {keep_size} original QA-pairs, but only {len(original_qas_flat)} available."
            )
            keep_size = len(original_qas_flat)

        random.shuffle(original_qas_flat)
        selected_original_qas = original_qas_flat[:keep_size]

        # Combine selected original + augmentation
        combined_qas = selected_original_qas + selected_aug_qas
        random.shuffle(combined_qas)

        # === Reconstruct structured data ===
        context_to_qas = {}
        for item in combined_qas:
            ctx = item["context"]
            context_to_qas.setdefault(ctx, []).append(item["qa"])

        merged_data = [
            {"paragraphs": [{"context": ctx, "qas": qas}]} for ctx, qas in context_to_qas.items()
        ]

    elif args.augment_fraction > 0:
        # Standard augmentation (additive)
        random.seed(args.seed)
        random.shuffle(flat_aug_qas)
        subset_size = int(orig_qa_count * args.augment_fraction)
        if len(flat_aug_qas) < subset_size:
            logging.warning(
                f"Requested {subset_size} augmentation QA-pairs, but only {len(flat_aug_qas)} available."
            )
        flat_aug_qas = flat_aug_qas[:subset_size]

        # === Reconstruct structured data ===
        context_to_qas = {}
        for item in flat_aug_qas:
            ctx = item["context"]
            context_to_qas.setdefault(ctx, []).append(item["qa"])

        augment_data_structured = [
            {"paragraphs": [{"context": ctx, "qas": qas}]} for ctx, qas in context_to_qas.items()
        ]

        merged_data = original_data + augment_data_structured

    else:
        # No augmentation
        merged_data = original_data

    merged_filename = os.path.join(args.data_dir, "merged_dataset.json")
    with open(merged_filename, "w", encoding="utf-8") as f:
        json.dump({"data": merged_data}, f, ensure_ascii=False, indent=4)

    total_qa_count = sum(len(p["qas"]) for item in merged_data for p in item["paragraphs"])
    logging.info(f"Merged dataset saved: {merged_filename}, total paragraphs: {len(merged_data)}, total QA-pairs: {total_qa_count}")
    try:
        with open(merged_filename, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"JSON validation failed: {e}")
        
    return merged_filename


def is_valid_qa_pair(context, qa):
    return (
        isinstance(context, str) and context.strip() and
        isinstance(qa, dict) and
        isinstance(qa.get("question", ""), str) and qa["question"].strip() and
        isinstance(qa.get("answers", []), list) and len(qa["answers"]) > 0
    )


def merge_datasets(original, augment):
    """
    Merges a fraction of the augmentation dataset into the original dataset.
    Args:
        original: The original dataset.
        augment: The augmented dataset.
        fraction: Fraction of the augmentation dataset to use.
    Returns:
        A concatenated dataset.
    """
    for orig_tensor, aug_tensor in zip(original.tensors, augment.tensors):
        # Check the dimensions of the tensors and align if needed
        if orig_tensor.ndimension() != aug_tensor.ndimension():
            print(f"Original tensor shape: {orig_tensor.shape}, Augmented tensor shape: {aug_tensor.shape}")
            print(orig_tensor[0], aug_tensor[0])
            # Ensure tensors have the same dimensions by unsqueezing if necessary
            if orig_tensor.ndimension() == 1:
                orig_tensor = orig_tensor.unsqueeze(1)  # Add a feature dimension
            if aug_tensor.ndimension() == 1:
                aug_tensor = aug_tensor.unsqueeze(1)  # Add a feature dimension
        
    # Concatenate tensors along the batch dimension
    merged_tensors = [torch.cat([orig_tensor, aug_tensor], dim=0) 
                      for orig_tensor, aug_tensor in zip(original.tensors, augment.tensors)]
    
    # Return a new TensorDataset
    return torch.utils.data.TensorDataset(*merged_tensors)


def load_multiple_datasets(args, tokenizer, file_paths):
    """
    Process augmentation datasets using the same workflow as the original dataset.

    Args:
        args: The arguments object containing dataset and processing parameters.
        tokenizer: The tokenizer instance.
        file_paths: List of paths to JSON files containing augmentation data.

    Returns:
        TensorDataset: Tokenized and processed augmentation datasets.
    """
    all_datasets = []

    for file_path in file_paths:
        # Use the appropriate processor
        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()

        # Parse examples from the augmentation file
        if args.augmentation_dir:
            examples = processor.get_train_examples(data_dir=args.augmentation_dir, filename=file_path)
        else:
            examples = processor.get_train_examples(data_dir=None, filename=file_path)


        # Convert examples to features and process them
        _, tokenized_dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )


        all_datasets.append(tokenized_dataset)

        # print(tokenized_dataset.shape)

    # Concatenate all augmentation datasets
    if len(all_datasets) > 1:
        merged_tensors = [
            torch.cat([dataset.tensors[i] for dataset in all_datasets], dim=0)
            for i in range(len(all_datasets[0].tensors))
        ]
        return torch.utils.data.TensorDataset(*merged_tensors)
    elif len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return None


# Tokenize the augmentation dataset
def tokenize_augment_dataset(augment_dataset, tokenizer, args):
    # Convert the dataset into SquadExample format before tokenization
    examples = [
        SquadExample(
            qas_id=entry["id"],
            question_text=entry["question"],
            context_text=entry["context"],
            answer_text=entry["answers"]["text"][0] if entry["answers"]["text"] else "",
            start_position_character=entry["answers"]["answer_start"][0] if entry["answers"]["answer_start"] else 0,
            title=entry.get("title_en", "")
        )
        for entry in augment_dataset
    ]

    # Tokenize examples
    features, tokenized_dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=not evaluate,  
        return_dataset="pt",
        threads=args.threads,
    )
    return tokenized_dataset


def train(args, train_dataset, model, tokenizer):
    """Train the model"""

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.stop_early:
        early_stopper = EarlyStopping(mode='max', patience=3, min_delta=0.01, percentage=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    early_stop = False 

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            # wandb.log({"lr": scheduler.get_lr()[0], "loss": (tr_loss/global_step) }, step=global_step)


            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # if args.local_rank == -1 and args.evaluate_during_training:
                    #   results = evaluate(args, model, tokenizer)
                        # for key, value in results.items():
                        #     print(f"eval_{key}: {value}")
                    
                    if args.local_rank == -1 and args.evaluate_during_training:

                        results = evaluate(args, model, tokenizer)
                        em_score = results["exact"]  # Make sure 'exact' is the correct key
                        for key, value in results.items():
                            print(f"eval_{key}: {value}")
                        
                        if args.stop_early:
                            best = early_stopper.best if early_stopper.best is not None else 0.0
                            delta = em_score - best
                            print(f"EM: {em_score}, Best: {best}, Δ: {delta}, Bad epochs: {early_stopper.num_bad_epochs}")

                            if early_stopper.best is not None:
                                improved = early_stopper.is_better(em_score, early_stopper.best)
                            else:
                                improved = True  # Assume first score is best by default

                            # improved = early_stopper.is_better(em_score, early_stopper.best)

                            if early_stopper.step(em_score):
                                logger.info(f"Early stopping triggered. Best EM was {early_stopper.best}")
                                early_stop = True
                                break
                            
                            if improved:
                                logger.info(f"New best EM: {em_score}")

                        
                        logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
            
        if early_stop:
            break  

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs.to_tuple()]

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # XLNet and XLM use a more complex post-processing procedure
    if args.model_type in ["xlnet", "xlm"]:
        start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging,
        )
    else:
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer,
        )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Ensure unique cache filename
    input_dir = args.data_dir if args.data_dir else "."
    unique_id = args.experiment_id if hasattr(args, "experiment_id") else str(uuid.uuid4())[:8]
    

    # Load data features from cache or dataset file
    # input_dir = args.data_dir if args.data_dir else "."
    # cached_features_file = os.path.join(
    #     input_dir,
    #     "cached_{}_{}_{}_{}".format(
    #         args.domain,
    #         "dev" if evaluate else "train",
    #         list(filter(None, args.model_name_or_path.split("/"))).pop(),
    #         str(args.max_seq_length),
    #     ),
    # )

    cached_features_file = os.path.join(
        args.cache_dir,
        "cached_{}_{}_{}_{}_{}".format(
            args.domain,
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            unique_id,  # Append unique identifier
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )

    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warning("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)
    
        logging.info(f"transformers squad.py location: {transformers.__file__}")
        logging.info(f"squad_convert_examples_to_features: {squad_convert_examples_to_features.__module__}")


        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        # Augment the dataset by subsets if specified
        # if not evaluate and args.augment_fraction > 0:
        #     logger.info(f"Augmenting dataset with {args.augment_fraction*100}% of {args.augment_dataset_name}")
        #     if args.augment_dataset_name == 'alexandrainst/scandi-qa':
        #         augment_dataset = load_dataset(args.augment_dataset_name, 'no', split='train')
        #     else:
        #         augment_dataset = load_dataset(args.augment_dataset_name, split='train')

        #     # Take subset of augmentation dataset
        #     augment_size = int(len(augment_dataset) * args.augment_fraction)
        #     augment_dataset = augment_dataset.select(range(augment_size))  # Subset here

        #     augment_tokenized_dataset = tokenize_augment_dataset(augment_dataset, tokenizer, args)
        #     dataset = merge_datasets(dataset, augment_tokenized_dataset)

        # Process augmentation datasets if there are multiple
        # if args.augment_with_files:
        #     logger.info(f"Processing augmentation datasets: {', '.join(args.augment_with_files)}")

        #     backtranslation_dataset = load_multiple_datasets(args, tokenizer, args.augment_with_files)
        #     dataset = merge_datasets(dataset, backtranslation_dataset)


        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

        if args.local_rank == 0 and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help=(
            "The maximum total input sequence length after WordPiece tokenization. Sequences "
            "longer than this will be truncated, and sequences shorter than this will be padded."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=(
            "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length."
        ),
    )
    parser.add_argument(
        "--early_stopping_patience",
        default=3,
        type=int,
        help=(
            "How many evals to wait before stopping training"
        ),
    )
    parser.add_argument(
        "--early_stopping_metric", type=str, default="exact",
        help="Metric to base early stopping on"
    )
    parser.add_argument(
        "--stop_early", action="store_true", help="whether to implement early stopping"
    )
    # arguments for augmented data injection
    parser.add_argument(
        "--augment_dataset_name", type=str, default=None,
        help="The name of the dataset to augment with (via the datasets library)."
    )
    parser.add_argument(
        "--augment_fraction", type=float, default=0.0,
        help="The fraction of the augmentation dataset to inject into the original dataset."
    )
    # parser.add_argument(
    #     "--augment_steps", type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0],
    #     help="Fractions of augmented data to inject during training iterations."
    # )
    parser.add_argument(
        "--augment_with_files",
        nargs="*",  # Accepts zero or more file paths
        default=None,
        help="List of JSON files containing augmentation datasets."
    )
    parser.add_argument(
        "--augmentation_dir", type=str, default=None,
        help="The path to directory with augmentation files."
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help=(
            "If true, all of the warnings related to data processing will be printed. "
            "A number of warnings are expected for a normal SQuAD evaluation."
        ),
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help=(
            "language id of input for language-specific xlm models (see"
            " tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
        ),
    )
    parser.add_argument('--ablation', type=float, default=0.0,
                        help="Fraction of original dataset to replace with augmentation. 0 = no ablation.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--domain", type=str, default="", help="wiki, news or total")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help=(
            "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html"
        ),
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        logging.info('Output directory exists. Will create a new one for this run.')
        # raise ValueError(
        #     "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        #         args.output_dir
        #     )
        # )

    # instead of raising an error, create a new output directory
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    #     base_output_dir = args.output_dir
    #     counter = 1
    #     while os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #         args.output_dir = f"{base_output_dir}{counter}"
    #         counter += 1
    #     os.makedirs(args.output_dir)
    # logging.info(f"Output directory: {args.output_dir}")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    # if args.model_type == 'deberta-v2':
    #     tokenizer = DebertaV2Tokenizer.from_pretrained(
    #     args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #     use_fast=False,
    #     cache_dir=args.cache_dir
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    wconfig = {
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        'batch_size': args.per_gpu_train_batch_size * max(1, args.n_gpu),
        'epochs': args.num_train_epochs,
        'model': args.model_name_or_path,
        'domain': args.domain,
        'warmup_steps': args.warmup_steps

    }
    # wandb.init(project="norquad", config=wconfig, entity="zoia-butenko-university-of-oslo")

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


    # Training
    if args.do_train:
        # If augmentation is enabled, merge datasets first
        if args.augment_with_files or args.augment_dataset_name:
            merged_dataset_path = merge_json_files(args)
            args.train_file = merged_dataset_path

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))
    # wandb.log(results)

    return results


if __name__ == "__main__":
    main()
