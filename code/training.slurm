#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=nn9851k
#SBATCH --time=06:00:00
#SBATCH --partition=accel --gpus=1
#SBATCH --cpus-per-task=16  # More CPU threads to help with preprocessing
#SBATCH --mem=16G  

source ${HOME}/.bashrc

# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

# sanity: exit on all errors and disallow unset environment variables
set -o errexit
set -o nounset

module purge


source /cluster/projects/nn9851k/zoiab/training-venv/bin/activate

# export PS1=\$

# wandb login d6526f972f7c7701b44a54018f3629756dc44d1d
export HF_DATASETS_CACHE=/cluster/projects/nn9851k/zoiab/hf_cache

python /cluster/home/zoiab/thesis/NorQuAD/evaluation/run_qa.py \
    --model_type=bert \
    --model_name_or_path=NbAiLab/nb-bert-base \
    --tokenizer_name=NbAiLab/nb-bert-base \
    --output_dir=/cluster/projects/nn9851k/zoiab/data/few-shot/results \
    --data_dir=/cluster/home/zoiab/thesis/NorQuAD/data/evaluation/all \
    --train_file=training_dataset_flattened.json \
    --predict_file=test_dataset_flattened.json \
    --cache_dir=/cluster/projects/nn9851k/zoiab/ \
    --do_train \
    --do_eval \
    --do_lower_case \
    --per_gpu_train_batch_size=16 \
    --per_gpu_eval_batch_size=16 \
    --learning_rate=5e-05 \
    --num_train_epochs=3 \
    --gradient_accumulation_steps=1 \
    --seed=42 \
    --overwrite_cache \
    --overwrite_output_dir \
    --evaluate_during_training \
    --augment_fraction=0.5 \
    --augment_with_files /cluster/projects/nn9851k/zoiab/data/few-shot/squad_formatted_norMistral_news_p2.json /cluster/projects/nn9851k/zoiab/data/few-shot/squad_formatted_norMistral_wiki_p2.json \

    # --augment_with_files /cluster/projects/nn9851k/zoiab/fine-tune/peft-dialogue-summary-training-1744561326/squad_formatted_ft_1744561326_news.json /cluster/projects/nn9851k/zoiab/fine-tune/peft-dialogue-summary-training-1744561326/squad_formatted_ft_1744561326_wiki.json\

    # --augment_with_files /cluster/projects/nn9851k/zoiab/fine-tune/peft-dialogue-summary-training-1743588087/squad_formatted_ft_1743588087_wiki.json /cluster/projects/nn9851k/zoiab/fine-tune/peft-dialogue-summary-training-1743588087/squad_formatted_ft_1743588087_news.json \


    # --augment_dataset_name=alexandrainst/scandi-qa \







    # --augment_with_files /cluster/projects/nn9851k/zoiab/data/is_train_NQiI_lower.json \



