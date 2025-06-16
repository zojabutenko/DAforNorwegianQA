#!/bin/bash -l

#SBATCH --account=nn9851k

#SBATCH --time=6:30:00
#SBATCH --partition=accel --gpus=1
#SBATCH --cpus-per-task=8  # More CPU threads to help with preprocessing
#SBATCH --mem=16G  

module purge
module use -a /cluster/projects/nn9851k/software/easybuild/modules
module load Python/3.8.6-GCCcore-10.2.0

# source /cluster/projects/nn9851k/zoiab/fewshot-env/bin/activate
source /cluster/projects/nn9851k/zoiab/training-venv/bin/activate

python3 few_shot.py \
    --num_instances=1000 \
    --run_from=0 \
    --model=norallm/normistral-7b-warm-instruct \
    --datapath=/cluster/home/zoiab/thesis/NorQuAD/data/evaluation/all/test_dataset_flattened.json \
    --outpath=/cluster/home/zoiab/thesis/few-shot \

# python3 few_shot.py --num_instances=1000 --run_from=10000 --prompt=2 --datapath=/cluster/projects/nn9851k/zoiab/2019_wikipedia.tar.gz --model=Mistral-7B-Instruct-v0.2
# python3 few_shot.py --num_instances=1000 --run_from=20000 --prompt=2 --model=norallm/normistral-7b-warm-instruct
#     --datapath=/cluster/projects/nn9851k/zoiab/2019_wikipedia.tar.gz \
