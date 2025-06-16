#!/bin/bash -l

#SBATCH --account=nn9851k

#SBATCH --time=5:00:00
#SBATCH --partition=accel --gpus=1
#SBATCH --cpus-per-task=8  # More CPU threads to help with preprocessing
#SBATCH --mem=16G  

module purge
module use -a /cluster/projects/nn9851k/software/easybuild/modules
module load Python/3.8.6-GCCcore-10.2.0

# source /cluster/projects/nn9851k/zoiab/ftenv/bin/activate
source /cluster/projects/nn9851k/zoiab/training-venv/bin/activate


python3 generate.py \
    --datapath=/cluster/home/zoiab/thesis/NorQuAD/data/evaluation/all/test_dataset_flattened.json \
    --modelpath=peft-dialogue-summary-training-1743588087/ \
    --num_instances=1000 \
    --outpath=/cluster/home/zoiab/thesis/fine-tune \
    --checkpoint=checkpoint-476 \
    --run_from=0 \

    # --modelpath=peft-dialogue-summary-training-1744561326/ \
