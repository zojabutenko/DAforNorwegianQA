#!/bin/bash -l

#SBATCH --account=nn9851k

#SBATCH --partition=accel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=13:00:00


module purge
module use -a /cluster/projects/nn9851k/software/easybuild/modules
module load Python/3.8.6-GCCcore-10.2.0

source /cluster/projects/nn9851k/zoiab/ftenv/bin/activate

# only run training
python3 finetune.py --rank=32 --batchsize=1 --lr=2e-4 --epochs=3