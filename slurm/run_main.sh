#!/bin/bash
#SBATCH --job-name=gossiprobofl_main
#SBATCH --output=slurm/logs/main_%j.out
#SBATCH --error=slurm/logs/main_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=wmglab-gpu

mkdir -p slurm/logs

# Load environment — adjust module names for your cluster
# module load cuda/12.8
# module load anaconda3

conda activate ml_env

cd $SLURM_SUBMIT_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python main.py \
    experiment.name=main_n50 \
    experiment.rounds=200 \
    data.num_clients=50 \
    gossip.aggregation=ssclip \
    attack.enabled=false

echo "Done: main run"
