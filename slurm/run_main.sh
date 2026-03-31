#!/bin/bash
#SBATCH --job-name=gossiprobofl_main
#SBATCH --output=slurm/logs/main_%j.out
#SBATCH --error=slurm/logs/main_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=wmglab-gpu
#SBATCH --chdir=/work/lc478/GossipRoboFL

# Load environment — adjust module names for your cluster
# module load cuda/12.8
# module load anaconda3

eval "$(/hpc/home/lc478/miniconda3/bin/conda shell.bash hook)"
conda activate ml_env

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
