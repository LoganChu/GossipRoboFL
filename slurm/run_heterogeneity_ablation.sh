#!/bin/bash
#SBATCH --job-name=gossiprobofl_het
#SBATCH --output=slurm/logs/het_%A_%a.out
#SBATCH --error=slurm/logs/het_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=wmglab-gpu
#SBATCH --array=0-4
#SBATCH --chdir=/work/lc478/GossipRoboFL

# 5 tasks:
#   0-3: alpha ablation (alpha in {0.1, 0.5, 1.0, 10.0}), homogeneous clients
#   4:   heterogeneous clients, alpha=0.5 (task 1 serves as its homogeneous control)

eval "$(/hpc/home/lc478/miniconda3/bin/conda shell.bash hook)"
conda activate ml_env
ALPHAS=("0.1" "0.5" "1.0" "10.0")

if [ $SLURM_ARRAY_TASK_ID -lt 4 ]; then
    ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
    HET="false"
    NAME=noniid_a${ALPHA/./}_hom
else
    ALPHA="0.5"
    HET="true"
    NAME=noniid_a05_het
fi

echo "Task $SLURM_ARRAY_TASK_ID: alpha=$ALPHA heterogeneous=$HET"

python main.py \
    experiment.name=$NAME \
    experiment.rounds=200 \
    data.num_clients=50 \
    data.alpha=$ALPHA \
    client.heterogeneous=$HET \
    gossip.aggregation=ssclip \
    attack.enabled=false \
    logging.save_model_every=0 \
    logging.topo_snap_every=0

echo "Done: alpha=$ALPHA heterogeneous=$HET"
