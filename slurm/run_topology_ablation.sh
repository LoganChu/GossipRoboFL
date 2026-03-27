#!/bin/bash
#SBATCH --job-name=gossiprobofl_topo
#SBATCH --output=slurm/logs/topo_%A_%a.out
#SBATCH --error=slurm/logs/topo_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=TODO_PARTITION_NAME
#SBATCH --array=0-4

# 5 tasks: comm_range in {0.2, 0.3, 0.4, 0.5, 0.6}

mkdir -p slurm/logs

conda activate ml_env
cd $SLURM_SUBMIT_DIR

RANGES=("0.2" "0.3" "0.4" "0.5" "0.6")
RANGE=${RANGES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: comm_range=$RANGE"

python main.py \
    experiment.name=topo_abl_r${RANGE/./} \
    experiment.rounds=200 \
    data.num_clients=50 \
    gossip.aggregation=ssclip \
    topology.comm_range=$RANGE \
    attack.enabled=false \
    logging.save_model_every=0

echo "Done: comm_range=$RANGE"
