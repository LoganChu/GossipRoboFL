#!/bin/bash
#SBATCH --job-name=gossiprobofl_byz
#SBATCH --output=slurm/logs/byz_%j.out
#SBATCH --error=slurm/logs/byz_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=wmglab-gpu

# 12 experiments: 3 methods x 4 fractions
# Distributed across 4 GPUs (3 experiments per GPU, run sequentially per GPU)
#
#   method index = task_id / 4   (0=mean, 1=clipped_gossip, 2=ssclip)
#   fraction index = task_id % 4 (0=0.0, 1=0.1, 2=0.2, 3=0.3)
#
#   GPU 0: tasks  0-2  (mean f=0.0,0.1,0.2)
#   GPU 1: tasks  3-5  (mean f=0.3, clipped_gossip f=0.0,0.1)
#   GPU 2: tasks  6-8  (clipped_gossip f=0.2,0.3, ssclip f=0.0)
#   GPU 3: tasks  9-11 (ssclip f=0.1,0.2,0.3)

cd "$(dirname "$(realpath "$0")")/.."
mkdir -p slurm/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_env

METHODS=("mean" "clipped_gossip" "ssclip")
FRACTIONS=("0.0" "0.1" "0.2" "0.3")

run_experiment() {
    local task_id=$1
    local gpu_id=$2
    local method=${METHODS[$((task_id / 4))]}
    local fraction=${FRACTIONS[$((task_id % 4))]}
    local attack_enabled="true"
    [ "$fraction" = "0.0" ] && attack_enabled="false"

    echo "GPU $gpu_id | Task $task_id: method=$method fraction=$fraction"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
        experiment.name=byz_abl_${method}_f${fraction/./} \
        experiment.rounds=200 \
        data.num_clients=50 \
        gossip.aggregation=$method \
        attack.enabled=$attack_enabled \
        attack.type=sign_flip \
        attack.fraction=$fraction \
        logging.save_model_every=0 \
        >> slurm/logs/byz_${method}_f${fraction/./}.out 2>&1
    echo "GPU $gpu_id | Done: method=$method fraction=$fraction"
}

# Run 2 GPU groups at a time to stay within RAM limits
(run_experiment 0 0; run_experiment 1 0; run_experiment 2 0) &
(run_experiment 3 1; run_experiment 4 1; run_experiment 5 1) &
wait

(run_experiment 6 2; run_experiment 7 2; run_experiment 8 2) &
(run_experiment 9 3; run_experiment 10 3; run_experiment 11 3) &
wait
echo "All 12 Byzantine ablation experiments complete."
