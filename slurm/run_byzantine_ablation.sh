#!/bin/bash
#SBATCH --job-name=gossiprobofl_byz
#SBATCH --output=slurm/logs/byz_%A_%a.out
#SBATCH --error=slurm/logs/byz_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=TODO_PARTITION_NAME
#SBATCH --array=0-11

# 12 tasks: 3 methods x 4 fractions
# Task IDs:
#   method index = TASK_ID / 4       (0=mean, 1=clipped_gossip, 2=ssclip)
#   fraction index = TASK_ID % 4     (0=0.0, 1=0.1, 2=0.2, 3=0.3)

mkdir -p slurm/logs

conda activate ml_env
cd $SLURM_SUBMIT_DIR

METHODS=("mean" "clipped_gossip" "ssclip")
FRACTIONS=("0.0" "0.1" "0.2" "0.3")

METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 4))]}
FRACTION=${FRACTIONS[$((SLURM_ARRAY_TASK_ID % 4))]}

echo "Task $SLURM_ARRAY_TASK_ID: method=$METHOD fraction=$FRACTION"

ATTACK_ENABLED="true"
if [ "$FRACTION" = "0.0" ]; then
    ATTACK_ENABLED="false"
fi

python main.py \
    experiment.name=byz_abl_${METHOD}_f${FRACTION/./} \
    experiment.rounds=200 \
    data.num_clients=50 \
    gossip.aggregation=$METHOD \
    attack.enabled=$ATTACK_ENABLED \
    attack.type=sign_flip \
    attack.fraction=$FRACTION \
    logging.save_model_every=0

echo "Done: method=$METHOD fraction=$FRACTION"
