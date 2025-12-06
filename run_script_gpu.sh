#!/bin/bash -l
#SBATCH --job-name=mezo_splitlearning
#SBATCH --account=fl-het
#SBATCH --partition=interactive
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40g

echo "Starting Split Learning Job"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "================================"

# Environment setup
# spack load /ujqlkat  # Commented out to avoid potential conflicts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mezo
export WANDB_MODE=disabled

# Load CUDA 12.6.3 (compatible with PyTorch 2.7.1+cu126)
# Unload conflicting CUDA package first if loaded, but since this is a fresh script run,
# we will just ensure we load the specific hash.
spack unload cuda
spack load /jmc4bek
export CUDA_HOME=$(spack location -i /jmc4bek)

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
OPTIMIZER=${OPTIMIZER:-sgd}

BS=${BS:-64}
LR=${LR:-1e-4}
ZOO_LR=${ZOO_LR:-1e-4}
EPS=${EPS:-1e-3}
SEED=${SEED:-365}
TRAIN=${TRAIN:-67349}
DEV=${DEV:-872}
EVAL=${EVAL:-1821}
STEPS=${STEPS:-3000}
EVAL_STEPS=${EVAL_STEPS:-25}
TRAINER=${TRAINER:-zo}
MODE=${MODE:-ft}
MOMENTUM=${MOMENTUM:-0.9}
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora --lora_r 8 --lora_alpha 16"
fi
TAG=mezo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED

TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
esac

echo $TAG
echo "BS: $BS"
echo "LR: $LR"
echo "ZOO_LR: $ZOO_LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAINER: $TRAINER"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "MODE: $MODE"
echo "MOMENTUM: $MOMENTUM"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"

echo "Starting Training"

python run.py \
    --model_name $MODEL \
    --task_name $TASK \
    --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $TRAINER --client_optimizer zo --server_optimizer zo \
    --learning_rate $LR --client_learning_rate $ZOO_LR --server_learning_rate $ZOO_LR \
    --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS --lr_scheduler_type "constant" \
    --eval_steps $EVAL_STEPS \
    --eval_strategy steps \
    --optim $OPTIMIZER \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"