#!/bin/bash -l
#SBATCH --job-name=mezo_splitlearning
#SBATCH --account=fl-het
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80g

echo "Starting Split Learning Job"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "================================"

# Environment setup (uncomment and modify as needed)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mezo
# module load python/3.8
# module load cuda/11.7

echo "All required files present"

# Optional CLI overrides (e.g., --task cb --train 250 --dev 56 --eval 56)
while (( "$#" )); do
    case "$1" in
        --task) TASK="$2"; shift 2;;
        --train) TRAIN="$2"; shift 2;;
        --dev) DEV="$2"; shift 2;;
        --eval) EVAL="$2"; shift 2;;
        --steps) STEPS="$2"; shift 2;;
        --eval-steps) EVAL_STEPS="$2"; shift 2;;
        --batch|--batch_size|--train_batch_size) BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --zoo_lr) ZOO_LR="$2"; shift 2;;
        --mu|--eps) EPS="$2"; shift 2;;
        --tuning) TUNING="$2"; shift 2;;
        --num-prefix) NUM_PREFIX="$2"; shift 2;;
        --lora-r) LORA_R="$2"; shift 2;;
        --num-pert) NUM_PERT="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --model|--model_name) MODEL_NAME="$2"; shift 2;;
        # New overrides
        --weight-decay|--weight_decay) WEIGHT_DECAY="$2"; shift 2;;
        --scheduler) SCHEDULER="$2"; shift 2;;
        --warmup-steps|--warmup_steps) WARMUP_STEPS="$2"; shift 2;;
        --warmup-ratio|--warmup_ratio) WARMUP_RATIO="$2"; shift 2;;
        --sgd-accum|--sgd_accum_steps) SGD_ACCUM_STEPS="$2"; shift 2;;
        --client_sgd_warmup_steps) CLIENT_SGD_WARMUP_STEPS="$2"; shift 2;;
        --client_sgd_every) CLIENT_SGD_EVERY="$2"; shift 2;;
        # ZOO + scheduler fine-tuning
        --zoo-momentum|--zoo_momentum) ZOO_MOMENTUM="$2"; shift 2;;
        --zoo-accum|--zoo_accum_steps) ZOO_ACCUM_STEPS="$2"; shift 2;;
        --sched-factor|--sched_factor) SCHED_FACTOR="$2"; shift 2;;
        --sched-patience|--sched_patience) SCHED_PATIENCE="$2"; shift 2;;
        --sched-threshold|--sched_threshold) SCHED_THRESHOLD="$2"; shift 2;;
        --sched-threshold-mode|--sched_threshold_mode) SCHED_THRESHOLD_MODE="$2"; shift 2;;
        --sched-cooldown|--sched_cooldown) SCHED_COOLDOWN="$2"; shift 2;;
        --sched-min-lr|--sched_min_lr) SCHED_MIN_LR="$2"; shift 2;;
        *) break;;
    esac
done


# Network settings: choose a unique free port per job
HOST="127.0.0.1"
JOB_ID=${SLURM_JOB_ID:-$$}
# Derive a base port from job id to reduce collisions, then probe forward for a free one
BASE_PORT=$((20000 + (JOB_ID % 20000)))
PORT=$BASE_PORT

is_port_free() {
    # returns 0 if free
    ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$1$"
}

for i in $(seq 0 99); do
    CAND=$((BASE_PORT + i))
    if is_port_free "$CAND"; then
        PORT=$CAND
        break
    fi
done

echo "Using host $HOST and port $PORT"

# Defaults (can be overridden above/by CLI)
# TRAIN=${TRAIN:-1000}      # Training examples
# DEV=${DEV:-500}           # Dev examples
# EVAL=${EVAL:-1000}        # Evaluation examples
# STEPS=${STEPS:-3000}      # Training steps
# EVAL_STEPS=${EVAL_STEPS:-50} # Evaluation steps
STEPS=${STEPS:-2000}      # Training steps
EVAL_STEPS=${EVAL_STEPS:-25} # Evaluation steps


MODEL_NAME="${MODEL_NAME:-facebook/opt-125m}"
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-32}        # SST-2 uses 32 in prior run
# Reduce eval batch and context length to avoid OOM on BoolQ LoRA
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
MAX_LENGTH=${MAX_LENGTH:-512}      # Reduced from 512 to cut KV/cache size
SEED=${SEED:-365}
# Task-specific sane defaults
TASK="${TASK:-sst2}"
TUNING="${TUNING:-lora}"
OPT="ZOO_SGD"

# Create per-run log directory based on dataset and tuning, and redirect stdout/stderr
LOG_DIR="${TASK}_${TUNING}_${OPT}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/${SLURM_JOB_NAME:-split_learning}_${SLURM_JOB_ID:-$$}.out") 2> >(tee -a "$LOG_DIR/${SLURM_JOB_NAME:-split_learning}_${SLURM_JOB_ID:-$$}.err" >&2)

# ZOO/optimization knobs
# NUM_PERT=${NUM_PERT:-10}
# NUM_PREFIX=${NUM_PREFIX:-20}
# LR=${LR:-1e-3}
# ZOO_LR=${ZOO_LR:-5e-4}
# EPS=${EPS:-5e-4}
# LORA_R=${LORA_R:-8}
# ESTIMATOR=${ESTIMATOR:-central}

NUM_PERT=${NUM_PERT:-5}
NUM_PREFIX=${NUM_PREFIX:-20}
LR=${LR:-1e-3}
ZOO_LR=${ZOO_LR:-5e-6}
# EPS=${EPS:-5e-6}
EPS=${EPS:-1e-3}
LORA_R=${LORA_R:-8}
ESTIMATOR=${ESTIMATOR:-forward}

# New optimizer/scheduler defaults (baseline-like for SST-2)
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
# SCHEDULER=${SCHEDULER:-cosine}
SCHEDULER=${SCHEDULER:-linear}
WARMUP_STEPS=${WARMUP_STEPS:-0}
WARMUP_RATIO=${WARMUP_RATIO:-0.06}
SGD_ACCUM_STEPS=${SGD_ACCUM_STEPS:-1}
CLIENT_SGD_WARMUP_STEPS=${CLIENT_SGD_WARMUP_STEPS:-800}
CLIENT_SGD_EVERY=${CLIENT_SGD_EVERY:-1}
# Additional ZOO/scheduler knobs
# ZOO_MOMENTUM=${ZOO_MOMENTUM:-0.0}
ZOO_MOMENTUM=${ZOO_MOMENTUM:-0.0}
ZOO_ACCUM_STEPS=${ZOO_ACCUM_STEPS:-1}
SCHED_FACTOR=${SCHED_FACTOR:-0.5}
SCHED_PATIENCE=${SCHED_PATIENCE:-6}
SCHED_THRESHOLD=${SCHED_THRESHOLD:-1e-3}
SCHED_THRESHOLD_MODE=${SCHED_THRESHOLD_MODE:-rel}
SCHED_COOLDOWN=${SCHED_COOLDOWN:-1}
SCHED_MIN_LR=${SCHED_MIN_LR:-1e-6}


echo "Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Eval Batch Size: $EVAL_BATCH_SIZE"
echo "   Max Length: $MAX_LENGTH"
echo "   Learning Rate: $LR"
echo "   ZOO LR: $ZOO_LR"
echo "   ZOO mu (eps): $EPS"
echo "   Task: $TASK"
echo "   Tuning: $TUNING"
echo "   LoRA r: $LORA_R"
echo "   Num Perturbations: $NUM_PERT"
echo "   Seed: $SEED"
echo "   Train/Dev/Eval: $TRAIN/$DEV/$EVAL"
echo "   Steps/EvalSteps: $STEPS/$EVAL_STEPS"
echo "   Server Optimizer: $SERVER_OPT (wd=$WEIGHT_DECAY)"
echo "   Client Optimizer: $CLIENT_OPT"
echo "   Scheduler: $SCHEDULER (warmup_steps=$WARMUP_STEPS warmup_ratio=$WARMUP_RATIO)"
echo ""


echo "Starting coordinator (server app)..."
python3 server.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --host $HOST \
    --port $PORT \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $EVAL_BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --zoo_lr $ZOO_LR \
    --seed $SEED \
    --estimator $ESTIMATOR \
    --use_zeroth_order \
    --mu $EPS \
    --max_steps $STEPS \
    --eval_steps $EVAL_STEPS \
    --task $TASK \
    --tuning $TUNING \
    --lora_r $LORA_R \
    --num_pert $NUM_PERT \
    --wire_fp16 off \
    --weight_decay $WEIGHT_DECAY \
    --scheduler $SCHEDULER \
    --sched_factor $SCHED_FACTOR \
    --sched_patience $SCHED_PATIENCE \
    --sched_threshold $SCHED_THRESHOLD \
    --sched_threshold_mode $SCHED_THRESHOLD_MODE \
    --sched_cooldown $SCHED_COOLDOWN \
    --sched_min_lr $SCHED_MIN_LR \
    --warmup_steps $WARMUP_STEPS \
    --warmup_ratio $WARMUP_RATIO \
    --sched_step_on_eval \
    --sgd_accum_steps $SGD_ACCUM_STEPS \
    --zoo_momentum $ZOO_MOMENTUM \
    --zoo_accum_steps $ZOO_ACCUM_STEPS \
    --client_sgd_warmup_steps $CLIENT_SGD_WARMUP_STEPS \
    --client_sgd_every $CLIENT_SGD_EVERY &
    # --eval_on_cpu &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server initialization
echo "Waiting 45 seconds for server initialization..."
sleep 45

# Check if server is still running
if ! ps -p $SERVER_PID > /dev/null; then
    echo "Server failed to start or crashed"
    echo "Checking for error messages..."
    tail -20 split_learning_${SLURM_JOB_ID}.err 2>/dev/null || echo "No error file yet"
    exit 1
fi

echo "Server is running"

# Start client
echo "Starting learner (client app)..."
python3 client.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --host $HOST \
    --port $PORT \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --zoo_lr $ZOO_LR \
    --seed $SEED \
    --estimator $ESTIMATOR \
    --mu $EPS \
    --train_batch_size $BATCH_SIZE \
    --num_pert $NUM_PERT \
    --test_batch_size $EVAL_BATCH_SIZE \
    --task $TASK \
    --lora_r $LORA_R \
    --tuning $TUNING \
    --max_steps $STEPS \
    --weight_decay $WEIGHT_DECAY &

CLIENT_EXIT_CODE=$?

# Move log files into structured directory
if ls mezo_splitlearning_* 1> /dev/null 2>&1; then
    for f in mezo_splitlearning_*; do
        mv "$f" "$LOG_DIR/" 2>/dev/null || true
    done
fi

# Wait for server to complete
echo "Waiting for server to finish..."
wait $SERVER_PID
SERVER_EXIT_CODE=$?

echo ""
echo "Job Summary:"
echo "==============="
echo "Client Exit Code: $CLIENT_EXIT_CODE"
echo "Server Exit Code: $SERVER_EXIT_CODE"

if [ $SERVER_EXIT_CODE -eq 0 ] && [ $CLIENT_EXIT_CODE -eq 0 ]; then
    echo "Split learning completed successfully!"
else
    echo "Split learning failed"
    echo "Check the output files for details:"
    echo "   split_learning_${SLURM_JOB_ID}.out"
    echo "   split_learning_${SLURM_JOB_ID}.err"
fi

echo ""
echo "Job finished at $(date)"
