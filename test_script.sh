#!/bin/bash -l
#SBATCH --job-name=mezo_splitlearning
#SBATCH --account=fl-het
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --output=cb_prefix_ZOO_SGD/%x_%j.out
#SBATCH --error=cb_prefix_ZOO_SGD/%x_%j.err
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g

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

# MODEL_NAME="facebook/opt-125m"
# EPOCHS=1
# BATCH_SIZE=16  # Small for testing
# MAX_LENGTH=512  # Short for speed
# LR=1e-3
# ZOO_LR=1e-2
# EPS=1e-1        # ZOO epsilon (perturbation scale)
# SEED=0          # Random seed
# Defaults (can be overridden above/by CLI)
TRAIN=${TRAIN:-1000}      # Training examples
DEV=${DEV:-500}           # Dev examples
EVAL=${EVAL:-1000}        # Evaluation examples
STEPS=${STEPS:-4000}      # Training steps
EVAL_STEPS=${EVAL_STEPS:-1000} # Evaluation steps
# NUM_PERT=5

MODEL_NAME="${MODEL_NAME:-facebook/opt-125m}"
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}        # Smaller for stability
MAX_LENGTH=${MAX_LENGTH:-512}      # Reduced
# LR=5e-4             # SGD learning rate
# ZOO_LR=5e-4         # ZOO learning rate (reduced)
# EPS=1e-3            # Smaller perturbation scale
SEED=${SEED:-42}
# Task-specific sane defaults (CB is tiny)
TASK="${TASK:-sst2}"
TUNING="${TUNING:-prefix}"
# If CB and sizes not provided, pick CB-appropriate sizes
if [ "$TASK" = "cb" ]; then
    TRAIN=${TRAIN:-250}
    DEV=${DEV:-56}
    EVAL=${EVAL:-56}
fi

# Steps and eval cadence
STEPS=${STEPS:-4000}
EVAL_STEPS=${EVAL_STEPS:-500}

# ZOO/optimization knobs
NUM_PERT=${NUM_PERT:-10}         # More perturbations for better ZOO signal
NUM_PREFIX=${NUM_PREFIX:-10}
LR=${LR:-1e-3}          # Client SGD lr (stability)
ZOO_LR=${ZOO_LR:-5e-4}  # Server ZOO lr (apply FD grads)
EPS=${EPS:-5e-4}        # ZOO perturbation scale (finite-diff mu base; RMS-scaled)

# LR=${LR:-5.00E-06}
# ZOO_LR=${ZOO_LR:-5.00E-06}
# EPS=${EPS:-1.00E-03}
LORA_R=${LORA_R:-8}
# MODEL_NAME="facebook/opt-125m"
# EPOCHS=1
# BATCH_SIZE=16  # Small for testing
# MAX_LENGTH=512  # Short for speed
# LR=1e-3
# ZOO_LR=1e-1
# EPS=1e-1        # ZOO epsilon (perturbation scale)
# SEED=0          # Random seed
# TRAIN=1000      # Training examples
# DEV=500         # Dev examples
# EVAL=1000       # Evaluation examples
# STEPS=2000      # Training steps
# EVAL_STEPS=2000 # Evaluation steps
# NUM_PERT=5


echo "Configuration:"
echo "   Model: $MODEL_NAME"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Max Length: $MAX_LENGTH"
echo "   Learning Rate: $LR"
echo "   Task: $TASK"
echo "   Tuning: $TUNING"
echo "   Train/Dev/Eval: $TRAIN/$DEV/$EVAL"
echo "   Steps/EvalSteps: $STEPS/$EVAL_STEPS"
echo ""

# Start server in background
echo "Starting coordinator (server app)..."
python3 server.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --host $HOST \
    --port $PORT \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --zoo_lr $ZOO_LR \
    --seed $SEED \
    --use_zeroth_order \
    --mu $EPS \
    --train_examples $TRAIN \
    --dev_examples $DEV \
    --eval_examples $EVAL \
    --max_steps $STEPS \
    --eval_steps $EVAL_STEPS \
    --num_prefix $NUM_PREFIX \
    --task $TASK \
    --tuning $TUNING \
    --lora_r $LORA_R \
    --num_pert $NUM_PERT \
    --wire_fp16 off &

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
    --mu $EPS \
    --train_batch_size $BATCH_SIZE \
    --num_pert $NUM_PERT \
    --test_batch_size $BATCH_SIZE \
    --task $TASK \
    --lora_r $LORA_R \
    --tuning $TUNING \
    --max_steps $STEPS &

CLIENT_EXIT_CODE=$?

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
