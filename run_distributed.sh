#!/bin/bash -l
#SBATCH --job-name=split_distributed
#SBATCH --account=fl-het
#SBATCH --partition=tier3
# Multi-GPU: 2x A100 40GB per node enables BS=64 with FO training
# Memory is distributed across GPUs via device_map="auto"
#SBATCH --gres=gpu:a100:3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=2-10:00:00
#SBATCH --nodes=2                    # 2 nodes: server + client
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g

# =============================================================================
# Distributed Split Learning - Using run_distributed.py (DeComFL-style)
# 
# This script runs split learning across two nodes using a SINGLE entry point:
#   - Node 0: python run_distributed.py --role server
#   - Node 1: python run_distributed.py --role client
#
# Benefits of single entry point (like DeComFL):
#   - Same script for both roles
#   - Consistent argument handling
#   - Easier maintenance
#
# The script:
#   1. Starts the server on node 0 with --role server
#   2. Waits for server to be ready (checks port availability)
#   3. Starts the client on node 1 with --role client
#   4. Client drives training, server responds
#   5. Cleans up when training completes
#
# Usage:
#   sbatch run_distributed.sh
#   
#   # With custom parameters:
#   MODEL=facebook/opt-1.3b TASK=SST2 BS=32 sbatch run_distributed.sh
#   
#   # Different optimizer modes:
#   CLIENT_OPTIMIZER=zo SERVER_OPTIMIZER=zo sbatch run_distributed.sh  # ZO/ZO (default)
#   CLIENT_OPTIMIZER=fo SERVER_OPTIMIZER=fo sbatch run_distributed.sh  # FO/FO
#   CLIENT_OPTIMIZER=zo SERVER_OPTIMIZER=fo sbatch run_distributed.sh  # Hybrid
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "  Distributed Split Learning"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# =============================================================================
# Configuration Parameters (same as run_script_gpu.sh)
# =============================================================================

# =============================================================================
# MEMORY PROFILING MODE: Set MEMORY_TEST=1 for quickest run to measure GPU memory
# Example: MEMORY_TEST=1 sbatch run_distributed.sh
# =============================================================================
MEMORY_TEST=${MEMORY_TEST:-0}

MODEL=${MODEL:-facebook/opt-125m}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
MOMENTUM=${MOMENTUM:-0.9}
TASK=${TASK:-BoolQ}
SEED=${SEED:-0}

if [ "$MEMORY_TEST" == "1" ]; then
    echo ">>> MEMORY TEST MODE: Running minimal config to measure GPU memory <<<"
    BS=${BS:-64}           # Smallest batch size
    TRAIN=${TRAIN:-}     # Minimal training examples
    DEV=${DEV:-0}         # No dev set
    EVAL=${EVAL:-0}       # No evaluation
    # STEPS=${STEPS:-85}     # Just 2 steps to see memory usage
    # STEPS=${STEPS:-4} 
    # STEPS=${STEPS:-39}
    STEPS=${STEPS:-9} 
    EVAL_STEPS=${EVAL_STEPS:-500}  # Effectively disable eval
else
    BS=${BS:-64}
    TRAIN=${TRAIN:-}  # Will be set per-task in case statement below
    DEV=${DEV:-0}
    EVAL=${EVAL:-}
    STEPS=${STEPS:-}
    EVAL_STEPS=${EVAL_STEPS:-25}
fi

MODE=${MODE:-ft}
MOMENTUM=${MOMENTUM:-0.9}

# ZO Configuration
ZO_VARIANT=${ZO_VARIANT:-central}
ZO_PERTURBATION=${ZO_PERTURBATION:-layer}
NUM_PERT=${NUM_PERT:-10}

# Split layer configuration (for OPT/GPT-2 models)
SPLIT_LAYER=${SPLIT_LAYER:-0}  # Layer index where to split the model (default: 3)

# Optimizer configuration (default: ZO/FO)
# Uses environment variables if passed via sbatch --export, otherwise defaults
CLIENT_OPTIMIZER=${CLIENT_OPTIMIZER:-zo}
SERVER_OPTIMIZER=${SERVER_OPTIMIZER:-zo}
OPTIMIZER=${OPTIMIZER:-sgd}

# Set trainer based on optimizer (zo if either is zo, else regular)
if [ "$CLIENT_OPTIMIZER" == "zo" ] || [ "$SERVER_OPTIMIZER" == "zo" ]; then
    TRAINER=zo
else
    TRAINER=regular
fi

# Debug output
echo "Optimizer configuration:"
echo "  CLIENT_OPTIMIZER: $CLIENT_OPTIMIZER"
echo "  SERVER_OPTIMIZER: $SERVER_OPTIMIZER"
echo "  TRAINER: $TRAINER"

# Learning rates
if [ "$MODE" == "lora" ]; then
    # LR=${LR:-5e-4}
    LR=${LR:-1e-3}
    # LR=${LR:-1e-4}
    ZOO_LR=${ZOO_LR:-5e-5}
    # ZOO_LR=${ZOO_LR:-1e-6}
    EPS=${EPS:-5e-3}
else
    LR=${LR:-1e-3}
    # LR=${LR:-5e-4}
    # LR=${LR:-1e-4}
    ZOO_LR=${ZOO_LR:-1e-6}
    EPS=${EPS:-1e-3}
fi

# Set learning rates based on optimizer type
if [ "$CLIENT_OPTIMIZER" == "zo" ]; then
    CLIENT_LR=${CLIENT_LR:-$ZOO_LR}
else
    CLIENT_LR=${CLIENT_LR:-$LR}
fi

if [ "$SERVER_OPTIMIZER" == "zo" ]; then
    SERVER_LR=${SERVER_LR:-$ZOO_LR}
else
    SERVER_LR=${SERVER_LR:-$LR}
fi

# Network configuration
# Use dynamic port based on SLURM_JOB_ID to avoid port conflicts when multiple jobs run on same node
# Port range: 50000-59999 (10000 ports available)
if [ -n "$SLURM_JOB_ID" ]; then
    BASE_PORT=50000
    PORT_OFFSET=$((SLURM_JOB_ID % 10000))
    DEFAULT_PORT=$((BASE_PORT + PORT_OFFSET))
else
    DEFAULT_PORT=50051
fi
PORT=${PORT:-$DEFAULT_PORT}

# LoRA/Prefix configuration
EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    EXTRA_ARGS="--lora --lora_r 8 --lora_alpha 16"
fi

TASK_ARGS=""
# Skip task-specific overrides in memory test mode (values already set above)
if [ "$MEMORY_TEST" != "2" ]; then
    case $TASK in
        BoolQ)
            TRAIN=${TRAIN:-9427}    # Full training set: 9,427 examples
            if [ "$MODE" == "lora" ]; then
                STEPS=${STEPS:-500}
            else
                STEPS=${STEPS:-1500}
            fi
            EVAL=${EVAL:-3270}
            ;;
        RTE)
            TRAIN=${TRAIN:-2490}    # Full training set: 2,490 examples
            STEPS=${STEPS:-1500}
            EVAL=${EVAL:-1000}
            ;;
        CB) # Small dataset - validation only has 56 examples
            TRAIN=${TRAIN:-250}     # Full training set: 250 examples
            DEV=100                   # Dont carve dev from train; eval uses validation set
            STEPS=${STEPS:-1000}
            EVAL=${EVAL:-1000}
            ;;
        WIC)
            TRAIN=${TRAIN:-5428}    # Full training set: 5,428 examples
            if [ "$MODE" == "lora" ]; then
                STEPS=${STEPS:-2000}
            else
                STEPS=${STEPS:-2000}
            fi
            EVAL=${EVAL:-1000}
            ;;
        WSC)
            TRAIN=${TRAIN:-554}     # Full training set: 554 examples
            if [ "$MODE" == "lora" ]; then
                STEPS=${STEPS:-3000}
            else
                STEPS=${STEPS:-1000}
            fi
            EVAL=${EVAL:-1000}
            ;;
        SST2)
            TRAIN=${TRAIN:-67349}   # Full training set: 67,349 examples
            if [ "$MODE" == "lora" ]; then
                STEPS=${STEPS:-2000}
            else
                STEPS=${STEPS:-3000}
            fi
            EVAL=${EVAL:-1000}
            ;;
        *)
            # Default for unknown tasks
            TRAIN=${TRAIN:-1000}
            STEPS=${STEPS:-1000}
            ;;
    esac
fi

MAX_WAIT=${MAX_WAIT:-300}  # Max seconds to wait for server
# =============================================================================
# Get Node Information
# =============================================================================

NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
SERVER_NODE=${NODELIST[0]}
CLIENT_NODE=${NODELIST[1]}

# Get server IP (run hostname on server node)
SERVER_IP=$(srun --nodes=1 --ntasks=1 -w $SERVER_NODE hostname -i 2>/dev/null | head -1 | awk '{print $1}')

echo ""
echo "=========================================="
echo "  Configuration Summary"
echo "=========================================="
if [ "$MEMORY_TEST" == "1" ]; then
    echo ">>> MEMORY TEST MODE ENABLED <<<"
    echo ""
fi
echo "SERVER NODE: $SERVER_NODE"
echo "SERVER IP:   $SERVER_IP:$PORT"
echo "CLIENT NODE: $CLIENT_NODE"
echo ""
echo "Model:       $MODEL_NAME"
echo "Task:        $TASK"
echo "Mode:        $MODE"
echo "Batch Size:  $BS"
echo "Max Steps:   $STEPS"
echo "Eval Steps:  $EVAL_STEPS"
echo "Trainer:     $TRAINER"
echo ""
echo "Client:      $CLIENT_OPTIMIZER (LR=$CLIENT_LR)"
echo "Server:      $SERVER_OPTIMIZER (LR=$SERVER_LR)"
echo "Number of Training Examples: $TRAIN"
echo "Number of Evaluation Examples: $EVAL"
echo "ZO Variant:  $ZO_VARIANT"
echo "ZO Pert:     $ZO_PERTURBATION"
echo "ZO Epsilon:  $EPS"
echo "Num Pert:    $NUM_PERT"
echo "Split Layer: $SPLIT_LAYER"
echo "Extra args:  $EXTRA_ARGS $TASK_ARGS"
echo "=========================================="

# =============================================================================
# Helper Functions
# =============================================================================

cleanup() {
    echo ""
    echo "[$(date)] Cleaning up..."
    # Kill any remaining processes
    pkill -f "run_distributed.py" 2>/dev/null || true
    echo "[$(date)] Cleanup complete"
}

trap cleanup EXIT

wait_for_server() {
    local host=$1
    local port=$2
    local max_wait=3000
    local waited=0
    
    echo "[$(date)] Waiting for server at $host:$port..."
    
    while [ $waited -lt $max_wait ]; do
        # Try to connect using Python (more reliable than nc/netcat)
        if srun --nodes=1 --ntasks=1 -w $CLIENT_NODE \
            python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('$host', $port)); s.close()" 2>/dev/null; then
            echo "[$(date)] Server is ready! (waited ${waited}s)"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo "[$(date)] Still waiting... (${waited}s / ${max_wait}s)"
    done
    
    echo "[$(date)] ERROR: Server did not start within ${max_wait}s"
    return 1
}

# =============================================================================
# Step 1: Start Server on Node 0
# =============================================================================

echo ""
echo "=========================================="
echo "  Step 1: Starting Server"
echo "=========================================="
echo "[$(date)] Launching server on $SERVER_NODE..."

# Start server in background using run_distributed.py --role server
srun --nodes=1 --ntasks=1 -w $SERVER_NODE \
    bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate mezo
        # Multi-GPU for split learning: use device_map for model sharding (large models)
        # All GPUs visible to single process - model distributed via device_map='auto'
        export WANDB_MODE=disabled
        # Help with CUDA memory fragmentation for FO training
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        spack unload cuda 2>/dev/null || true
        spack load /jmc4bek
        export CUDA_HOME=\$(spack location -i /jmc4bek)
        
        echo '[SERVER] Loading model on '\$(hostname)' with '\$(nvidia-smi -L | wc -l)' GPUs at '\$(date)'...'
        echo '[SERVER] Will listen on 0.0.0.0:$PORT once model loaded'
        
        # Force unbuffered Python output and redirect stderr to stdout
        export PYTHONUNBUFFERED=1
        
        # Note: For split learning, we use single-process multi-GPU via device_map
        # accelerate launch with multiple processes won't work due to TCP connection model
        python -u run_distributed.py \
            --role server \
            --host 0.0.0.0 \
            --port $PORT \
            --device cuda \
            --backend tcp \
            --model_name $MODEL \
            --task_name $TASK \
            --split_layer $SPLIT_LAYER \
            --seed $SEED --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
            --max_steps $STEPS \
            --sgd_momentum $MOMENTUM \
            --trainer $TRAINER --client_optimizer $CLIENT_OPTIMIZER --server_optimizer $SERVER_OPTIMIZER \
            --learning_rate $LR --client_learning_rate $CLIENT_LR --server_learning_rate $SERVER_LR \
            --zo_variant $ZO_VARIANT --zo_perturbation $ZO_PERTURBATION --num_pert $NUM_PERT \
            --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS --lr_scheduler_type \"constant\" \
            --eval_steps $EVAL_STEPS \
            --eval_strategy steps \
            --optimizer $OPTIMIZER \
            $EXTRA_ARGS \
            $TASK_ARGS
    " &

SERVER_PID=$!
echo "[$(date)] Server process started (PID: $SERVER_PID)"

# =============================================================================
# Step 2: Wait for Server to be Ready
# =============================================================================

echo ""
echo "=========================================="
echo "  Step 2: Waiting for Server"
echo "=========================================="

# Give server a few seconds to start loading model
sleep 5

# Wait for server to be ready to accept connections
if ! wait_for_server $SERVER_IP $PORT $MAX_WAIT; then
    echo "[$(date)] FATAL: Server failed to start"
    exit 1
fi

# =============================================================================
# Step 3: Start Client on Node 1
# =============================================================================

echo ""
echo "=========================================="
echo "  Step 3: Starting Client"
echo "=========================================="
echo "[$(date)] Launching client on $CLIENT_NODE..."
echo "[$(date)] Connecting to $SERVER_IP:$PORT"

# Start client using run_distributed.py --role client
srun --nodes=1 --ntasks=1 -w $CLIENT_NODE \
    bash -c "
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate mezo
        # Multi-GPU for split learning: use device_map for model sharding (large models)
        # All GPUs visible to single process - model distributed via device_map='auto'
        export WANDB_MODE=disabled
        # Help with CUDA memory fragmentation for FO training
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        spack unload cuda 2>/dev/null || true
        spack load /jmc4bek
        export CUDA_HOME=\$(spack location -i /jmc4bek)
        
        echo '[CLIENT] Starting on '\$(hostname)' at '\$(date)
        echo '[CLIENT] Connecting to $SERVER_IP:$PORT'
        
        # Force unbuffered Python output
        export PYTHONUNBUFFERED=1
        
        # Note: For split learning, we use single-process multi-GPU via device_map
        # accelerate launch with multiple processes won't work due to TCP connection model
        python -u run_distributed.py \
            --role client \
            --server_host $SERVER_IP \
            --port $PORT \
            --device cuda \
            --backend tcp \
            --model_name $MODEL \
            --task_name $TASK \
            --sgd_momentum $MOMENTUM \
            --split_layer $SPLIT_LAYER \
            --seed $SEED --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
            --max_steps $STEPS \
            --trainer $TRAINER --client_optimizer $CLIENT_OPTIMIZER --server_optimizer $SERVER_OPTIMIZER \
            --learning_rate $LR --client_learning_rate $CLIENT_LR --server_learning_rate $SERVER_LR \
            --zo_variant $ZO_VARIANT --zo_perturbation $ZO_PERTURBATION --num_pert $NUM_PERT \
            --zo_eps $EPS --per_device_train_batch_size $BS --per_device_eval_batch_size $BS --lr_scheduler_type \"constant\" \
            --eval_steps $EVAL_STEPS \
            --eval_strategy steps \
            --optimizer $OPTIMIZER \
            $EXTRA_ARGS \
            $TASK_ARGS
    " &

CLIENT_PID=$!
echo "[$(date)] Client process started (PID: $CLIENT_PID)"

# =============================================================================
# Step 4: Wait for Training to Complete
# =============================================================================

echo ""
echo "=========================================="
echo "  Step 4: Training in Progress"
echo "=========================================="
echo "[$(date)] Waiting for training to complete..."
echo "Server PID: $SERVER_PID"
echo "Client PID: $CLIENT_PID"
echo ""
echo "Monitor progress with: tail -f ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"

# Wait for client to finish (client drives the training loop)
wait $CLIENT_PID
CLIENT_EXIT=$?

echo ""
echo "[$(date)] Client finished with exit code: $CLIENT_EXIT"

# Give server a moment to finish any pending operations
sleep 2

# Server will be cleaned up by the trap

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "  Training Complete"
echo "=========================================="
echo "End Time:    $(date)"
echo "Exit Code:   $CLIENT_EXIT"
echo "Job ID:      $SLURM_JOB_ID"
echo ""
echo "Logs:"
echo "  Output: ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "  Error:  ${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
echo "=========================================="

exit $CLIENT_EXIT
