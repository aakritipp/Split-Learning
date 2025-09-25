#!/bin/bash -l
#SBATCH --job-name=mezo_splitlearning
#SBATCH --account=fl-het
#SBATCH --partition=debug
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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


# MODEL_NAME="facebook/opt-125m"
# EPOCHS=1
# BATCH_SIZE=16  # Small for testing
# MAX_LENGTH=512  # Short for speed
# LR=1e-3
# ZOO_LR=1e-2
# EPS=1e-1        # ZOO epsilon (perturbation scale)
# SEED=0          # Random seed
# TRAIN=1000      # Training examples
# DEV=500         # Dev examples
# EVAL=1000       # Evaluation examples
# STEPS=1000      # Training steps
# EVAL_STEPS=1000 # Evaluation steps
# NUM_PERT=5

MODEL_NAME="facebook/opt-125m"
EPOCHS=1
BATCH_SIZE=16        # Smaller for stability
MAX_LENGTH=512      # Reduced
# LR=5e-4             # SGD learning rate
# ZOO_LR=5e-4         # ZOO learning rate (reduced)
# EPS=1e-3            # Smaller perturbation scale
SEED=42
TRAIN=1000
DEV=500
EVAL=1000
STEPS=1000
EVAL_STEPS=1000
NUM_PERT=10         # More perturbations
NUM_PREFIX=20
# LR2=1e-1
TASK="xsum"
TUNING="lora"
LR=5e-3          # For AdamW (prefix tuning)
ZOO_LR=5e-4      # For ZOO
EPS=5e-4         # Even smaller perturbation
LORA_R=8
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
echo ""

# Start server in background
echo "Starting server..."
python3 server.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --mu $EPS \
    --train_batch_size $BATCH_SIZE \
    --test_batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --lr $LR \
    --zoo_lr $ZOO_LR \
    --seed $SEED \
    --train_examples $TRAIN \
    --dev_examples $DEV \
    --eval_examples $EVAL \
    --max_steps $STEPS \
    --eval_steps $EVAL_STEPS \
    --num_prefix $NUM_PREFIX \
    --task $TASK \
    --tuning $TUNING \
    --lora_r $LORA_R \
    --num_pert $NUM_PERT &     

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
echo "Starting client..."
python3 client.py \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
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