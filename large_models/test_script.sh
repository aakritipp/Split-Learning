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

MODEL_NAME="facebook/opt-125m"
EPOCHS=1
BATCH_SIZE=4        # Smaller for stability
MAX_LENGTH=512      # Reduced
# LR=5e-4             # SGD learning rate
# ZOO_LR=5e-4         # ZOO learning rate (reduced)
# EPS=1e-3            # Smaller perturbation scale
SEED=42
TRAIN=100
DEV=50
EVAL=100
STEPS=100
EVAL_STEPS=100
NUM_PERT=10         # More perturbations
NUM_PREFIX=5
# LR2=1e-1
SEED=42
LR=1e-4          # For SGD
ZOO_LR=1e-4      # For ZOO
EPS=5e-4         # Even smaller perturbation
HOST="127.0.0.1"
PORT="12345"

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
    --host "$HOST" --port "$PORT" \
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
    --num_pert $NUM_PERT &       

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server initialization
echo "Waiting 60 seconds for server initialization..."

TIMEOUT=300
INTERVAL=2
ELAPSED=0
while ! HOST="$HOST" PORT="$PORT" python3 - <<'PY'
import os, socket, sys
host = os.environ.get("HOST") or "127.0.0.1"
port_env = os.environ.get("PORT") or "12345"
try:
    port = int(port_env)
except Exception:
    sys.exit(2)  # bad env -> treat as not ready

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2.0)
try:
    s.connect((host, port))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
do
  sleep $INTERVAL
  ELAPSED=$((ELAPSED + INTERVAL))
  if [ $ELAPSED -ge $TIMEOUT ]; then
    echo "Server did not open $HOST:$PORT within $TIMEOUT seconds."
    if ps -p $SERVER_PID > /dev/null; then
      echo "Server process is running (PID $SERVER_PID) but not listening yet."
    else
      echo "Server process is not running."
    fi
    exit 1
  fi
done
echo "Server is listening on $HOST:$PORT"

# Start client
echo "Starting client..."
python3 client.py \
    --host "$HOST" --port "$PORT" \
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
    --max_steps $STEPS &

CLIENT_PID=$!

echo "Waiting for server to finish..."
wait $CLIENT_PID
CLIENT_EXIT_CODE=$?

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