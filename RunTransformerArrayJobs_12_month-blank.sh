#!/bin/bash
#
# Slurm Batch Script  for Perlmutter (NERSC) - TODO: FILL IN THE MISSING VALUES
# Run this by saying sbatch --array=0-15 RunTransformerArrayJobs_12_month.sh
# (4 patchify * 1 horizon * 4 models = 16 tasks)

# --- Slurm Directives ---
#SBATCH --job-name=12-month     # TODO SET Name of job (choose a short name so sqs shows it all)
#SBATCH --account=xxxxx         # NERSC project account
#SBATCH --qos=regular           # TODO: Specify the qos to use (e.g., debug, regular, shared)
#SBATCH --nodes=1               # Runs on a single node with GPU access
#SBATCH --constraint=gpu        # TODO: Specify node type: 'cpu' or 'gpu'
#SBATCH --gpus-per-node=4                   
#SBATCH --time=00:30:00         # TODO: SET EVERY TIME Maximum run time of 5 hours (HH:MM:SS)

# Output and standard errors will go here:
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err                    

#SBATCH --mail-type=BEGIN,END,FAIL               # Email notifications for job events
#   Options: BEGIN, END, FAIL, ALL, ARRAY_TASKS.

#SBATCH --mail-user=xxxxxx       # TODO - Replace with email address for notifications (replace)

# --- Environment Setup ---
echo "Starting job on $(hostname) at $(date)"
echo "SLURM_JOB_ID (Array ID) = ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"

# Load the Conda module and activate the environment
# This is crucial for accessing your custom Python packages and dependencies.
module load conda
conda activate /pscratch/sd/xxxxxx/conda_env/sic_sie_env

patchify_strategies=("rows" "lon_spilldown" "latitude_spillover_redo" "latlon_spillover")
forecast_horizons=(12)
model_files=("VisionTransformerPENONE.py" "VisionTransformerPESO.py" "VisionTransformerPETO.py" "VisionTransformerPECOM.py")

num_patchify=${#patchify_strategies[@]}
num_horizons=${#forecast_horizons[@]}
num_models=${#model_files[@]}

# Total number of unique combinations for patchify/horizon is: num_patchify * num_horizons
num_base_tasks=$(( num_patchify * num_horizons ))

task_id=${SLURM_ARRAY_TASK_ID}

total_tasks=$(( num_models * num_base_tasks ))
if (( task_id >= total_tasks )); then
  echo "Error: SLURM_ARRAY_TASK_ID ($task_id) exceeds total task count ($total_tasks)"
  exit 1
fi

model_idx=$(( task_id / num_base_tasks ))
base_task_idx=$(( task_id % num_base_tasks ))

patchify_idx=$(( base_task_idx / num_horizons ))
horizon_idx=$(( base_task_idx % num_horizons ))

PYTHON_FILE=${model_files[$model_idx]}
PATCHIFY_TO_USE=${patchify_strategies[$patchify_idx]}
FORECAST_HORIZON=${forecast_horizons[$horizon_idx]}

export SLURM_PATCHIFY_TO_USE=$PATCHIFY_TO_USE
export SLURM_FORECAST_HORIZON=$FORECAST_HORIZON

# --- Rename job dynamically ---
MODEL_NAME=$(basename $PYTHON_FILE .py)
NEW_NAME="${MODEL_NAME}_h${FORECAST_HORIZON}_${PATCHIFY_TO_USE}"
scontrol update JobId=$SLURM_JOB_ID JobName=$NEW_NAME

# --- Rename logs ---
OUT_FILE="${NEW_NAME}.out"
ERR_FILE="${NEW_NAME}.err"

echo "Redirecting stdout to $OUT_FILE and stderr to $ERR_FILE"
exec >  >(tee -a "$OUT_FILE")   # stdout
exec 2> >(tee -a "$ERR_FILE" >&2)  # stderr

echo "Renamed job to: $NEW_NAME"
echo "Patchify strategy: $SLURM_PATCHIFY_TO_USE"
echo "Forecast horizon: $FORECAST_HORIZON"
echo "Python script: $PYTHON_FILE"

# --- Run ---
python $PYTHON_FILE

echo "Job finished at $(date)"