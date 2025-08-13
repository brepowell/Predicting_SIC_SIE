#!/bin/bash
#
# Slurm Batch Script  for Perlmutter (NERSC) - TODO: FILL IN THE MISSING VALUES
# Run this by saying sbatch Runtransformer.sh
#

# --- Slurm Directives ---
#SBATCH --job-name=LSD                          # Name of job (choose a short name so sqs shows it all)
#SBATCH --account=mXXXXXX                        # NERSC project account (TODO: replace with actual account)
#SBATCH --qos=regular                          # Specify the queue/qos to use (e.g., debug, regular, shared)
#SBATCH --nodes=1                                # Runs on a single node with GPU access
#SBATCH --constraint=gpu                         # Specify node type: 'cpu' or 'gpu'
#SBATCH --gpus-per-node=4                        # Removed ntasks per node
#SBATCH --time=05:00:00                           # Maximum run time of 5 hours (HH:MM:SS)


#SBATCH --output=slurm-%j.out                    # Standard output file name (will include job ID)
#   %j is a placeholder that Slurm replaces with the job ID.

#SBATCH --error=slurm-%j.err                     # Standard error file name (will include job ID)
#   Errors will be directed here.

#SBATCH --mail-type=BEGIN,END,FAIL               # Email notifications for job events
#   Options: BEGIN, END, FAIL, ALL, ARRAY_TASKS.

#SBATCH --mail-user=your_email@example.com       # TODO - Replace with email address for notifications (replace)

# --- Environment Setup ---
echo "Starting job on $(hostname) at $(date)"

# Load the Conda module and activate the environment
# This is crucial for accessing your custom Python packages and dependencies.
module load conda
conda activate sic_sie_env

# TODO - TRY THIS: Set environment variables
# Ran latitude_spillover_redo on 8/2/2025 at 3:00 pm - failed after 1 hour! Time limite exceeded?? I had min time set
export SLURM_PATCHIFY_TO_USE="rows"  
# "lon_spilldown": "LSD", "latitude_spillover_redo": "PSO", rows, latlon_spillover
export FORECAST_HORIZON=3
# --- Job Execution ---

# Run the Python script
# Make sure the .py file is in the directory from which I submit the job,
# or provide the full path to it.
echo "Running Python script..."
python VisionTransformer.py

echo "Job finished at $(date)"

# --- End of Script ---
