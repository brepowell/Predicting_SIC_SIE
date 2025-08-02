#!/bin/bash
#
# Slurm Batch Script  for Perlmutter (NERSC) - TODO: FILL IN THE MISSING VALUES
#

# --- Slurm Directives ---
#SBATCH --job-name=Transformerfd_nTM_D128_B16_lt40_P210_L256_T7_Fh3_e10_LSD  # Name of job
#SBATCH --account=mXXXXXX                        # NERSC project account (TODO: replace with actual account)
#SBATCH --queue=regular                          # Specify the queue/qos to use (e.g., debug, regular, shared)
#SBATCH --nodes=1                                # Runs on a single node with GPU access
#SBATCH --ntasks-per-node=1                      # Number of tasks (processes) to run per node - single GPU
#SBATCH --time-min=1:00:00                       # Minimum run time of 1 hour (HH:MM:SS)
#SBATCH --time=5:00:00                           # Maximum run time of 5 hours (HH:MM:SS)
#SBATCH --constraint=gpu                         # Specify node type: 'cpu' or 'gpu'

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
# export MY_VAR="some_value"

# --- Job Execution ---

# Run the Python script
# Make sure the .py file is in the directory from which I submit the job,
# or provide the full path to it.
echo "Running Python script..."
python VisionTransformer.py

echo "Job finished at $(date)"

# --- End of Script ---
