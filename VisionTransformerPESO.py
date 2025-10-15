#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
print('Xarray version', xr.__version__)


# In[2]:


import zarr
print('Zarr version', zarr.__version__)


# In[3]:


import numpy as np
print('Numpy version', np.__version__)


# In[4]:


import torch
from torch.utils.data import Dataset, DataLoader

print('PyTorch version', torch.__version__)


# In[5]:


from perlmutterpath import *  # Contains the data_dir and mesh_dir variables
NUM_FEATURES = 2              # C: Number of features per cell (ex., Freeboard, Ice Area)


# In[6]:


from NC_FILE_PROCESSING.patchify_utils import *

# Available patchify functions
PATCHIFY_FUNCTIONS = {
    "agglomerative": compute_agglomerative_patches,
    "breadth_first_bfs_basic": build_patches_from_seeds_bfs_basic,
    "breadth_first_improved_padded": build_patches_from_seeds_improved_padded,
    "dbscan": get_clusters_dbscan,
    "kmeans": cluster_patches_kmeans,
    "knn_basic": compute_knn_patches,
    "knn_disjoint": compute_disjoint_knn_patches,
    "latlon_spillover": patchify_by_latlon_spillover,              # RELIABLE PERFORMANCE
    "latitude_neighbors": patchify_by_latitude,
    "latitude_simple": patchify_by_latitude_simple,
    "latitude_spillover_redo": patchify_with_spillover,            # checking
    "lon_spilldown": patchify_by_lon_spilldown,                    # TERRIBLE PERFORMANCE
    "rows": get_rows_of_patches,                                   # RELIABLE PERFORMANCE
    "staggered_polar_descent": patchify_staggered_polar_descent,
}

PATCHIFY_ABBREVIATIONS = {
    "agglomerative": "AGG",
    "breadth_first_bfs_basic": "BFSB",
    "breadth_first_improved_padded": "BPIP",
    "dbscan": "DBSCAN",
    "kmeans": "KM",
    "knn_basic": "KNN",
    "knn_disjoint": "DKNN",
    "latlon_spillover": "LLSO",               # RELIABLE PERFORMANCE
    "latitude_neighbors": "LAT",
    "latitude_simple": "LSIM",
    "latitude_spillover_redo": "PSO",         # Uses PSO (Patchify SpillOver) - TRY THIS
    "lon_spilldown": "LSD",                   # lOW PERFORMING
    "rows": "ROW",                            # RELIABLE PERFORMANCE
    "staggered_polar_descent": "SPD",
}


# # Variables for the Model
# 
# Check over these CAREFULLY!
# 
# Note that if you use the login node for training for the Jupyter notebook version (even for the trial dataset that is much smaller), you run the risk of getting the error: # OutOfMemoryError: CUDA out of memory.

# In[7]:


# --- Run Settings:
MONTHLY =                      True
TRIAL_RUN =                    False   # SET THIS TO USE THE PRACTICE SET (MUCH FASTER AND SMALLER, for debugging)
NORMALIZE_ON =                 False   # SET THIS TO USE NORMALIZATION ON FREEBOARD (Results are independent of patchify used)
TRAINING =                     False    # SET THIS TO RUN THE TRAINING LOOP (Use on full dataset for results)
FAST_EVAL_ON =                 False    # SET THIS TO RUN THE METRICS AT THE BOTTOM (Use on full dataset for results)
SLOW_EVAL_ON =                 True
MAP_WITH_CARTOPY_ON =          False   # Make sure the Cartopy library is included in the kernel

# Only run ONCE for daily and once for monthly!!
# Run Settings (already performed, not needed now - KEEP FALSE!!!)
PLOT_DATA_SPLIT_DISTRIBUTION = False   # Run the data split function to see the train, val, test distribution
MAX_FREEBOARD_ON =            False   # To normalize with a pre-defined maximum for outlier handling

# --- Time-Related Variables:
CONTEXT_LENGTH = 12           # T: Number of historical time steps used for input

# --- Model Hyperparameters:
D_MODEL = 128                 # d_model: Dimension of the transformer's internal representations (embedding dimension)
N_HEAD = 8                    # nhead: Number of attention heads
NUM_TRANSFORMER_LAYERS = 4    # num_layers: Number of TransformerEncoderLayers
BATCH_SIZE = 4                # Monthly requires smaller batch size, because of fewer samples
NUM_EPOCHS = 40

# Note that when using multiple GPUs batch size will be multiplied by the number of GPUs available (x4).
# Monthly: Use a batch size of 4 and num epochs of 40 for best results
# Daily: Use a batch size of 16 for 3 day forecast; lower this for longer range; kernel may die with larger sizes

# --- Performance-Related Variables:
NUM_WORKERS = 64   # 64 worked fast for the 7 day forecast; too many workers causes it to stall out
PREFETCH_FACTOR = 4 # 4 worked fast for the 7 day forecast (tried 16 for 4 GPUs, but ran out of shared memory; 8 works ok)

# Daily: For 7 day forecast: num workers = 64; prefetch factor = 4

# --- Feature-Related Variables:
MAX_FREEBOARD_FOR_NORMALIZATION = 1    # Only works when you set MAX_FREEBOARD_ON too; Zarr files don't have this saved.

# --- Space-Related Variables:
LATITUDE_THRESHOLD = 40          # Determines number of cells and patches (could use -90 to use the entire dataset).
CELLS_PER_PATCH = 256            # L: Number of cells within each patch (based on ViT paper 16 x 16 = 256)

# SLURM - CONVERT THIS TO A .PY FILE FIRST
PATCHIFY_TO_USE = os.environ.get("SLURM_PATCHIFY_TO_USE", "rows") # for SLURM
FORECAST_HORIZON = int(os.environ.get("SLURM_FORECAST_HORIZON", 7)) # for SLURM


# ## Other Variables Dependent on Those Above ^

# In[8]:


mesh = xr.open_dataset(mesh_dir)
latCell = np.degrees(mesh["latCell"].values)
lonCell = np.degrees(mesh["lonCell"].values)
mesh.close()
print("Total nCells:       ", len(latCell))

mask = latCell >= LATITUDE_THRESHOLD
masked_ncells_size = np.count_nonzero(mask)
print("Mask size:          ", masked_ncells_size)

NUM_PATCHES = masked_ncells_size // CELLS_PER_PATCH    # P: Approximate number of spatial patches to expect

print("cells_per_patch:    ", CELLS_PER_PATCH)
print("n_patches:          ", NUM_PATCHES)

# The input dimension for the patch embedding linear layer.
# Each patch at a given time step has NUM_FEATURES * CELLS_PER_PATCH features.
# This is the 'D' dimension used in the Transformer's input tensor (B, T, P, D).
PATCH_EMBEDDING_INPUT_DIM = NUM_FEATURES * CELLS_PER_PATCH # 2 * 256 = 512

DEFAULT_PATCHIFY_METHOD_FUNC = PATCHIFY_FUNCTIONS[PATCHIFY_TO_USE]

# --- Common Parameters for all functions ---
COMMON_PARAMS = {
    "latCell": latCell,
    "lonCell": lonCell,
    "cells_per_patch": CELLS_PER_PATCH, 
    "num_patches": NUM_PATCHES,
    "latitude_threshold": LATITUDE_THRESHOLD,
    "seed": 42
}

cellsOnCell = np.load(f'cellsOnCell.npy')

# --- Function-specific Parameters (if any) ---
SPECIFIC_PARAMS = {
    "latitude_spillover_redo": {"step_deg": 5, "max_lat": 90},
    "latitude_simple": {"step_deg": 5, "max_lat": 90},
    "latitude_neighbors": {"step_deg": 5, "max_lat": 90},
    "breadth_first_improved_padded": {"cellsOnCell": cellsOnCell, "pad_to_exact_size": True},
    "breadth_first_bfs_basic": {"cellsOnCell": cellsOnCell},
    "agglomerative": {"n_neighbors": 5},
    "kmeans": {},
    "dbscan": {},
    "rows": {},
    "knn_basic": {},
    "knn_disjoint": {},
    "latlon_spillover": {},
    "lon_spilldown": {},
    "staggered_polar_descent": {},
}


# In[9]:


if NORMALIZE_ON:
    if MAX_FREEBOARD_ON:
        norm = f"nT{MAX_FREEBOARD_FOR_NORMALIZATION}" # Using the specified max value
    else:
        norm = "nTM" # Using the absolute max
else:
    norm = "nF" # Normalization is off
    
# Get the abbreviation, with a fallback for functions not yet mapped
patching_strategy_abbr = PATCHIFY_ABBREVIATIONS.get(PATCHIFY_TO_USE, "UNKNWN")

if patching_strategy_abbr == "UNKNWN":
    raise ValueError("Check the name of the patchify function")

if TRAINING and not torch.cuda.is_available():
    raise ValueError("There is a problem with Torch not recognizing the GPUs")
else:
    if torch.cuda.device_count() != 0:
        BATCH_SIZE = BATCH_SIZE * torch.cuda.device_count()
    # import psutil
    # NUM_WORKERS = psutil.cpu_count() - 2 # not working!
    # print(f"num_workers is {NUM_WORKERS}")

# Model nome convention
model_version = (
    f"peSO_"
    f"{'M' if MONTHLY else 'D'}" # MONTHLY OR DAILY
    f"{'tr' if TRIAL_RUN else 'fd'}_" # TRIAL DATASET OR FULL DATASET
    f"{norm}_D{D_MODEL}_B{BATCH_SIZE}_lt{LATITUDE_THRESHOLD}_P{NUM_PATCHES}_L{CELLS_PER_PATCH}"
    f"_T{CONTEXT_LENGTH}_Fh{FORECAST_HORIZON}_e{NUM_EPOCHS}_{patching_strategy_abbr}"
)

# Place to save and load the data
PROCESSED_DATA_DIR = (
    f"{'Monthly_' if MONTHLY else 'Daily_'}{'tr' if TRIAL_RUN else 'fd'}_{norm}_data.zarr"
)

print(f"Model Version: {model_version}")
print(f"Dataset Name: {PROCESSED_DATA_DIR}")


# ### Notes:
# 
# * TRY: NUM_WORKERS as 16 to 32 - profile to see if the GPU is still waiting on the CPU.
# * TRY: NUM_WORKERS as 64 - the number of CPU cores available.
# * TRY: NUM_WORKERS experiment with os.cpu_count() - 2
# * TRY: NUM_WORKERS experiment with (logical_cores_per_gpu * num_gpus)
# 
# num_workers considerations:
# * Too few workers: GPUs might become idle waiting for data.
# * Too many workers: Can lead to increased CPU memory usage and context switching overhead.

# # More Imports

# In[10]:


import sys
print('System Version:', sys.version)


# In[11]:


#print(sys.executable) # for troubleshooting kernel issues
#print(sys.path)


# In[12]:


import os
#print(os.getcwd())


# In[13]:


import pandas as pd
print('Pandas version', pd.__version__)


# In[14]:


import matplotlib
import matplotlib.pyplot as plt
print('Matplotlib version', matplotlib.__version__)


# In[15]:


import seaborn as sns
print('Seaborn version', sns.__version__)


# # Hardware Details

# In[16]:


if TRAINING and not torch.cuda.is_available():
    raise ValueError("There is a problem with Torch not recognizing the GPUs")
else:
    print(f"Number of GPUs: {torch.cuda.device_count()}") # check the number of available CUDA devices
    # will print 1 on login node; 4 on GPU exclusive node; 1 on shared GPU node


# In[17]:


#print(torch.cuda.get_device_properties(0)) #provides information about a specific GPU
#total_memory=40326MB, multi_processor_count=108, L2_cache_size=40MB


# In[18]:


import psutil
import platform

# Get general CPU information
processor_name = platform.processor()
print(f"Processor Name: {processor_name}")

# Get core counts
physical_cores = psutil.cpu_count(logical=False)
logical_cores = psutil.cpu_count(logical=True)
print(f"Physical Cores: {physical_cores}")
print(f"Logical Cores: {logical_cores}")

# Get CPU frequency
cpu_frequency = psutil.cpu_freq()
if cpu_frequency:
    print(f"Current CPU Frequency: {cpu_frequency.current:.2f} MHz")
    print(f"Min CPU Frequency: {cpu_frequency.min:.2f} MHz")
    print(f"Max CPU Frequency: {cpu_frequency.max:.2f} MHz")

# Get CPU utilization (percentage)
# The interval argument specifies the time period over which to measure CPU usage.
# Setting percpu=True gives individual core utilization.
cpu_percent_total = psutil.cpu_percent(interval=1)
print(f"Total CPU Usage: {cpu_percent_total}%")

# cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
# print("CPU Usage per Core:")
# for i, percent in enumerate(cpu_percent_per_core):
#     print(f"  Core {i+1}: {percent}%")



# # Custom Pytorch Dataset
# Example from NERSC of using ERA5 Dataset:
# 
# https://github.com/NERSC/dl-at-scale-training/blob/main/utils/data_loader.py

# # __ init __ - masks and loads the data into tensors

# In[19]:


import os
import time
from datetime import datetime
from datetime import timedelta

import glob
from torch.utils.data import Dataset
from typing import List, Union, Callable, Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler

from NC_FILE_PROCESSING.patchify_utils import *
from NC_FILE_PROCESSING.metrics_and_plots import *
from perlmutterpath import * # Contains the data_dir and mesh_dir variables

import logging

# Set level to logging.INFO to see the statements
logging.basicConfig(filename=f'{model_version}_dataset.log', filemode='w', level=logging.INFO)

class MonthlyOrDailyNetCDFDataset(Dataset):
    """
    PyTorch Dataset that concatenates a directory of month-wise NetCDF files
    along their 'Time' dimension and yields daily data *plus* its timestamp.

    Parameters
    ----------
    transform : Callable | None
        Optional - transform applied to the data tensor *only*.
    processed_data_path
        The Zarr directory from which to retrieve the preprocesed data  
    context_length
        The number of time steps to fetch for input in the prediction step (needed for __len__)
    forecast_horizon
        The number of time steps to predict in the future (needed for __len__)
    load_normalized
        Determines which version of the pre-processed data to load
    num_patches
        How many patches to use for the patchify function
    cells_per_patch
        How many cells to have in each patch for patchify
    patchify_func : Callable
        The patchify function to use (ex., patchify_by_latlon_spillover).
    patchify_func_key : str
        The string key identifying the patchify function (e.g., "latlon_spillover")
        used to look up its specific parameters.

    """
    def __init__(
        self,
        transform: Callable = None,

        # File parameters
        processed_data_path: str = PROCESSED_DATA_DIR,

        # For __len__ function
        context_length: int = CONTEXT_LENGTH,
        forecast_horizon: int = FORECAST_HORIZON,

        # Run Settings      
        load_normalized: bool = NORMALIZE_ON, 

        # Patchification parameters
        num_patches: int = NUM_PATCHES,
        cells_per_patch: int = CELLS_PER_PATCH,
        patchify_func: Callable = DEFAULT_PATCHIFY_METHOD_FUNC, # Default patchify function
        patchify_func_key: str = PATCHIFY_TO_USE, # Key to look up specific params
    ):

        """ __init__ needs to 
        1) Load the pre-processed data from Zarr
        2) Apply the specified patchify and store patch_ids so the data loader can use them
        """

        # Check for performance
        start_time = time.perf_counter()

        # Save parameters
        self.transform = transform
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon

        self.load_normalized = load_normalized
        self.num_patches = num_patches
        self.cells_per_patch = cells_per_patch
        self.patchify_func = patchify_func # Store the specified patchify function
        self.patchify_func_key = patchify_func_key # Store the key for looking up specific params
        self.processed_data_path = processed_data_path # Store the processed data path

        # --- 1. Load pre-processed data from Zarr ---
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(f"Pre-processed Zarr file not found. Please run `preprocess_data_variables.py` first.")

        try:
            '''
            Note that "time" will be 2100 for Monthly and 63875 for Daily
            
            Dimensions:         (nCells_full: 465044, time: 2100, nCells_masked: 53973)
            Coordinates:
              * nCells_full     (nCells_full) int64 4MB 0 1 2 3 ... 465041 465042 465043
              * nCells_masked   (nCells_masked) int64 432kB 0 1 2 3 ... 53970 53971 53972
              * time            (time) datetime64[ns] 511kB 1850-01-01T00:30:00 ... 2024-...
            Data variables:
                cell_mask       (nCells_full) bool 465kB dask.array<chunksize=(232522,), meta=np.ndarray>
                freeboard       (time, nCells_masked) float32 14GB dask.array<chunksize=(999, 1687), meta=np.ndarray>
                full_to_masked  <U1337572 5MB ...
                ice_area        (time, nCells_masked) float32 14GB dask.array<chunksize=(999, 1687), meta=np.ndarray>
                masked_to_full  <U1337572 5MB ...
                num_raw_files   int64 8B ...
                times           (time) datetime64[ns] 511kB dask.array<chunksize=(31938,), meta=np.ndarray>
                '''
            processed_ds = xr.open_zarr(self.processed_data_path)
            print(processed_ds)
            logging.info(processed_ds)

            # Times
            print(f"processed_ds['times'].values.shape: {processed_ds['times'].values.shape}  # Expected: ({'2100' if MONTHLY else '63875'},)")
            logging.info(f"times.shape = {processed_ds['times'].values.shape}")
            self.times = processed_ds["times"].values

            print(f"Type of self.times: {type(self.times)}")
            print(f"Data type of elements in self.times: {self.times.dtype}")
            print(self.times[0])
            
            # Ice Area
            print(f"processed_ds['ice_area'].load().values.shape: {processed_ds['ice_area'].load().values.shape}  # Expected: ({'2100' if MONTHLY else '63875'}, 53973)")
            logging.info(f"ice_area.shape = {processed_ds['ice_area'].load().values.shape}")
            self.ice_area = processed_ds["ice_area"].load().values
            
            # Freeboard
            print(f"processed_ds['freeboard'].load().values.shape: {processed_ds['freeboard'].load().values.shape}  # Expected: ({'2100' if MONTHLY else '63875'}, 53973)")
            logging.info(f"freeboard.shape = {processed_ds['freeboard'].load().values.shape}")
            self.freeboard = processed_ds["freeboard"].load().values
            
            # Cell Mask
            print(f"processed_ds['cell_mask'].values.shape: {processed_ds['cell_mask'].values.shape}  # Expected: (465044,)")
            logging.info(f"cell_mask.shape = {processed_ds['cell_mask'].values.shape}")
            self.cell_mask = processed_ds["cell_mask"].values
            
            # full_to_masked
            self.full_to_masked = eval(processed_ds["full_to_masked"].item())
            print(f"evaluated full_to_masked shape: {len(self.full_to_masked)}  # Expected: (53973,)")
            logging.info(f"evaluated full_to_masked shape = {np.shape(self.full_to_masked)}")
            
            # masked_to_full
            self.masked_to_full = eval(processed_ds["masked_to_full"].item())
            print(f"evaluated masked_to_full shape: {len(self.masked_to_full)}  # Expected: (53973,)")
            logging.info(f"evaluated masked_to_full shape = {np.shape(self.masked_to_full)}")
            
            # num_raw_files
            print(f"processed_ds['num_raw_files'].item(): {processed_ds['num_raw_files'].item()}  # Expected: 2100")
            logging.info(f"num_raw_files = {processed_ds['num_raw_files'].item()}")
            self.num_raw_files = processed_ds["num_raw_files"].item()
                        
            logging.info(f"Loaded pre-processed data from Zarr in {time.perf_counter() - start_time:.2f} seconds.")
            
        except Exception as e:
            logging.error(f"Error loading Zarr store: {e}")
            raise RuntimeError(f"Failed to load Zarr store. Check the file or rerun preprocessing: {e}")

        # --- 2. Patchify  ---
        logging.info(f"=== Patchifying using {self.patchify_func_key} algorithm ===")
        patchify_start_time = time.perf_counter()

        patchify_call_params = COMMON_PARAMS.copy()
        patchify_call_params.update(SPECIFIC_PARAMS.get(self.patchify_func_key, {}))

        self.latCell = COMMON_PARAMS["latCell"]
        
        # Use the dynamic patchify function
        #     Returns 
        # full_nCells_patch_ids : np.ndarray
        #     Array of shape (nCells,) giving patch ID or -1 if unassigned.
        # indices_per_patch_id : List[np.ndarray]
        #     List of patches, each a list of cell indices (np.ndarray of ints) that correspond with nCells array.
        # patch_latlons : np.ndarray
        #     Array of shape (n_patches, 2) containing (latitude, longitude) for one
        #     representative cell per patch (the first cell added to the patch)
        self.full_nCells_patch_ids, self.indices_per_patch_id, self.patch_latlons, self.algorithm = self.patchify_func(**patchify_call_params)

        logging.info(f"Longitude and latitude array shape: {self.patch_latlons.shape} should be (n_patches, 2)")
        logging.info(f"Minimum latitude:  {np.min(self.patch_latlons, axis=0)[0]}")
        logging.info(f"Maximum latitude:  {np.max(self.patch_latlons, axis=0)[0]}")
        logging.info(f"Minimum longitude: {np.min(self.patch_latlons, axis=0)[1]}")
        logging.info(f"Maximum longitude: {np.max(self.patch_latlons, axis=0)[1]}")
        logging.info(f"Should be between lat_threshold and 90 degrees for latitude")
        logging.info(f"Should be between 0 and 360 degrees for longitude")
        
        # Convert full-domain patch indices to masked-domain indices
        # This ensures there's no out of bounds problem,
        # like index 296237 is out of bounds for axis 1 with size 53973
        self.indices_per_patch_id = [
            [self.full_to_masked[i] for i in patch if i in self.full_to_masked]
            for patch in self.indices_per_patch_id
        ]
        
        logging.info(f"Patchifying completed in {time.perf_counter() - patchify_start_time:.2f} seconds.")

        # Stats on how many dates there are
        logging.info(f"Total time steps collected (days or months): {len(self.times)}")
        logging.info(f"Unique times: {len(np.unique(self.times))}")
        logging.info(f"First 35 time values: {self.times[:35]}") # TODO: MAYBE CHANGE FOR MONTHS?
        logging.info(f"Last 35 time values: {self.times[-35:]}")

        logging.info(f"Shape of combined ice_area array: {self.ice_area.shape}")
        logging.info(f"Shape of combined freeboard array: {self.freeboard.shape}")

        logging.info(f"Elapsed time for MonthlyOrDailyNetCDFDataset __init__: {time.perf_counter() - start_time} seconds")
        print(f"Elapsed time for MonthlyOrDailyNetCDFDataset __init__: {time.perf_counter() - start_time} seconds")

    def __len__(self) -> int:
        """
        Returns the total number of possible starting indices (idx) for a valid sequence.
        A valid sequence needs `self.context_length` for input and `self.forecast_horizon` for the target.
        
        ex) For daily data, if the total number of days is 365, 
        the context_length is 7 and the forecast_horizon is 3, then
        
        365 - (7 + 3) + 1 = 365 - 10 + 1 = 356 is the final valid starting index

        ex) For monthly data, if the total number of months is 12, 
        the context_length is 6 and the forecast_horizon is 3, then
        12 - (6 + 3) + 1 = 12 - 9 + 1 = 4 is the final valid starting index

        return len(self.freeboard) - (self.context_length + self.forecast_horizon)
                
        """
        required_length = self.context_length + self.forecast_horizon

        # Error check
        if len(self.freeboard) < required_length:
            return 0 # Not enough raw data to form even one sample

        # The number of valid starting indices
        return len(self.freeboard) - required_length + 1

    def get_patch_tensor(self, time_step_idx: int) -> torch.Tensor:
        
        """
        Retrieves the feature data for a specific time step, organized into patches.

        This method extracts 'freeboard' and 'ice_area' data for a given time step (month or day)
        and then reshapes it according to the pre-defined patches. Each patch
        will contain its own set of feature values.

        Parameters
        ----------
        time_step_idx : int
            The integer index of the time step to retrieve data for, relative to the
            concatenated dataset's time dimension.

        Returns
        -------
        torch.Tensor
            A tensor containing the feature data organized by patches for the
            specified month or day.
            Shape: (context_length, num_patches, num_features, patch_size)
            Where:
            - num_patches: Total number of patches (ex., 210).
            - num_features: The number of features per cell (currently 2: freeboard, ice_area).
            - patch_size: The number of cells within each patch (ex., 256)
            
        """

        freeboard_now = self.freeboard[time_step_idx]  # (nCells,)
        ice_area_now = self.ice_area[time_step_idx]    # (nCells,)
        features = np.stack([freeboard_now, ice_area_now], axis=0)  # (2, nCells)
        patch_tensors = []

        for patch_indices in self.indices_per_patch_id:
            patch = features[:, patch_indices]  # (2, patch_size)
            patch_tensors.append(torch.tensor(patch, dtype=torch.float32))

        return torch.stack(patch_tensors)  # (context_length, num_patches, num_features, patch_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int, int]:

        """__ getitem __ needs to 
        
        1. Given an input of a certain date id, get the input and the target tensors
        2. Return all the patches for the input and the target
           Features are: [freeboard, ice_area] over masked cells. 
           
        """
        # Start with the id of the time step (month or day) in question
        start_idx = idx

        # Extract datetime object for the start of the sequence
        start_date_np = self.times[start_idx]

        year  = start_date_np.astype("datetime64[Y]").astype(int) + 1970
        month = start_date_np.astype("datetime64[M]").astype(int) % 12 + 1
        day   = (start_date_np.astype("datetime64[D]") - start_date_np.astype("datetime64[M]")).astype(int) + 1
        
        # end_idx is the exclusive end of the input sequence,
        # and the inclusive start of the target sequence.
        end_idx = idx + self.context_length    # ex. start is 0, context length is 3, end is 3, exclusive
        target_start = end_idx                 # target starts indexing at 3

        # the target sequence ends after forecast horizon
        target_end = end_idx + self.forecast_horizon       # target ends at 3 + 7 = 10 exclusive

        if target_end > len(self.freeboard):
            raise IndexError(
                f"Requested time window exceeds dataset. "
                f"Problematic idx: {idx}, "
                f"Context Length: {self.context_length}, "
                f"Forecast Horizon: {self.forecast_horizon}, "
                f"Calculated target_end: {target_end}, "
                f"Actual dataset length (len(self.freeboard)): {len(self.ice_area)}"
            )

        # Build input tensor
        input_seq = [self.get_patch_tensor(i) for i in range(start_idx, end_idx)]
        input_tensor = torch.stack(input_seq)
    
        # Build target tensor: shape (forecast_horizon, num_patches)
        target_seq = self.ice_area[end_idx:target_end]
        target_patches = []
        for time_step in target_seq:
            patch_time_step = [
                torch.tensor(time_step[patch_indices]) for patch_indices in self.indices_per_patch_id
            ]
            
            # After stacking, patch_time_step_tensor will be (num_patches, CELLS_PER_PATCH)
            patch_time_step_tensor = torch.stack(patch_time_step)  # (num_patches,)
            target_patches.append(patch_time_step_tensor)

        # Final target tensor shape: (forecast_horizon, num_patches, CELLS_PER_PATCH)
        target_tensor = torch.stack(target_patches)  # (forecast_horizon, num_patches)
        
        return (
            input_tensor,
            target_tensor,
            start_idx,
            end_idx,
            target_start,
            target_end,
            year,   # for positional encoding
            month,  # for positional encoding
            day     # for positional encoding
        )
        

    def __repr__(self):
        """ Format the string representation of the data """
        return (
            f"<"
            f"{self.processed_data_path}"
            f"\nInstance of MonthlyOrDailyNetCDFDataset"
            f"\n{len(self)} viable time steps (only includes up to the last possible input date)"
            f"\n{len(self.freeboard[0])} cells/time_step"
            f"\n{self.num_raw_files} files loaded "
            f"\n{len(self.ice_area)} ice_area length"
            f"\n{len(self.freeboard)} freeboard length"
            f"\nPatchify Algorithm: {self.algorithm}" # What patchify algorithm was used
            f"\n # The following should be ({'2100' if MONTHLY else '63875'}, 53973) for {'Monthly' if MONTHLY else 'Daily'} data at a latitude_threshold of 40"
            f"\n{self.ice_area.shape} shape of ice_area"
            f"\n{self.freeboard.shape} shape of freeboard"
            f"\n{len(self.indices_per_patch_id)} indices_per_patch_id # Should be {len(self.freeboard[0]) // CELLS_PER_PATCH}"
            f">" 

        )

    def time_to_dataframe(self) -> pd.DataFrame:
            """Return a DataFrame of time features you can merge with predictions."""
            t = pd.to_datetime(self.times)            # pandas Timestamp index
            return pd.DataFrame(
                {
                    "time": t,
                    "year": t.year,
                    "month": t.month,
                    "day": t.day,
                    "doy": t.dayofyear,
                }
            )


# --- Custom Collate Function ---
def custom_collate_fn(batch: List[Tuple]) -> Tuple:
    """
    Custom collate function to handle the specific output of DailyNetCDFDataset.__getitem__.
    Ensures that year, month, and day are batched as (B,) tensors.
    """
    print("\n--- DEBUG: Inside custom_collate_fn ---")
    
    transposed_batch = list(zip(*batch))

    # Debug prints for the raw lists before torch.tensor()
    print(f"COLLECTED BATCH SIZE: {len(batch)}")
    print(f"TYPE OF TRANSPOSED_BATCH[6] (years list): {type(transposed_batch[6])}")
    print(f"TYPE OF TRANSPOSED_BATCH[7] (months list): {type(transposed_batch[7])}")
    print(f"TYPE OF TRANSPOSED_BATCH[8] (days list): {type(transposed_batch[8])}")

    if len(transposed_batch[8]) > 0:
        print(f"FIRST ELEMENT OF DAYS LIST: {transposed_batch[8][0]}")
        print(f"TYPE OF FIRST ELEMENT OF DAYS LIST: {type(transposed_batch[8][0])}")
        if isinstance(transposed_batch[8][0], np.ndarray):
            print(f"SHAPE OF FIRST ELEMENT (if numpy array): {transposed_batch[8][0].shape}")
        elif isinstance(transposed_batch[8][0], torch.Tensor):
            print(f"SHAPE OF FIRST ELEMENT (if torch tensor): {transposed_batch[8][0].shape}")
    
    # Collate input_tensor and target_tensor
    input_tensors = torch.stack(transposed_batch[0])
    target_tensors = torch.stack(transposed_batch[1])

    # Collate scalar integers (start_idx, end_idx, target_start, target_end, year, month, day)
    start_indices = torch.tensor(transposed_batch[2], dtype=torch.long)
    end_indices = torch.tensor(transposed_batch[3], dtype=torch.long)
    target_starts = torch.tensor(transposed_batch[4], dtype=torch.long)
    target_ends = torch.tensor(transposed_batch[5], dtype=torch.long)
    
    years = torch.tensor(transposed_batch[6], dtype=torch.long)
    months = torch.tensor(transposed_batch[7], dtype=torch.long)
    days = torch.tensor(transposed_batch[8], dtype=torch.long)

    print(f"DEBUG: Shape of years AFTER COLLATING: {years.shape}")
    print(f"DEBUG: Shape of months AFTER COLLATING: {months.shape}")
    print(f"DEBUG: Shape of days AFTER COLLATING: {days.shape}")
    print("--- END DEBUG: Inside custom_collate_fn ---\n")

    return (
        input_tensors,
        target_tensors,
        start_indices,
        end_indices,
        target_starts,
        target_ends,
        years,
        months,
        days
    )


# # DataLoader

# In[20]:


from torch.utils.data import Dataset, Subset

class YearlySubset(Dataset):
    def __init__(self, raw_subset, context_length=CONTEXT_LENGTH, forecast_horizon=FORECAST_HORIZON):
        """
        Args:
            raw_subset (Subset): A Subset of the main dataset containing raw day indices.
            context_length (int): The length of the input sequence.
            forecast_horizon (int): The length of the target sequence.
        """
        self.raw_subset = raw_subset
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon

        # The indices of the raw data that the subset has access to.
        # These are the global indices of the original dataset.
        self.raw_indices = self.raw_subset.indices

        # Calculate the number of viable samples within this subset.
        # A sample starts at index `i` of the raw_subset.
        # A sample requires a total window of `context_length + forecast_horizon` time steps
        required_window_size = self.context_length + self.forecast_horizon

        # The last valid starting index in the raw_subset is its length minus the window size.
        self.num_viable_samples = len(self.raw_subset) - required_window_size + 1
        
        # Check for invalid cases
        if self.num_viable_samples <= 0:
            print(f"Warning: {type(self.raw_subset).__name__} is too small to form a single sample.")
            self.num_viable_samples = 0

    def __len__(self):
        return self.num_viable_samples

    def __getitem__(self, idx):
        """
        Returns a sample from the raw_subset.
        The idx here is for the viable samples within the subset, not the global dataset.
        """
        # Get the global index of the starting day for this sample.
        # This idx is relative to the start of the raw_subset.
        global_start_idx = self.raw_indices[idx]
        
        # The main dataset's __getitem__ takes a global index.
        # We need to call the main dataset's __getitem__ with the correct global index.
        # The main dataset is `self.raw_subset.dataset`.
        
        # The main dataset's __getitem__ method already handles the windowing, so we
        # just need to pass the correct starting index.
        return self.raw_subset.dataset[global_start_idx]

    def __repr__(self):
        return (
            f"<{type(self.raw_subset).__name__} with {self.num_viable_samples} viable samples>"
        )


# In[21]:


from torch.utils.data import DataLoader
from torch.utils.data import Subset

print(f"===== Making the Dataset Class: TRIAL_RUN MODE IS {TRIAL_RUN} ===== ")

# load all the data from one folder
dataset = MonthlyOrDailyNetCDFDataset()

# Patch locations for positional embedding
PATCH_LATLONS_TENSOR = torch.tensor(dataset.patch_latlons, dtype=torch.float32)

print("========== SPLITTING THE DATASET ===================")
# DIFFERENT SUBSET OPTIONS FOR TRAINING / VALIDATION / TESTING for the trial data vs. full dataset
if TRIAL_RUN:
    total_time_steps = len(dataset)
    train_end = int(total_time_steps * 0.7)
    val_end = int(total_time_steps * 0.85)
    
    train_set = Subset(dataset, range(0, train_end))
    val_set   = Subset(dataset, range(train_end, val_end))
    test_set  = Subset(dataset, range(val_end, total_time_steps))
    
else:

    # --- Splitting by Valid Years (considering the context_length and forecast_horizon) ---

    # Convert dataset.times to pandas DatetimeIndex for easier year-based filtering
    all_times_pd = pd.to_datetime(dataset.times)

    # Define the start and end years for each set - keep this for the full dataset
    train_start_year = 1850
    train_end_year = 2012   
    val_start_year = 2013
    val_end_year = 2018
    test_start_year = 2019
    test_end_year = 2024

    # Now use these times for year-based splitting
    all_times_pd = pd.to_datetime(dataset.times)
    train_mask = (all_times_pd.year >= train_start_year) & (all_times_pd.year <= train_end_year)
    val_mask = (all_times_pd.year >= val_start_year) & (all_times_pd.year <= val_end_year)
    test_mask = (all_times_pd.year >= test_start_year) & (all_times_pd.year <= test_end_year)
    
    # Get the integer indices where the masks are True
    train_indices = np.where(train_mask)[0].tolist()
    val_indices = np.where(val_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()
    
    # Create the raw subsets first
    raw_train_set = Subset(dataset, train_indices)
    raw_val_set = Subset(dataset, val_indices)
    raw_test_set = Subset(dataset, test_indices)

    # The Subset now gets indices that are valid for the *sliding window*
    # Now, wrap the raw subsets in our new YearlySubset class to get valid windows
    train_set = YearlySubset(raw_train_set, CONTEXT_LENGTH, FORECAST_HORIZON)
    val_set = YearlySubset(raw_val_set, CONTEXT_LENGTH, FORECAST_HORIZON)
    test_set = YearlySubset(raw_test_set, CONTEXT_LENGTH, FORECAST_HORIZON)
        
    train_end = train_indices[-1]
    val_end = val_indices[-1]

print("Training data length:   ", len(train_set))
print("Validation data length: ", len(val_set))
print("Testing data length:    ", len(test_set))

total_time_steps = len(train_set) + len(val_set) + len(test_set)
print("Total Time Steps = ", total_time_steps)

print("Number of training batches", len(train_set)//BATCH_SIZE)
print("Number of validation batches", len(val_set)//BATCH_SIZE)
print("Number of test batches after drop_last incomplete batch", len(test_set)//BATCH_SIZE)

print("===== Printing Dataset ===== ")
print(dataset)                 # calls __repr__ → see how many files & time steps loaded

print("===== Sample at dataset[0] ===== ")
input_tensor, target_tensor, start_idx, end_idx, target_start, target_end, year, month, day = dataset[0]

print(f"Fetched start index {start_idx}: Time={dataset.times[start_idx]}")
print(f"Fetched end   index {end_idx}: Time={dataset.times[end_idx]}")

print(f"Fetched target start index {target_start}: Time={dataset.times[target_start]}")
print(f"Fetched target end   index {target_end}: Time={dataset.times[target_end]}")

print(f"Year is {year}")
print(f"Month is {month}")
print(f"Day is {day}")

def print_set_dates(dataset_subset, set_name):
    """ Print start and end dates for each set (Training, Validation, Testing)"""
    if len(dataset_subset) == 0:
        print(f"{set_name} set: No data available.")
        return

    # Get the global index of the first viable sample's start day.
    first_global_idx = dataset_subset.raw_subset.indices[0]

    # The cosmetic end date is simply the last day in the raw subset's indices.
    raw_end_date_idx = dataset_subset.raw_subset.indices[-1]

    # Get the global index of the last viable sample's start day.
    # The last viable sample's start index is relative to the subset, so it's `len(dataset_subset) - 1`.
    # We must map this back to the global index using the raw_indices.
    last_viable_start_idx = len(dataset_subset) - 1
    
    # Ensure there's at least one viable sample before trying to access its index
    if last_viable_start_idx < 0:
        print(f"{set_name} set: Not enough data to form a single viable sample.")
        return

    last_global_idx_of_viable_start = dataset_subset.raw_subset.indices[last_viable_start_idx]

    # The 'actual last viable target day' is the end date of the last full sample's target sequence.
    # This requires adding the context and forecast lengths to the last global start index.
    valid_end_date_idx = last_global_idx_of_viable_start + dataset_subset.context_length + dataset_subset.forecast_horizon - 1
    
    # Fetch the actual datetime objects

    # Note: For the training, validation, and testing sets, each item (idx) represents the *start*
    # of a `context_length + forecast_horizon` window.
    # So, the start date of a set is the `dataset.times` value at the global index of its first item.
    start_date = dataset.times[first_global_idx]
    raw_end_date = dataset.times[raw_end_date_idx]
    valid_end_date = dataset.times[valid_end_date_idx]

    print(f"{set_name} set start date: {start_date}")
    print(f"{set_name} set end date (cosmetic): {raw_end_date}")
    print(f"(actual last viable target day): {valid_end_date}")
    
    logging.info(f"{set_name} set start date: {start_date}")
    logging.info(f"{set_name} set end date (cosmetic): {raw_end_date})")
    logging.info(f"(actual last viable target day): {valid_end_date}")
    
    return str(start_date), str(raw_end_date) # Returning raw_end_date for consistency with previous calls

print("===== Start and End Dates for Each Set =====")
train_set_start_year, train_set_end_year = print_set_dates(train_set, "Training")
val_set_start_year, val_set_end_year = print_set_dates(val_set, "Validation")
test_set_start_year, test_set_end_year = print_set_dates(test_set, "Testing")

train_set_start_year, train_set_end_year = train_set_start_year[:4], train_set_end_year[:4]
val_set_start_year, val_set_end_year = val_set_start_year[:4], val_set_end_year[:4]
test_set_start_year, test_set_end_year = test_set_start_year[:4], test_set_end_year[:4]

print("===== Starting DataLoader ====")
# wrap in a DataLoader
# 1. Use pinned memory for faster asynch transfer to GPUs)
# 2. Use a prefetch factor so that the GPU is fed w/o a ton of CPU memory use
# 3. Use shuffle=False to preserve time order (especially for forecasting)
# 4. Use drop_last=True to prevent it from testing on incomplete batches
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR, drop_last=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR, drop_last=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=PREFETCH_FACTOR, drop_last=True)

print(f"train_loader length: {len(train_loader)}")
print(f"val_loader length: {len(val_loader)}")
print(f"test_loader length: {len(test_loader)}")

if len(val_loader) < 1 or len(test_loader) < 1:
    raise ValueError("Make BATCH_SIZE smaller to avoid zero-length loaders")

print("input_tensor should be of shape (context_length, num_patches, num_features, patch_size)")
print(f"actual input_tensor.shape = {input_tensor.shape}")
print("target_tensor should be of shape (forecast_horizon, num_patches, patch_size)")
print(f"actual target_tensor.shape = {target_tensor.shape}")


# ## Checking the distribution of train, validation, and testing sets
# This takes a long time. Only do this once for the data.

# In[22]:


# --- Conditional Data collection for Training and Validation Sets ---
from NC_FILE_PROCESSING.metrics_and_plots import *

if PLOT_DATA_SPLIT_DISTRIBUTION:
    print("\n--- Collecting ground truth data directly from subsets ---")
    start_time_collect_data_new = time.perf_counter()
    all_train_actual_values = []
    all_val_actual_values = []
    all_test_actual_values = []

    # Iterate directly over the subsets
    for i in range(len(test_set)):
        _, sample_y, *_ = test_set[i]
        all_test_actual_values.append(sample_y.cpu().numpy().flatten())
    final_test_values = np.concatenate(all_test_actual_values)
    print("Finished iterating over test_set")

    for i in range(len(val_set)):
        _, sample_y, *_ = val_set[i]
        all_val_actual_values.append(sample_y.cpu().numpy().flatten())
    final_val_values = np.concatenate(all_val_actual_values)
    print("Finished iterating over val_set")
    
    for i in range(len(train_set)):
        _, sample_y, *_ = train_set[i]
        all_train_actual_values.append(sample_y.cpu().numpy().flatten())
    final_train_values = np.concatenate(all_train_actual_values)
    print("Finished iterating over train_set")
    
    print("--- Finished collecting ground truth data from subsets ---")
    print(f"Elapsed time for collecting ground truth method: {time.perf_counter() - start_time_collect_data_new:.2f} seconds")

else:
    final_train_values = np.array([])
    final_val_values = np.array([])
    final_test_values = np.array([])

# 9. Conditional Plotting All Three SIC Distributions (Train, Val, Test Sets)
if PLOT_DATA_SPLIT_DISTRIBUTION:
    print("\n--- Plotting data distributions (Train, Val, Test) ---")
    start_time_plot_data_dist = time.perf_counter()
    plot_sic_distribution_bars(
        train_data=final_train_values,
        val_data=final_val_values,
        test_data=final_test_values,
        start_date=train_set_start_year,
        end_date=test_set_end_year,
        num_bins=10
    )
    print(f"Elapsed time for plotting data distribution comparison: {time.perf_counter() - start_time_plot_data_dist:.2f} seconds")

    # Calculate Pairwise Jensen-Shannon Distances for Data Splits
    print("\n--- Pairwise Jensen-Shannon Distances for Data Splits ---")
    start_time_jsd_pairwise = time.perf_counter()
    distributions_for_jsd = {
        'train': final_train_values,
        'validation': final_val_values,
        'test': final_test_values
    }
    
    jsd_bins = np.linspace(0, 1, 10 + 1)
    pairwise_jsd_results = jensen_shannon_distance_pairwise(distributions_for_jsd, jsd_bins)
    
    for pair, jsd_val in pairwise_jsd_results.items():
        print(f"JSD ({pair}): {jsd_val:.4f}")
    end_time_jsd_pairwise = time.perf_counter()
    print(f"Elapsed time for Jensen Shannon Pairwise Calculation: {end_time_jsd_pairwise - start_time_jsd_pairwise:.2f} seconds")


# # Positional Encoding

# In[23]:


import torch
import torch.nn as nn
import time

class TemporalPositionalEncoder(nn.Module):
    """
    A positional encoder that handles temporal (year, month, day)
    positional embeddings using sinusoidal encoding.

    Parameters
    ----------
    d_model : int
        The dimension of the model's hidden states (embedding dimension) for
        temporal information. Must be an even number.
    dropout : float, optional
        Dropout rate to apply to the temporal positional embeddings. Defaults to 0.1.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number for this temporal encoding setup.")
        
        # Allocate dimensions, ensuring each is even.
        self.d_model_year = 2 * (d_model // 6)
        self.d_model_month = 2 * (d_model // 6)
        self.d_model_day = d_model - (self.d_model_year + self.d_model_month)
        
        if self.d_model_year % 2 != 0 or self.d_model_month % 2 != 0 or self.d_model_day % 2 != 0:
            raise RuntimeError("Internal error: Temporal dimension allocation resulted in odd dimensions.")

        self.dropout = nn.Dropout(dropout)

        # Base frequencies for sinusoidal encoding, chosen based on the range of each component
        self.year_div_term_base = 10000.0
        self.month_div_term_base = 1000.0
        self.day_div_term_base = 100.0

    def _generate_sinusoidal_encoding(self, positions: torch.Tensor, dim: int, div_base: float) -> torch.Tensor:
        """
        Generates sinusoidal positional encoding for a single component (e.g., year, month, day).
        Expects 'dim' to always be an even number.
        """
        if dim == 0:
            return torch.empty(positions.shape[0], 0, device=positions.device)
        if dim % 2 != 0:
            raise ValueError(f"Dim ({dim}) must be an even number for sinusoidal encoding.")

        position = positions.unsqueeze(1)  # (B, 1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=positions.device) * -(np.log(div_base) / dim))
        
        pe = torch.zeros(positions.shape[0], dim, device=positions.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor,
                years: torch.Tensor, months: torch.Tensor, days: torch.Tensor,
                context_length: int, num_patches: int) -> torch.Tensor:
        """
        Adds temporal positional encodings (year, month, day) to the input tensor.

        Args:
            x (torch.Tensor): Input from patch_embed, shape (B, T*P, d_model).
            years (torch.Tensor): Batch of year values, shape (B,).
            months (torch.Tensor): Batch of month values, shape (B,).
            days (torch.Tensor): Batch of day values, shape (B,).
            context_length (int): The T (time steps) dimension of the input sequence.
            num_patches (int): The P (number of patches) dimension of the input sequence.

        Returns:
            torch.Tensor: Tensor with temporal positional encodings added, shape (B, T*P, d_model).
        """
        # Generate embeddings for each temporal component
        pe_year = self._generate_sinusoidal_encoding(years.float(), self.d_model_year, self.year_div_term_base)  # (B, d_model_year)
        pe_month = self._generate_sinusoidal_encoding(months.float(), self.d_model_month, self.month_div_term_base)  # (B, d_model_month)
        pe_day = self._generate_sinusoidal_encoding(days.float(), self.d_model_day, self.day_div_term_base)  # (B, d_model_day)

        # Concatenate temporal components to form the full temporal embedding for each sequence
        temp_pos_batch = torch.cat([pe_year, pe_month, pe_day], dim=-1)  # (B, d_model_temporal)
        
        # Reshape to match the input tensor `x`
        temporal_pe = temp_pos_batch.unsqueeze(1).unsqueeze(1).expand(-1, context_length, num_patches, -1)
        temporal_pe = temporal_pe.reshape(x.shape[0], x.shape[1], self.d_model)

        x = x + temporal_pe
        x = self.dropout(x)
        return x


class SpatialPositionalEncoder(nn.Module):
    """
    A learnable positional encoder for spatial (patch) information.
    Instead of using a fixed function, this module learns a unique embedding for each patch.

    Parameters
    ----------
    d_model : int
        The dimension of the model's hidden states (embedding dimension) for
        spatial information.
    num_patches : int
        The total number of unique patches.
    """
    def __init__(self, d_model: int, num_patches: int):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        
        # Create a learnable embedding layer for each patch
        self.patch_embeddings = nn.Embedding(num_patches, d_model)

    def forward(self, patch_indices: torch.Tensor) -> torch.Tensor:
        """
        Returns the learnable positional embeddings for the given patch indices.

        Args:
            patch_indices (torch.Tensor): A tensor of shape (B * T * P) containing
                                          the integer indices for each patch.

        Returns:
            torch.Tensor: A tensor of shape (B * T * P, d_model) with the
                          learnable embeddings.
        """
        return self.patch_embeddings(patch_indices)


class CombinedPositionalEncoder(nn.Module):
    """
    A positional encoder that combines temporal and spatial positional embeddings
    from separate modules.

    Parameters
    ----------
    d_model : int
        The dimension of the model's hidden states (embedding dimension).
    num_patches : int
        The total number of unique patches.
    dropout : float, optional
        Dropout rate to apply to the combined positional embeddings. Defaults to 0.1.
    """
    def __init__(self, d_model: int, num_patches: int, dropout: float = 0.1):
        super().__init__()
        
        # Ensure d_model is even for a 50/50 split
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number for this positional encoding setup.")
        
        d_model_temporal = d_model // 2
        d_model_spatial = d_model // 2

        # Initialize the separate temporal and spatial encoding modules
        self.temporal_encoder = TemporalPositionalEncoder(
            d_model=d_model_temporal,
            dropout=dropout
        )
        self.spatial_encoder = SpatialPositionalEncoder(
            d_model=d_model_spatial,
            num_patches=num_patches,
        )
        self.dropout = nn.Dropout(dropout)
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor,
                years: torch.Tensor, months: torch.Tensor, days: torch.Tensor,
                context_length: int) -> torch.Tensor:
        """
        Adds combined temporal and spatial positional encodings to the input.

        Args:
            x (torch.Tensor): Input from patch_embed, shape (B, T*P, d_model).
            years (torch.Tensor): Batch of year values, shape (B,).
            months (torch.Tensor): Batch of month values, shape (B,).
            days (torch.Tensor): Batch of day values, shape (B,).
            context_length (int): The T (time steps) dimension of the input sequence.

        Returns:
            torch.Tensor: Tensor with combined positional encodings, shape (B, T*P, d_model).
        """
        B, seq_len, D = x.shape

        device = x.device
        years = years.to(device)
        months = months.to(device)
        days = days.to(device)

        # Generate temporal encodings
        temp_pe = self.temporal_encoder(
            x=torch.zeros((B, seq_len, self.temporal_encoder.d_model), device=device),
            years=years,
            months=months,
            days=days,
            context_length=context_length,
            num_patches=self.num_patches
        )        
        
        # Generate spatial encodings from learnable embeddings
        patch_indices = torch.arange(self.num_patches, device=device).unsqueeze(0).expand(B * context_length, -1)
        spat_pe = self.spatial_encoder(patch_indices.reshape(-1)).reshape(B, seq_len, self.spatial_encoder.d_model)

        # Combine and add to the input
        combined_pe = torch.cat([temp_pe, spat_pe], dim=-1)
        x = x + combined_pe
        x = self.dropout(x)
        return x


# # Transformer Class

# In[24]:


import torch
import torch.nn as nn
import time

class IceForecastTransformer(nn.Module):
    
    """
    A Transformer-based model for forecasting ice conditions based on sequences of
    historical patch data.

    Parameters
    ----------
    input_patch_features_dim : int
        The dimensionality of the feature vector for each individual patch (ex. 2 features).
        This is the input dimension for the patch embedding layer.
    num_patches : int
        The total number of geographical patches that the `nCells` data was divided into.
        (ex., 256 patches).
    context_length : int, optional
        The number of historical time steps (days or months) to use as input for the transformer.
    forecast_horizon : int, optional
        The number of future time steps (days or months) to predict for each patch.
    d_model : int, optional
        The dimension of the model's hidden states (embedding dimension).
        This is the size of the vectors that flow through the Transformer encoder.
    nhead : int, optional
        The number of attention heads in the multi-head attention mechanism within
        each Transformer encoder layer. Defaults to 8.
    num_layers : int, optional
        The number of Transformer encoder layers in the model. Defaults to 4.

    Attributes
    ----------
    patch_embed : nn.Linear
        Linear layer to project input patch features into the `d_model` hidden space.
    encoder : nn.TransformerEncoder
        The Transformer encoder module composed of `num_layers` encoder layers.
    mlp_head : nn.Sequential
        A multi-layer perceptron head for outputting predictions for each patch.
    """
    
    def __init__(self,
                 input_patch_features_dim: int = PATCH_EMBEDDING_INPUT_DIM, # D: The flat feature dimension of a single patch (ex., 512)
                 num_patches: int = NUM_PATCHES,  # P: Number of spatial patches
                 context_length: int = CONTEXT_LENGTH, # T: Number of historical time steps
                 forecast_horizon: int = FORECAST_HORIZON, # Number of future time steps to predict (usually 1)
                 d_model: int = D_MODEL,        # d_model: Transformer's embedding dimension
                 nhead: int = N_HEAD,           # nhead: Number of attention heads
                 num_layers: int = NUM_TRANSFORMER_LAYERS, # num_layers: Number of TransformerEncoderLayers
                 dropout: float = 0.1,  
                 dim_feedforward: int = PATCH_EMBEDDING_INPUT_DIM, 
                 activation: str = "gelu",
                 ):

        super().__init__()

        """
        The transformer should
        1. Accept a sequence of data (ex. 3 months of patches). 
           The context_length parameter says how many time steps to use for input.
        2. Encode each patch with the transformer.
        3. Output the patches for regression (ex. predict the 4th month).
           The forecast_horizon parameter says how many time steps to use for the output prediction.
        
        """

        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.num_patches = num_patches
        self.d_model = d_model
        self.input_patch_features_dim = input_patch_features_dim
   
        print("Calling IceForecastTransformer __init__")
        start_time = time.perf_counter()

        # Patch embedding layer: projects the raw patch features (512)
        # into d_model (128) hidden space dimension
        self.patch_embed = nn.Linear(input_patch_features_dim, d_model)

        # # --- Positional Encoding Layer (NOT LEARNABLE)
        # if patch_latlons is None:
        #     raise ValueError("patch_latlons must be provided for positional encoding.")

        # # Assert that num_patches matches the provided patch_latlons
        # if num_patches != patch_latlons.shape[0]:
        #     raise ValueError(f"num_patches ({num_patches}) does not match "
        #                      f"patch_latlons.shape[0] ({patch_latlons.shape[0]})")

        # Initialize only the spatial encoder
        self.pos_encoder = SpatialPositionalEncoder(d_model=d_model, num_patches=num_patches)
        
        # Transformer Encoder
        # batch_first=True means input/output tensors are (batch, sequence, features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,               
            activation=activation,          
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output MLP head: (B, P, CELLS_PER_PATCH * forecast_horizon)
        # Make a prediction for every cell per patch. 
        # The Sigmoid is CRITICAL. It ensures there are no out of bounds predictions.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, CELLS_PER_PATCH * forecast_horizon),
            nn.Sigmoid()
        )

        end_time = time.perf_counter()
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")
        print("End of IceForecastTransformer __init__")
        
    def forward(self, x, years: torch.Tensor, months: torch.Tensor, days: torch.Tensor):
        
        """
        B = Batch size
        T = Time (context_length)
        P = Patch count
        D = Patch Dimension (cells per patch * feature count)
        x: Tensor of shape (B, T, P, D)
        Output: Tensor of shape (batch_size, forecast_horizon, num_patches)
        Output: (B, forecast_horizon, P)
        """
        
        # Initial input x shape from DataLoader / pre-processing:
        # (B, T, P, D) i.e., (Batch_Size, Context_Length, Num_Patches, Input_Patch_Features_Dim)
        # Example: (16, 7, 140, 512)
        
        B, T, P, D = x.shape

        # Flatten time and patches for the Transformer Encoder:
        # Each (Time, Patch) combination becomes a single token in the sequence.
        # Output shape: (B, T * P, D)
        # Example: (16, 7 * 140 = 980, 512)
        
        # Flatten time and patches for the Transformer Encoder: (B, T * P, D)
        # This treats each patch at each time step as a distinct token
        x = x.view(B, T * P, D)

        # Project patch features to the transformer's d_model dimension
        x = self.patch_embed(x)  # Output: (B, T * P, d_model) ex., (16, 980, 128)

        # Create a tensor of patch indices
        patch_indices = torch.arange(P, device=x.device).unsqueeze(0).expand(B * T, -1).reshape(-1)
        
        # Add only the spatial positional embeddings
        x = x + self.pos_encoder(patch_indices).reshape(B, T * P, self.d_model)

        # Apply transformer encoder layers
        x = self.encoder(x)      # Output: (B, T * P, d_model) ex., (16, 980, 128)

        # Reshape back to separate time and patches: (B, T, P, d_model) ex., (16, 7, 140, 128)
        x = x.view(B, T, P, self.d_model) 

        # Mean pooling over the time (context_length) dimension for each patch.
        # This aggregates information from all historical time steps for each patch's final prediction.        
        x = x.mean(dim=1)  # Output: (B, P, d_model) ex., (16, 140, 128)

        # Apply MLP head to predict values for each cell in each patch
        # The MLP head outputs (B, P, CELLS_PER_PATCH * forecast_horizon)
        x = self.mlp_head(x) # ex. (16, 140, 256 * 3) = (16, 140, 768)

        # Reshape the output to (B, forecast_horizon, P, CELLS_PER_PATCH)
        # Explicitly reshape the last dimension to seperate the forecast horizon out
        x = x.view(B, P, self.forecast_horizon, CELLS_PER_PATCH) # Reshape into forecast_horizon and CELLS_PER_PATCH
        x = x.permute(0, 2, 1, 3) # Permute to (B, forecast_horizon, P, CELLS_PER_PATCH)

        return x


# # Training Loop

# In[25]:


if TRAINING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch import Tensor
    import torch.nn.functional as F
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = IceForecastTransformer()
    
    # Wrap the model with DataParallel if there's more than one GPU available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model.to(device)
    
    print("\n--- Model Architecture ---")
    print(model)
    print("--------------------------\n")
    
    logging.info("\n--- Model Architecture ---")
    logging.info(str(model)) # Log the full model structure
    logging.info(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logging.info("--------------------------\n")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    logging.info("== TIMER IS STARTING FOR TRAINING ==")
    start_time = time.perf_counter()
    logging.info("===============================")
    logging.info("       STARTING EPOCHS       ")
    logging.info("===============================")
    logging.info(f"Number of epochs: {NUM_EPOCHS}")
    logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
    
        # Unpack the train_loader
        for batch_idx, (input_tensor, target_tensor,
                        start_idx, end_idx, target_start, target_end,
                        years, months, days) in enumerate(train_loader):
    
            # Move input and target to the device
            # x: (B, context_length, num_patches, num_features, patch_size)
            # y: (B, forecast_horizon, num_patches, CELLS_PER_PATCH)
            x = input_tensor.to(device)  # Shape: (B, T, P, C, L)
            y = target_tensor.to(device)  # Shape: (B, forecast_horizon, P, L)
            years = years.to(device)
            months = months.to(device)
            days = days.to(device)
            
            # Reshape x for transformer input
            B, T, P, C, L = x.shape
            x_reshaped_for_transformer_D = x.view(B, T, P, C * L)
            
            # Run through transformer - y_pred is (B, forecast_horizon, num_patches, CELLS_PER_PATCH) 
            # Pass the input tensor and the temporal components
            y_pred = model(x_reshaped_for_transformer_D, years, months, days)
            
            # Compute loss
            loss = criterion(y_pred, y) # y_pred and y should now have identical shapes (B, FH, P, CPC)
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}") # Keep print for immediate console feedback
    
        # --- Validation loop ---
        model.eval()
        val_loss = 0

        with torch.no_grad():
            
            # Unpack the validation loader
            for batch_idx_val, (x_val, y_val, start_idx_val, end_idx_val, target_start_val, target_end_val, 
                                years_val, months_val, days_val) in enumerate(val_loader): 
    
                # Move to GPU if available
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                years_val = years_val.to(device)
                months_val = months_val.to(device)
                days_val = days_val.to(device)
    
                # Extract dimensions from x_val for reshaping
                # x_val before reshaping: (B_val, T_val, P_val, C_val, L_val)
                B_val, T_val, P_val, C_val, L_val = x_val.shape
                
                # Reshape x_val for transformer input
                x_val_reshaped_for_transformer_input = x_val.view(B_val, T_val, P_val, C_val * L_val)
    
                # Model output is (B, forecast_horizon, P, CELLS_PER_PATCH)
                y_val_pred = model(x_val_reshaped_for_transformer_input, years_val, months_val, days_val)
    
                # Compute validation loss (y_val_pred and y_val should have identical shapes)
                val_loss += criterion(y_val_pred, y_val).item() # y_val is (B, forecast_horizon, P, L)
        
        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}") # Keep print for immediate console feedback
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    logging.info("===============================================")
    logging.info(f"Elapsed time for TRAINING: {elapsed_time:.2f} seconds")
    logging.info("===============================================")
    print("===============================================")
    print(f"Elapsed time for TRAINING: {elapsed_time:.2f} seconds")
    print("===============================================")

else:
    print("SKIPPED TRAINING")


# TODO OPTION: Try temporal attention only (ex., Informer, Time Series Transformer).
# 
# # Save the Model

# In[26]:


# Define the path where to save or load the model
PATH = f"{model_version}_model.pth"

if TRAINING:
    # Save the model's state_dict
    torch.save(model.state_dict(), PATH)
    print(f"Saved model at {PATH}")

else:
    print("SKIPPED SAVING THE MODEL")


# # === BELOW - CAN BE USED ANY TIME FROM A .PTH FILE
# 
# Make sure and run the cells that contain constants or run all, but comment out the "save" and the training loop cell.

# # Re-Load the Model

# In[27]:


from NC_FILE_PROCESSING.metrics_and_plots import *

if EVALUATING_ON:

    import torch
    import torch.nn as nn
    
    if not torch.cuda.is_available():
        raise ValueError("There is a problem with Torch not recognizing the GPUs")

    print(PATH)

    # Instantiate the model (must have the same architecture as when it was saved)
    # Create an identical instance of the original __init__ parameters
    loaded_model = IceForecastTransformer()
    
    # Load the saved state_dict
    # weights_only=True is good practice for safety
    state_dict = torch.load(PATH, weights_only=True)

    # Create an ordered dictionary to store the keys
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    # Loop through the state_dict and remove the "module." prefix because of the multiple GPUs used
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # Remove the "module." prefix
            new_state_dict[name] = v
        else:
            # In case some keys don't have the prefix (though unlikely for a DataParallel model)
            new_state_dict[k] = v

    # Load the modified state_dict into the model
    loaded_model.load_state_dict(new_state_dict)
    
    # Set the model to evaluation mode
    loaded_model.eval()
    
    # Move the model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    
    print("Model loaded successfully!")


# # Metrics

# In[28]:


from scipy.stats import entropy

if FAST_EVAL_ON:    

    # Accumulators for errors
    all_abs_errors = [] # To store absolute errors for each cell in each patch
    all_mse_errors = [] # To store MSE for each cell in each patch
    
    # Accumulators for histogram data
    all_predicted_values_flat = []
    all_actual_values_flat = []
    
    print("\nStarting evaluation and metric calculation...")
    logging.info("\nStarting evaluation and metric calculation...")
    print("==================")
    print(f"DEBUG: Batch Size: {BATCH_SIZE}")
    print(f"DEBUG: Context Length: {CONTEXT_LENGTH}")
    print(f"DEBUG: Forecast Horizon: {FORECAST_HORIZON}")
    print(f"DEBUG: Number of batches in test_loader (with drop_last=True): {len(test_loader)} Batches")
    print("==================")
    print(f"DEBUG: len(test_set): {len(test_set)} time steps")
    print(f"DEBUG: len(dataset) for splitting: {len(dataset)} time steps")
    print(f"DEBUG: train_end: {train_end}")
    print(f"DEBUG: val_end: {val_end}")
    print(f"DEBUG: range for test_set: {range(val_end, total_time_steps)}") 
    # Should be range(302, 356) for daily and range(2027, 2058) for monthly
    print("==================")

    start_time_metrics = time.perf_counter()
    
    for batch_idx, (sample_x, sample_y, 
                    start_idx, end_idx, target_start, target_end) in enumerate(test_loader):
        print(f"Processing batch {batch_idx+1}/{len(test_loader)}")
    
        # Move to device and apply initial reshape as done in training
        sample_x = sample_x.to(device)
        sample_y = sample_y.to(device) # Actual target values

        # Initial reshape of x for the Transformer model
        B_sample, T_sample, P_sample, C_sample, L_sample = sample_x.shape
        sample_x_reshaped = sample_x.view(B_sample, T_sample, P_sample, C_sample * L_sample)

        # Perform inference
        with torch.no_grad(): # Essential for inference to disable gradient calculations
            predicted_y_patches = loaded_model(sample_x_reshaped)

        # Ensure predicted_y_patches and sample_y have the same shape for comparison
        # Expected shape: (B, forecast_horizon, NUM_PATCHES, CELLS_PER_PATCH)
        if predicted_y_patches.shape != sample_y.shape:
            print(f"Shape mismatch: Predicted {predicted_y_patches.shape}, Actual {sample_y.shape}")
            continue # Skip this batch if shapes are incompatible

        # Calculate errors for each cell in each patch, across the forecast horizon and batch
        # The errors will implicitly be averaged over the batch when we take the mean later
        diff = predicted_y_patches - sample_y
        abs_error_batch = torch.abs(diff)
        mse_error_batch = diff ** 2

        # Accumulate errors (move to CPU for storage if memory is a concern)
        all_abs_errors.append(abs_error_batch.cpu())
        all_mse_errors.append(mse_error_batch.cpu())

        # Collect data for histograms (flatten all values)
        all_predicted_values_flat.append(predicted_y_patches.cpu().numpy().flatten())
        all_actual_values_flat.append(sample_y.cpu().numpy().flatten())

    # --- Final Data Concatenation ---
    if all_abs_errors and all_mse_errors:
        combined_abs_errors = torch.cat(all_abs_errors, dim=0)
        combined_mse_errors = torch.cat(all_mse_errors, dim=0)
        mean_abs_error_per_cell_patch = combined_abs_errors.mean(dim=(0, 1)).numpy()
        mean_mse_per_cell_patch = combined_mse_errors.mean(dim=(0, 1)).numpy()

    if all_predicted_values_flat and all_actual_values_flat:
        final_predicted_values = np.concatenate(all_predicted_values_flat)
        final_actual_values = np.concatenate(all_actual_values_flat)

    #######
    # SIC #
    #######
    
    print("\n--- Error Metrics (Averaged per Cell per Patch) ---")
    print(f"Mean Absolute Error (shape {mean_abs_error_per_cell_patch.shape}):")
    # print(mean_abs_error_per_cell_patch) # Uncomment to see the full tensor
    print(f"Overall Mean Absolute Error:            {mean_abs_error_per_cell_patch.mean().item():.4f}")

    print(f"\nMean Squared Error (shape {mean_mse_per_cell_patch.shape}):")
    # print(mean_mse_per_cell_patch) # Uncomment to see the full tensor

    mse = mean_mse_per_cell_patch.mean().item()
    print(f"Overall Mean Squared Error:             {mse:.4f}")

    rmse = np.sqrt(mse)
    print(f"Overall Root Mean Squared Error (RMSE): {rmse}")

    # --- Save error arrays for later use with Cartopy ---
    print("\n--- Saving Error Arrays ---")
    np.save(f"{model_version}_MAE_per_cell_patch.npy", mean_abs_error_per_cell_patch)
    np.save(f"{model_version}_MSE_per_cell_patch.npy", mean_mse_per_cell_patch)
    print(f"Mean ABS Error array saved as {model_version}_MAE_per_cell_patch.npy")
    print(f"Mean MSE Error array saved as {model_version}_MSE_per_cell_patch.npy")

    #######
    # SIE #
    #######
    
    # --- Sea Ice Extent (SIE) Prediction with Sklearn/Seaborn ---
    print("\n--- Sea Ice Extent (SIE) Metrics (Threshold > 0.15) ---")

    # Apply the threshold for sea ice extent (SIE)
    threshold_cm = 0.15 
    sie_actual_flat = np.where(final_actual_values > threshold_cm, 1, 0)
    sie_predicted_flat = np.where(final_predicted_values > threshold_cm, 1, 0)

    # Calculate and print classification report
    print("\nClassification Report:")
    report = classification_report(sie_actual_flat, sie_predicted_flat, target_names=['No Ice', 'Ice'], zero_division=0)
    print(report)

    # --- Plotting Confusion Matrix with Percentages ---
    cm = confusion_matrix(sie_actual_flat, sie_predicted_flat)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = np.array([
        [f"{cm[0, 0]:,}\n({cm_percent[0, 0]:.2%})", f"{cm[0, 1]:,}\n({cm_percent[0, 1]:.2%})"],
        [f"{cm[1, 0]:,}\n({cm_percent[1, 0]:.2%})", f"{cm[1, 1]:,}\n({cm_percent[1, 1]:.2%})"]
    ])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', xticklabels=['No Ice', 'Ice'], yticklabels=['No Ice', 'Ice'], ax=ax)
    ax.set_title(f'Sea Ice Extent Confusion Matrix (Threshold > {threshold_cm}) for {patching_strategy_abbr}', fontsize=16)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{model_version}_SIE_Confusion_Matrix.png")
    plt.close()
    print(f"\nConfusion matrix plot saved as {model_version}_SIE_Confusion_Matrix.png")

    # --- ROC Curve and AUC Calculation with Sklearn ---
    print("\n--- ROC Curve and AUC Metrics ---")

    # Use the continuous predicted values and binary actual values
    y_true_binary = np.where(final_actual_values > 0.15, 1, 0)
    y_scores = final_predicted_values

    # Calculate ROC curve data
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_scores)

    # Calculate AUC score
    auc = roc_auc_score(y_true_binary, y_scores)

    # Plot the ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve {patching_strategy_abbr}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"{model_version}_ROC_Curve.png")
    plt.close()

    print(f"\nArea Under the Curve (AUC): {auc:.4f}")
    print(f"ROC curve plot saved as {model_version}_ROC_Curve.png")
    # --- End of Sklearn ROC Curve Section ---

    print(f"Elapsed time for metrics: {time.perf_counter() - start_time_metrics} seconds")


# In[ ]:


from NC_FILE_PROCESSING.metrics_and_plots import *
if SLOW_EVAL_ON:

    start_full_evaluation = time.perf_counter()
    
    # Create lists to store errors with month metadata
    all_predicted_values = []
    all_actual_values = []
    all_abs_errors_with_month = []
    all_mse_errors_with_month = []
    all_predicted_values_with_month = []
    all_actual_values_with_month = []
    mse_per_step = {step: [] for step in range(1, FORECAST_HORIZON + 1)}
    monthly_step_mse_data = {}
    worst_rmse_list = [] # Format: (RMSE_value, 'YYYY-MM-DD', forecast_step)
    
    for batch_idx, (sample_x, sample_y, start_idx, end_idx, target_start, target_end) in enumerate(test_loader):
        
        print(f"Processing batch {batch_idx+1}/{len(test_loader)}")
    
        # Move to device and apply initial reshape as done in training
        sample_x = sample_x.to(device)
        sample_y = sample_y.to(device) # Actual target values

        # Initial reshape of x for the Transformer model
        B_sample, T_sample, P_sample, C_sample, L_sample = sample_x.shape
        sample_x_reshaped = sample_x.view(B_sample, T_sample, P_sample, C_sample * L_sample)

        # Perform inference
        with torch.no_grad(): # Essential for inference to disable gradient calculations
            predicted_y_patches = loaded_model(sample_x_reshaped)

        # Ensure predicted_y_patches and sample_y have the same shape for comparison
        # Expected shape: (B, forecast_horizon, NUM_PATCHES, CELLS_PER_PATCH)
        if predicted_y_patches.shape != sample_y.shape:
            print(f"Shape mismatch: Predicted {predicted_y_patches.shape}, Actual {sample_y.shape}")
            continue # Skip this batch if shapes are incompatible

        # Calculate errors for each cell in each patch, across the forecast horizon and batch
        # The errors will implicitly be averaged over the batch when we take the mean later
        diff = predicted_y_patches - sample_y
        abs_error_batch = torch.abs(diff)
        mse_error_batch = diff ** 2

        # Loop through each item in the batch
        for i in range(B_sample):
            
            # Get the start date for this specific sample
            start_time_np = dataset.times[start_idx[i]]
        
            # Cast start_time_np to Month precision to allow addition with np.timedelta64(step, 'M')
            start_time_month_np = start_time_np.astype('datetime64[M]') 
        
            start_year  = start_time_np.astype("datetime64[Y]").astype(int) + 1970
            start_month = start_time_np.astype("datetime64[M]").astype(int) % 12 + 1
            start_day   = (start_time_np.astype("datetime64[D]") - start_time_np.astype("datetime64[M]")).astype(int) + 1
            

            # Loop through the forecast horizon to get the month for each forecast step
            for step in range(FORECAST_HORIZON):

                # The forecast date is the start date plus the forecast step (in months)
                # Use the month-precision date object for addition
                forecast_date_np = start_time_month_np + np.timedelta64(step, 'M') 
                forecast_date_str = str(forecast_date_np).split('T')[0] # Get 'YYYY-MM-DD'
                
                # NOTE: start_month is a NumPy scalar, ensure it's converted to a Python int if used outside NumPy math
                forecast_month = (start_month.item() + step - 1) % 12 + 1
                forecast_step = step + 1
                
                # Select the squared errors for this specific sample and forecast step
                mse_error_step = mse_error_batch[i, step, :, :].cpu().numpy()
                
                # --- Calculate RMSE for this single prediction ---
                # RMSE for this sample/step is the sqrt of the mean of all cell-wise squared errors
                current_rmse = np.sqrt(np.mean(mse_error_step))
                
                # --- Store the RMSE and metadata in the list ---
                worst_rmse_list.append({
                    'rmse': current_rmse,
                    'date': forecast_date_str,
                    'step': forecast_step
                })

                # Select the data for this specific sample and forecast step
                predicted_step = predicted_y_patches[i, step, :, :].cpu().numpy()
                actual_step = sample_y[i, step, :, :].cpu().numpy()
                # abs_error_step = abs_error_batch[i, step, :, :].cpu().numpy()

                # Collect flattened arrays for overall JSD calculation
                all_predicted_values.append(predicted_step.flatten())
                all_actual_values.append(actual_step.flatten())

                # Create a unique key for the dictionary
                data_key = (forecast_month, forecast_step)
    
                # Initialize a list if the key doesn't exist
                if data_key not in monthly_step_mse_data:
                    monthly_step_mse_data[data_key] = []

                # Append the flattened errors
                monthly_step_mse_data[data_key].extend(mse_error_step.flatten())

                # Append the flattened errors to the list for the current forecast step
                # Note: The forecast step is `step + 1` since `step` starts from 0.
                mse_per_step[step + 1].extend(mse_error_step.flatten())
                
                # # Flatten the data for this step and store it with the month
                # all_abs_errors_with_month.append({
                #     'month': forecast_month,
                #     'errors': abs_error_step.flatten()
                # })
                # all_predicted_values_with_month.append({
                #     'month': forecast_month,
                #     'values': predicted_step.flatten()
                # })
                # all_actual_values_with_month.append({
                #     'month': forecast_month,
                #     'values': actual_step.flatten()
                # })
    
                # Flatten the squared errors for this step and store it with the month
                all_mse_errors_with_month.append({
                    'month': forecast_month,
                    'errors': mse_error_step.flatten()
                })

    ###########
    #   SIC   #
    ###########

    # --- Group errors by month ---
    monthly_mse = {month: [] for month in range(1, 13)}
    for item in all_mse_errors_with_month:
        monthly_mse[item['month']].extend(item['errors'])
    
    # --- Calculate and Plot Monthly Degradation (RMSE) ---
    monthly_rmse = {month: np.sqrt(np.mean(errors)) for month, errors in monthly_mse.items() if errors}
    months = sorted(monthly_rmse.keys())
    rmse_values = [monthly_rmse[m] for m in months]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=months, y=rmse_values, ax=ax, palette="rocket")
    ax.set_title(f"{model_version} \n Root Mean Squared Error by Forecasted Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(f"{model_version}_monthly_RMSE.png")
    plt.close()
    print(f"Monthly RMSE plot saved as {model_version}_monthly_RMSE.png")
    
    ## Temporal Degradation Plot (RMSE vs. Forecast Horizon)
    
    print("\n--- Calculating Temporal Degradation Metrics ---")
    
    # Calculate the RMSE for each forecast step by taking the mean of all squared errors
    rmse_per_step = []
    forecast_steps = sorted(mse_per_step.keys())
    
    for step in forecast_steps:
        if mse_per_step[step]:
            avg_mse = np.mean(mse_per_step[step])
            rmse_per_step.append(np.sqrt(avg_mse))
        else:
            rmse_per_step.append(np.nan) # Handle cases with no data

    # --- Save the plot data ---
    plot_data = {
        'forecast_steps': forecast_steps,
        'rmse_per_step': rmse_per_step
    }
    np.save(f'{model_version}_temporal_degradation_data.npy', plot_data)
    
    # --- Plot Degradation Curve (RMSE vs. Forecast Step) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.set_xlabel('Forecast Horizon (Months)')
    ax.set_ylabel('RMSE')
    ax.plot(forecast_steps, rmse_per_step, color='tab:blue', marker='o', linestyle='-', label='RMSE')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.set_xticks(forecast_steps)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_title(f'{model_version} \n SIC Temporal Degradation Per Month')
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'{model_version}_SIC_temporal_degradation.png')
    plt.close()
    
    print(f"Plot saved: {model_version}_SIC_temporal_degradation.png")
    print(f"Elapsed time for full evaluation: {time.perf_counter() - start_full_evaluation} seconds")

    # --- Calculate average RMSE for each month and forecast step ---
    avg_monthly_rmse = {}
    for key, mse_values in monthly_step_mse_data.items():
        if mse_values:
            avg_mse = np.mean(mse_values)
            avg_monthly_rmse[key] = np.sqrt(avg_mse)
    
    # --- Prepare data for heatmap ---
    all_months = sorted(list(set(k[0] for k in avg_monthly_rmse.keys())))
    all_forecast_steps = sorted(list(set(k[1] for k in avg_monthly_rmse.keys())))
    
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(all_months), len(all_forecast_steps)))
    
    for (month, step), rmse in avg_monthly_rmse.items():
        row_idx = all_months.index(month)
        col_idx = all_forecast_steps.index(step)
        heatmap_data[row_idx, col_idx] = rmse
    
    # --- Plot the heatmap with the color bar/legend ---
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # The cbar_kws argument adds the color bar to the side of the plot and sets its label.
    sns.heatmap(
        heatmap_data, 
        cmap='viridis', # Choose a suitable colormap
        ax=ax, 
        xticklabels=all_forecast_steps, 
        yticklabels=all_months,
        cbar_kws={'label': 'Root Mean Squared Error (RMSE)'} # The key addition for the label
    )
    
    ax.set_title(f"RMSE Degradation by Month and Forecast Horizon")
    ax.set_xlabel("Forecast Horizon (Months)")
    ax.set_ylabel("Month")
    
    # Set the y-axis labels to use month names for better readability
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_yticklabels([month_names[m-1] for m in all_months], rotation=0)

    plt.tight_layout()
    plt.savefig(f"{model_version}_heatmap_RMSE_degradation.png")
    plt.close()
    print(f"Monthly and step RMSE degradation heatmap saved.")

    # WORST RSME SCORES
    sorted_worst_rmse = sorted(worst_rmse_list, key=lambda x: x['rmse'], reverse=True)
    
    # Select the top N worst dates
    TOP_N = 30
    top_worst_dates = sorted_worst_rmse[:TOP_N]
    
    print("\n" + "="*50)
    print(f"       TOP {TOP_N} WORST RMSE DATES AND FORECAST STEPS")
    print("="*50)
    print(f"{'RMSE':<10} {'Date':<15} {'Forecast Step':<15}")
    print("-" * 40)
    for entry in top_worst_dates:
        print(f"{entry['rmse']:.4f}   {entry['date']:<15} {entry['step']:<15}")
    print("="*50)

    # --- JENSEN-SHANNON DIFFERENCE (JSD) CALCULATION ---
    
    print("\n--- Calculating Overall Distribution Metrics (JSD) ---")
    
    # Concatenate all predicted and actual arrays into two massive 1D arrays
    final_predicted_flat = np.concatenate(all_predicted_values)
    final_actual_flat = np.concatenate(all_actual_values)
    
    plot_actual_vs_predicted_sic_distribution(
        predicted_flat=final_predicted_flat,
        actual_flat=final_actual_flat,
        model_version=model_version,
        patching_strategy_abbr=patching_strategy_abbr,
        title_suffix=" (Overall Distribution)"
    )

    ###########
    #   SIE   #
    ###########

    # # --- Group SIE data by month and plot ---
    # monthly_sie_data = {month: {'predicted': [], 'actual': []} for month in range(1, 13)}
    # for i in range(len(all_predicted_values_with_month)):
    #     month = all_predicted_values_with_month[i]['month']
    #     monthly_sie_data[month]['predicted'].extend(all_predicted_values_with_month[i]['values'])
    #     monthly_sie_data[month]['actual'].extend(all_actual_values_with_month[i]['values'])
    
    # # --- Loop through months to generate per-month SIE metrics ---
    # for month, data in monthly_sie_data.items():
    #     if not data['actual']:
    #         continue
        
    #     # Convert to NumPy arrays for calculation
    #     actual_values = np.array(data['actual'])
    #     predicted_values = np.array(data['predicted'])
    
    #     # Apply threshold for SIE
    #     sie_actual = (actual_values > 0.15).astype(int)
    #     sie_predicted = (predicted_values > 0.15).astype(int)
    
    #     # Print Classification Report
    #     print(f"\n--- SIE Classification Report for Month {month} ---")
    #     print(classification_report(sie_actual, sie_predicted, target_names=['No Ice', 'Ice'], zero_division=0))
    
    #     # Plot Confusion Matrix
    #     cm = confusion_matrix(sie_actual, sie_predicted)
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Ice', 'Ice'], yticklabels=['No Ice', 'Ice'])
    #     plt.title(f'Confusion Matrix - Month {month}')
    #     plt.xlabel('Predicted Label')
    #     plt.ylabel('True Label')
    #     plt.savefig(f"{model_version}_cmatrix_{month}.png")
    #     plt.close()


# # Make a Single Prediction

# In[29]:



    if MAP_WITH_CARTOPY_ON:
        # Load one batch
        data_iter = iter(test_loader)
        sample_x, sample_y, start_idx, end_idx, target_start, target_end, years, months, days = next(data_iter)
        
        print(f"Shape of sample_x {sample_x.shape}")
        print(f"Shape of sample_y {sample_y.shape}")   
        
        print(f"Fetched sample_x start index {start_idx}: Time={dataset.times[start_idx]}")
        print(f"Fetched sample_x end   index {end_idx}:   Time={dataset.times[end_idx]}")
        
        print(f"Fetched sample_y (target) start index {target_end}: Time={dataset.times[target_end]}")
        print(f"Fetched sample_y (target) end   index {target_end}: Time={dataset.times[target_end]}")
        
        # Move to device and apply initial reshape as done in training
        sample_x = sample_x.to(device)
        sample_y = sample_y.to(device) # Keep sample_y for actual comparison
        
        # Initial reshape of x for the Transformer model
        B_sample, T_sample, P_sample, C_sample, L_sample = sample_x.shape
        sample_x_reshaped = sample_x.view(B_sample, T_sample, P_sample, C_sample * L_sample)
        
        print(f"Sample x for inference shape (reshaped): {sample_x_reshaped.shape}")
        
        # Perform inference
        with torch.no_grad(): # Essential for inference to disable gradient calculations
            predicted_y_patches = loaded_model(sample_x_reshaped, years, months, days)
        
        print(f"Predicted y patches shape: {predicted_y_patches.shape}")
        print(f"Expected shape: (B, forecast_horizon, NUM_PATCHES, CELLS_PER_PATCH) ex., (16, {loaded_model.forecast_horizon}, 140, 256)")
                         
        # Option 1: Select a specific day from the forecast horizon (ex., the first day)
        # This is the shape (B, NUM_PATCHES, CELLS_PER_PATCH) for that specific day.
        predicted_for_day_0 = predicted_y_patches[:, 0, :, :].cpu()
        print(f"Predicted ice area for Day 0 (specific day) shape: {predicted_for_day_0.shape}")
        
        # Ensure sample_y has the same structure
        actual_for_day_0 = sample_y[:, 0, :, :].cpu()
        print(f"Actual ice area for Day 0 (specific day) shape: {actual_for_day_0.shape}")
        
        # Save predictions so that I can use cartopy by switching kernels for the next jupyter cell
        np.save(f'patches/ice_area_patches_predicted_{PATH}_day0.npy', predicted_for_day_0)
        np.save(f'patches/ice_area_patches_actual_{PATH}_day0.npy', actual_for_day_0)
    
        # Option 2: Iterate through all forecast days
        all_predicted_ice_areas = []
        all_actual_ice_areas = []
        
        for time_step_idx in range(loaded_model.forecast_horizon):
            predicted_day = predicted_y_patches[:, time_step_idx, :, :].cpu()
            all_predicted_ice_areas.append(predicted_day)
        
            actual_day = sample_y[:, time_step_idx, :, :].cpu()
            all_actual_ice_areas.append(actual_day)
        
            print(f"Processing forecast day {time_step_idx}: Predicted shape {predicted_day.shape}, Actual shape {actual_day.shape}")
        
            # Save each day's prediction/actual data if needed
            # np.save(f'patches/ice_area_patches_predicted_day{time_step_idx}.npy', predicted_day)
            # np.save(f'patches/ice_area_patches_actual_day{time_step_idx}.npy', actual_day)


# # Recover nCells from Patches for Visualization

# In[30]:


if MAP_WITH_CARTOPY_ON:

    ########################################
    # SWAP KERNELS IN THE JUPYTER NOTEBOOK #
    ########################################
    
    from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
    from NC_FILE_PROCESSING.nc_utility_functions import *
    from NC_FILE_PROCESSING.patchify_utils import *
    
    import numpy as np
    
    predicted_ice_area_patches = np.load(f'patches/SIC_predicted_{model_version}_day0.npy')
    actual_y_ice_area_patches = np.load(f'patches/SIC_actual_{model_version}_day0.npy')
    
    NUM_PATCHES = len(predicted_ice_area_patches[0])
    print("NUM_PATCHES is", NUM_PATCHES)
    
    latCell, lonCell = load_mesh(perlmutterpathMesh)
    TOTAL_GRID_CELLS = len(lonCell) 
    cell_mask = latCell >= LATITUDE_THRESHOLD
    
    # Extract Freeboard (index 0) and Ice Area (index 1) for predicted and actual
    # Predicted output is (B, 1, NUM_PATCHES, CELLS_PER_PATCH)
    # The model predicts ice area, which is the second feature (index 1)
    # if the output of the model aligns with the order of features *within* the original patch_dim.
    
    # Load the original patch-to-cell mapping
    # indices_per_patch_id = [
    #     [idx_cell_0_0, ..., idx_cell_0_255],
    #     [idx_cell_1_0, ..., idx_cell_1_255],
    #     ...
    # ]
    
    full_nCells_patch_ids, indices_per_patch_id, patch_latlons = patchify_by_latlon_spillover(
                latCell, lonCell, k=256, max_patches=NUM_PATCHES, LATITUDE_THRESHOLD=LATITUDE_THRESHOLD)
    
    # Select one sample from the batch for visualization (ex., the first one)
    # Output is (NUM_PATCHES, CELLS_PER_PATCH) for this single sample
    sample_predicted_cells_per_patch = predicted_ice_area_patches[2] # First item in batch
    sample_actual_cells_per_patch = predicted_ice_area_patches[2] # First item in batch
    
    # Initialize empty arrays for the full grid (nCells)
    recovered_predicted_grid = np.full(TOTAL_GRID_CELLS, np.nan)
    recovered_actual_grid = np.full(TOTAL_GRID_CELLS, np.nan)
    
    # Populate the full grid using the patch data and mapping
    for patch_idx in range(NUM_PATCHES):
        cell_indices_in_patch = indices_per_patch_id[patch_idx]
        
        # For predicted values
        recovered_predicted_grid[cell_indices_in_patch] = sample_predicted_cells_per_patch[patch_idx]
        nan_mask = np.isnan(recovered_predicted_grid)
        nan_count = np.sum(nan_mask)
    
        # For actual values
        recovered_actual_grid[cell_indices_in_patch] = sample_actual_cells_per_patch[patch_idx]
    
    print(f"Recovered predicted grid shape: {recovered_predicted_grid.shape}")
    print(f"Recovered actual grid shape: {recovered_actual_grid.shape}")
    
    fig, northMap = generate_axes_north_pole()
    generate_map_north_pole(fig, northMap, latCell, lonCell, recovered_predicted_grid, f"model {model_version} ice area recovered")
    
    fig, northMap = generate_axes_north_pole()
    generate_map_north_pole(fig, northMap, latCell, lonCell, recovered_actual_grid, f"model {model_version} ice area actual")

