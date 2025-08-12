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


from perlmutterpath import *  # Contains the data_dir and mesh_dir variables
NUM_FEATURES = 2              # C: Number of features per cell (ex., Freeboard, Ice Area)


# In[5]:


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

# In[6]:


# --- Run Settings:
#PATCHIFY_TO_USE = "rows"   # Change this to use other patching techniques

TRIAL_RUN =                    False   # SET THIS TO USE THE PRACTICE SET (MUCH FASTER AND SMALLER, for debugging)
NORMALIZE_ON =                 True    # SET THIS TO USE NORMALIZATION ON FREEBOARD (Results are independent of patchify used)
TRAINING =                     True    # SET THIS TO RUN THE TRAINING LOOP (Use on full dataset for results)
EVALUATING_ON =                False    # SET THIS TO RUN THE METRICS AT THE BOTTOM (Use on full dataset for results)
PLOT_DAY_BY_DAY_METRICS =      False    # See a comparison of metrics per forecast day

# Only run ONCE!!
PLOT_DATA_SPLIT_DISTRIBUTION = False    # Run the data split function to see the train, val, test distribution

# Run Settings (already performed, not needed now - KEEP FALSE!!!)
PLOT_DATA_FULL_DISTRIBUTION = False   # SET THIS TO PLOT THE OUTLIERS (Run ONCE with full set. Results are independent of variables set here)
MAX_FREEBOARD_ON =            False   # To normalize with a pre-defined maximum for outlier handling
MAP_WITH_CARTOPY_ON =         False   # Make sure the Cartopy library is included in the kernel

# --- Time-Related Variables:
CONTEXT_LENGTH = 7            # T: Number of historical time steps used for input
#FORECAST_HORIZON = 3          # Number of future time steps to predict (ex. 1 day for next time step)

# --- Model Hyperparameters:
D_MODEL = 128                 # d_model: Dimension of the transformer's internal representations (embedding dimension)
N_HEAD = 8                    # nhead: Number of attention heads
NUM_TRANSFORMER_LAYERS = 4    # num_layers: Number of TransformerEncoderLayers
BATCH_SIZE = 16               # 16 for context/forecast of 7 and 3; lower for longer range
NUM_EPOCHS = 10

# --- Performance-Related Variables:
NUM_WORKERS = 64   # 64 worked fast for the 7 day forecast
PREFETCH_FACTOR = 4 # 4 worked fast for the 7 day forecast

# --- Feature-Related Variables:
MAX_FREEBOARD_FOR_NORMALIZATION = 1    # Only works when you set MAX_FREEBOARD_ON too; bad results

# --- Space-Related Variables:
LATITUDE_THRESHOLD = 40          # Determines number of cells and patches (could use -90 to use the entire dataset).
CELLS_PER_PATCH = 256            # L: Number of cells within each patch (based on ViT paper 16 x 16 = 256)

# SLURM - CONVERT THIS TO A .PY FILE FIRST
PATCHIFY_TO_USE = os.environ.get("SLURM_PATCHIFY_TO_USE", "rows") # for SLURM
FORECAST_HORIZON = os.environ.get("SLURM_FORECAST_HORIZON", 7) # for SLURM


# ## Other Variables Dependent on Those Above ^

# In[7]:


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


# In[8]:


if TRIAL_RUN:
    model_mode = "tr" # Training Dataset
else:
    model_mode = "fd" # Full Dataset

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
                   
# Model nome convention - fd:full data, etc.
model_version = (
    f"{model_mode}_{norm}_D{D_MODEL}_B{BATCH_SIZE}_lt{LATITUDE_THRESHOLD}_P{NUM_PATCHES}_L{CELLS_PER_PATCH}"
    f"_T{CONTEXT_LENGTH}_Fh{FORECAST_HORIZON}_e{NUM_EPOCHS}_{patching_strategy_abbr}"
)

# Place to save and load the data
PROCESSED_DATA_DIR = (
    f"{model_mode}_{norm}_preprocessed_data.zarr"
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

# In[9]:


import sys
print('System Version:', sys.version)


# In[10]:


#print(sys.executable) # for troubleshooting kernel issues
#print(sys.path)


# In[11]:


import os
#print(os.getcwd())


# In[12]:


import pandas as pd
print('Pandas version', pd.__version__)


# In[13]:


import matplotlib
import matplotlib.pyplot as plt
print('Matplotlib version', matplotlib.__version__)


# In[14]:


import torch
from torch.utils.data import Dataset, DataLoader

print('PyTorch version', torch.__version__)


# In[15]:


import seaborn as sns
print('Seaborn version', sns.__version__)


# # Hardware Details

# In[16]:


if TRAINING and not torch.cuda.is_available():
    raise ValueError("There is a problem with Torch not recognizing the GPUs")
else:
    print(torch.cuda.device_count()) # check the number of available CUDA devices
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

class DailyNetCDFDataset(Dataset):
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
        The number of days to fetch for input in the prediction step (needed for __len__)
    forecast_horizon
        The number of days to predict in the future (needed for __len__)
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
        processed_data_path: str = "./",

        # For len function
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
        zarr_filename = f"{model_mode}_{norm}_preprocessed_data.zarr"
        full_zarr_path = os.path.join(self.processed_data_path, zarr_filename)

        if not os.path.exists(full_zarr_path):
            raise FileNotFoundError(f"Pre-processed Zarr file not found at {full_zarr_path}. Please run `preprocess_data_variables.py` first.")

        try:
            # Check on what's in the zarr
            #z = zarr.open(full_zarr_path, mode='r')
            #print(z.tree())
            '''
            Dimensions:         (nCells_full: 465044, time: 63875, nCells_masked: 53973)
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
            processed_ds = xr.open_zarr(full_zarr_path)
            print(processed_ds)
            logging.info(processed_ds)

            # Times
            print(f"processed_ds['times'].values.shape: {processed_ds['times'].values.shape}  # Expected: (63875,)")
            logging.info(f"times.shape = {processed_ds['times'].values.shape}")
            self.times = processed_ds["times"].values
            
            # Ice Area
            print(f"processed_ds['ice_area'].load().values.shape: {processed_ds['ice_area'].load().values.shape}  # Expected: (63875, 53973)")
            logging.info(f"ice_area.shape = {processed_ds['ice_area'].load().values.shape}")
            self.ice_area = processed_ds["ice_area"].load().values
            
            # Freeboard
            print(f"processed_ds['freeboard'].load().values.shape: {processed_ds['freeboard'].load().values.shape}  # Expected: (63875, 53973)")
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

        # Convert full-domain patch indices to masked-domain indices
        # This ensures there's no out of bounds problem,
        # like index 296237 is out of bounds for axis 1 with size 53973
        self.indices_per_patch_id = [
            [self.full_to_masked[i] for i in patch if i in self.full_to_masked]
            for patch in self.indices_per_patch_id
        ]
        
        logging.info(f"Patchifying completed in {time.perf_counter() - patchify_start_time:.2f} seconds.")

        # Stats on how many dates there are
        logging.info(f"Total days collected: {len(self.times)}")
        logging.info(f"Unique days: {len(np.unique(self.times))}")
        logging.info(f"First 35 days: {self.times[:35]}")
        logging.info(f"Last 35 days: {self.times[-35:]}")

        logging.info(f"Shape of combined ice_area array: {self.ice_area.shape}")
        logging.info(f"Shape of combined freeboard array: {self.freeboard.shape}")

        logging.info(f"Elapsed time for DailyNetCDFDataset __init__: {time.perf_counter() - start_time} seconds")
        print(f"Elapsed time for DailyNetCDFDataset __init__: {time.perf_counter() - start_time} seconds")

    def __len__(self):
        """
        Returns the total number of possible starting indices (idx) for a valid sequence.
        A valid sequence needs `self.context_length` days for input and `self.forecast_horizon` days for target.
        
        ex) If the total number of days is 365, the context_length is 7 and the forecast_horizon is 3, then
        
        365 - (7 + 3) + 1 = 365 - 10 + 1 = 356 valid starting indices
        return len(self.freeboard) - (self.context_length + self.forecast_horizon)
        """
        required_length = self.context_length + self.forecast_horizon

        # Error check
        if len(self.freeboard) < required_length:
            return 0 # Not enough raw data to form even one sample

        # The number of valid starting indices
        return len(self.freeboard) - required_length + 1

    def get_patch_tensor(self, day_idx: int) -> torch.Tensor:
        
        """
        Retrieves the feature data for a specific day, organized into patches.

        This method extracts 'freeboard' and 'ice_area' data for a given day
        and then reshapes it according to the pre-defined patches. Each patch
        will contain its own set of feature values.

        Parameters
        ----------
        day_idx : int
            The integer index of the day to retrieve data for, relative to the
            concatenated dataset's time dimension.

        Returns
        -------
        torch.Tensor
            A tensor containing the feature data organized by patches for the
            specified day.
            Shape: (context_length, num_patches, num_features, patch_size)
            Where:
            - num_patches: Total number of patches (ex., 140).
            - num_features: The number of features per cell (currently 2: freeboard, ice_area).
            - patch_size: The number of cells within each patch.
            
        """

        freeboard_day = self.freeboard[day_idx]  # (nCells,)
        ice_area_day = self.ice_area[day_idx]    # (nCells,)
        features = np.stack([freeboard_day, ice_area_day], axis=0)  # (2, nCells)
        patch_tensors = []

        for patch_indices in self.indices_per_patch_id:
            patch = features[:, patch_indices]  # (2, patch_size)
            patch_tensors.append(torch.tensor(patch, dtype=torch.float32))

        return torch.stack(patch_tensors)  # (context_length, num_patches, num_features, patch_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.datetime64]:

        """__ getitem __ needs to 
        
        1. Given an input of a certain date id, get the input and the target tensors
        2. Return all the patches for the input and the target
           Features are: [freeboard, ice_area] over masked cells. 
           
        """
        # Start with the id of the day in question
        start_idx = idx

        # end_idx is the exclusive end of the input sequence,
        # and the inclusive start of the target sequence.
        end_idx = idx + self.context_length    # ex. start day is 0, context length is 3, end is 3, exclusive
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
        for day in target_seq:
            patch_day = [
                torch.tensor(day[patch_indices]) for patch_indices in self.indices_per_patch_id
            ]
            
            # After stacking, patch_day_tensor will be (num_patches, CELLS_PER_PATCH)
            patch_day_tensor = torch.stack(patch_day)  # (num_patches,)
            target_patches.append(patch_day_tensor)

        # Final target tensor shape: (forecast_horizon, num_patches, CELLS_PER_PATCH)
        target_tensor = torch.stack(target_patches)  # (forecast_horizon, num_patches)
        
        return input_tensor, target_tensor, start_idx, end_idx, target_start, target_end

    def __repr__(self):
        """ Format the string representation of the data """
        return (
            f"<DailyNetCDFDataset: {len(self)} viable days (only includes up to the last possible input date), "
            f"\n{len(self.freeboard[0])} cells/day"
            f"\n{self.num_raw_files} files loaded "
            f"\n{len(self.ice_area)} ice_area length"
            f"\n{len(self.freeboard)} freeboard length"
            f"\nPatchify Algorithm: {self.algorithm}" # What patchify algorithm was used
            f"\n{self.ice_area[-1].shape} shape of ice_area # Should be (63875, 53973) for latitude_threshold of 40"
            f"\n{self.freeboard[-1].shape} shape of freeboard # Should be (63875, 53973) for latitude_threshold of 40"
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
        # A sample requires a total window of `context_length + forecast_horizon` days.
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
dataset = DailyNetCDFDataset()

# Patch locations for positional embedding
PATCH_LATLONS_TENSOR = torch.tensor(dataset.patch_latlons, dtype=torch.float32)

print("========== SPLITTING THE DATASET ===================")
# DIFFERENT SUBSET OPTIONS FOR TRAINING / VALIDATION / TESTING for the trial data vs. full dataset
if TRIAL_RUN:
    total_days = len(dataset)
    train_end = int(total_days * 0.7)
    val_end = int(total_days * 0.85)
    
    train_set = Subset(dataset, range(0, train_end))
    val_set   = Subset(dataset, range(train_end, val_end))
    test_set  = Subset(dataset, range(val_end, total_days))
    
else:

    # --- Splitting by Valid Years (considering the context_length and forecast_horizon) ---

    # Convert dataset.times to pandas DatetimeIndex for easier year-based filtering
    all_times_pd = pd.to_datetime(dataset.times)

    # Define the start and end years for each set - keep this for the full dataset
    train_start_year = 1850
    train_end_year = 2011   
    val_start_year = 2012
    val_end_year = 2017
    test_start_year = 2018
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

total_days = len(train_set) + len(val_set) + len(test_set)
print("Total days = ", total_days)

print("Number of training batches", len(train_set)//BATCH_SIZE)
print("Number of training batches", len(val_set)//BATCH_SIZE)

print("Number of test batches after drop_last incomplete batch", len(test_set)//BATCH_SIZE)
print("Number of test days to drop after drop_last incomplete batch", len(test_set)//BATCH_SIZE)

print("===== Printing Dataset ===== ")
print(dataset)                 # calls __repr__ → see how many files & days loaded

print("===== Sample at dataset[0] ===== ")
input_tensor, target_tensor, start_idx, end_idx, target_start, target_end = dataset[0]

print(f"Fetched start index {start_idx}: Time={dataset.times[start_idx]}")
print(f"Fetched end   index {end_idx}: Time={dataset.times[end_idx]}")

print(f"Fetched target start index {target_start}: Time={dataset.times[target_start]}")
print(f"Fetched target end   index {target_end}: Time={dataset.times[target_end]}")

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

print("input_tensor should be of shape (context_length, num_patches, num_features, patch_size)")
print(f"actual input_tensor.shape = {input_tensor.shape}")
print("target_tensor should be of shape (forecast_horizon, num_patches, patch_size)")
print(f"actual target_tensor.shape = {target_tensor.shape}")


# # Transformer Class

# In[22]:


import torch
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        The number of historical days (time steps) to use as input for the transformer.
    forecast_horizon : int, optional
        The number of future days to predict for each patch.
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
                 activation: str = "gelu" 
                 ):

        super().__init__()

        """
        The transformer should
        1. Accept a sequence of days (ex. 7 days of patches). 
           The context_length parameter says how many days to use for input.
        2. Encode each patch with the transformer.
        3. Output the patches for regression (ex. predict the 8th day).
           The forecast_horizon parameter says how many days to use for the output prediction.
        
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

    def forward(self, x):
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

# In[23]:


if TRAINING:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch import Tensor
    import torch.nn.functional as F
    
    import logging
    
    # Set level to logging.INFO to see the statements
    logging.basicConfig(filename='IceForecastTransformerInstance.log', filemode='w', level=logging.INFO)
    
    model = IceForecastTransformer().to(device)
    
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
    
        for batch_idx, (input_tensor, target_tensor, start_idx, end_idx, target_start, target_end) in enumerate(train_loader):  
    
            # Move input and target to the device
            # x: (B, context_length, num_patches, input_patch_features_dim), y: (B, forecast_horizon, num_patches)
            x = input_tensor.to(device)  # Shape: (B, T, P, C, L)
            y = target_tensor.to(device)  # Shape: (B, forecast_horizon, P, L)
    
            # Reshape x for transformer input
            B, T, P, C, L = x.shape
            x_reshaped_for_transformer_D = x.view(B, T, P, C * L)
    
            # Run through transformer
            y_pred = model(x_reshaped_for_transformer_D) # y_pred is (B, forecast_horizon, num_patches) ex., (16, 1, 140)
            
            # Compute loss
            loss = criterion(y_pred, y) # DIRECTLY compare y_pred and y
        
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
            for batch in val_loader:
                # Unpack the full tuple
                x_val, y_val, start_idx, end_idx, target_start, target_end = batch
        
                # Move to GPU if available
                x_val = x_val.to(device)
                y_val = y_val.to(device)
    
                # Extract dimensions from x_val for reshaping
                # x_val before reshaping: (B_val, T_val, P_val, C_val, L_val)
                B_val, T_val, P_val, C_val, L_val = x_val.shape
                
                # Reshape x_val for transformer input
                x_val_reshaped_for_transformer_input = x_val.view(B_val, T_val, P_val, C_val * L_val)
    
                # Model output is (B, forecast_horizon, P, L)
                y_val_pred = model(x_val_reshaped_for_transformer_input) 
    
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

# In[24]:


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

# In[25]:


from NC_FILE_PROCESSING.metrics_and_plots import *

if EVALUATING_ON:
    
    import torch
    import torch.nn as nn
    
    if not torch.cuda.is_available():
        raise ValueError("There is a problem with Torch not recognizing the GPUs")
    
    # Instantiate the model (must have the same architecture as when it was saved)
    # Create an identical instance of the original __init__ parameters
    loaded_model = IceForecastTransformer()
    
    # Load the saved state_dict (weights_only=True helps ensure safety of pickle files)
    loaded_model.load_state_dict(torch.load(PATH, weights_only=True))
    
    # Set the model to evaluation mode
    loaded_model.eval()
    
    # Move the model to the appropriate device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    
    print("Model loaded successfully!")


# In[26]:


# --- Conditional Data collection for Training and Validation Sets ---
# SEE IF THIS CAN BE MORE EFFICIENT - HANGS FOR FULL DATASET
if PLOT_DATA_SPLIT_DISTRIBUTION:
    print("--- Collecting ground truth data from training and validation sets ---")
    start_time_collect_data = time.perf_counter()
    all_train_actual_values = []
    all_val_actual_values = []
    all_test_actual_values = []

    for i, (sample_x, sample_y, *_) in enumerate(train_loader):
        all_train_actual_values.append(sample_y.cpu().numpy().flatten())
    
    for i, (sample_x, sample_y, *_) in enumerate(val_loader):
        all_val_actual_values.append(sample_y.cpu().numpy().flatten())

    for i, (sample_x, sample_y, *_) in enumerate(test_loader):
        all_test_actual_values.append(sample_y.cpu().numpy().flatten())

    final_train_values = np.concatenate(all_train_actual_values)
    final_val_values = np.concatenate(all_val_actual_values)
    final_test_values = np.concatenate(all_test_actual_values)

    print("--- Finished collecting ground truth data from training and validation sets --")
    print(f"Elapsed time for plotting data distribution comparison: {time.perf_counter() - start_time_collect_data:.2f} seconds")

    
else:
    # Define empty arrays if not plotting distribution to avoid NameError later
    final_train_values = np.array([])
    final_val_values = np.array([])

# 9. Conditional Plotting All Three SIC Distributions (Train, Val, Test Sets)
# This block plots the distribution of SIC values across training, validation, and test sets.
if PLOT_DATA_SPLIT_DISTRIBUTION:
    logging.info("\nPlotting data distributions (Train, Val, Test)...")
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
    # This calculates the Jensen-Shannon Distance between the distributions of the data splits.
    print("\n--- Pairwise Jensen-Shannon Distances for Data Splits ---")
    start_time_jsd_pairwise = time.perf_counter()
    distributions_for_jsd = {
        'train': final_train_values,
        'validation': final_val_values,
        'test': final_test_values
    }
    
    jsd_bins = np.linspace(0, 1, 10 + 1) # 10 bins for JSD calculation
    pairwise_jsd_results = jensen_shannon_distance_pairwise(distributions_for_jsd, jsd_bins)
    
    for pair, jsd_val in pairwise_jsd_results.items():
        print(f"JSD ({pair}): {jsd_val:.4f}")
    end_time_jsd_pairwise = time.perf_counter()
    print(f"Elapsed time for Jensen Shannon Pairwise Calculation: {end_time_jsd_pairwise - start_time_jsd_pairwise:.2f} seconds")


# # Metrics

# In[ ]:


# Updated EVALUATING_ON section
if EVALUATING_ON:
    import io
    import time
    import xarray as xr
    import zarr
    import pandas as pd
    from NC_FILE_PROCESSING.metrics_and_plots import * 
    
    start_full_evaluation = time.perf_counter()
    
    nCells_array = np.load('cellAreas.npy')
    
    # Create a string buffer to capture output
    captured_output = io.StringIO()
    
    # Redirect stdout to the buffer
    sys.stdout = captured_output
    
    print("\nStarting evaluation and metric calculation...")
    logging.info("\nStarting evaluation and metric calculation...")
    print("==================")
    print(f"DEBUG: Batch Size: {BATCH_SIZE} Days")
    print(f"DEBUG: Context Length: {CONTEXT_LENGTH} Days")
    print(f"DEBUG: Forecast Horizon: {FORECAST_HORIZON} Days")
    print(f"DEBUG: Number of batches in test_loader (with drop_last=True): {len(test_loader)} Batches")
    print("==================")
    print(f"DEBUG: len(test_set): {len(dataset) - val_end} Days (approx, as test_set is not directly defined here)")
    print(f"DEBUG: len(dataset) for splitting: {len(dataset)} Days")
    print(f"DEBUG: train_end: {train_end} (must end before the end of the context window + forecast horizon)")
    print(f"DEBUG: val_end: {val_end} (must end before the end of the context window + forecast horizon)")
    print(f"DEBUG: range for test_set: {range(val_end, total_days)}")
    print("==================")
    
    print("\n--- Running Test Set Evaluation for Temporal Degradation ---")
    start_time_test_eval = time.perf_counter()

    # --- Data Accumulators for Xarray ---
    # Accumulate data as lists of numpy arrays
    # Initialize accumulators
    all_predicted_sic_data = []
    all_actual_sic_data = []
    all_dates = []
    all_forecast_steps = []
    all_predicted_sie_data = []
    all_actual_sie_data = []

    # Pre-compute the mapping dictionary. This is still a good idea.
    masked_to_full_dict = dataset.masked_to_full
    indices_per_patch_id_list = dataset.indices_per_patch_id
    
    print("\n--- Running Test Set Evaluation for Temporal Degradation ---")
    
    for i, (sample_x, sample_y, start_idx, end_idx, target_start_idx, target_end_idx) in enumerate(test_loader):
        # 1. Get model predictions for the whole batch
        sample_x = sample_x.to(device)
        
        # Reshape for model input
        B_sample, T_context, P_sample, C_sample, L_sample = sample_x.shape
        sample_x_reshaped = sample_x.view(B_sample, T_context, P_sample, C_sample * L_sample)
        
        with torch.no_grad():
            predicted_y_patches = loaded_model(sample_x_reshaped)
        
        # 2. Move data to CPU and convert to NumPy arrays once for the whole batch
        predicted_patches_np = predicted_y_patches.detach().cpu().numpy()
        actual_patches_np = sample_y.cpu().numpy()
        
        # 3. Process dates and forecast steps in a vectorized way (this part is fine)
        batch_size, forecast_horizon = predicted_patches_np.shape[0], predicted_patches_np.shape[1]
        
        target_start_dates = dataset.times[target_start_idx]
        target_start_dates_np = pd.to_datetime(target_start_dates).to_numpy()
        forecast_steps = np.arange(1, forecast_horizon + 1)
        
        dates_for_batch = target_start_dates_np[:, np.newaxis] + pd.to_timedelta(forecast_steps - 1, unit='D').to_numpy()
        
        all_dates.extend(dates_for_batch.flatten().tolist())
        all_forecast_steps.extend(np.tile(forecast_steps, batch_size).tolist())
        
        # --- reconstruction part ---
        
        # Get the number of patches for this specific batch
        # This is `P_sample` from the reshaped input `sample_x`
        
        # We assume the patches in the batch are from a contiguous block of patch IDs
        # This is a safe assumption for the DataLoader's `__getitem__`
        batch_patch_indices_list = indices_per_patch_id_list[:P_sample]
        batch_patch_indices_flat = np.concatenate(batch_patch_indices_list)
        batch_full_indices_flat = np.array([masked_to_full_dict[idx] for idx in batch_patch_indices_flat])
        
        # Initialize temp arrays for the full grid for the entire batch
        total_steps_in_batch = batch_size * forecast_horizon
        total_grid_cells = len(dataset.latCell)
        temp_predicted_grids = np.full((total_steps_in_batch, total_grid_cells), np.nan)
        temp_actual_grids = np.full((total_steps_in_batch, total_grid_cells), np.nan)
        
        # Now, loop through each step in the batch, but use vectorized assignment inside
        for step_idx in range(total_steps_in_batch):
            # Flatten the patch data for this specific step
            predicted_step_flat = predicted_patches_np.reshape(-1, P_sample, C_sample)[step_idx].flatten()
            actual_step_flat = actual_patches_np.reshape(-1, P_sample, C_sample)[step_idx].flatten()
            
            # This single line replaces your slow nested Python loops!
            temp_predicted_grids[step_idx, batch_full_indices_flat] = predicted_step_flat
            temp_actual_grids[step_idx, batch_full_indices_flat] = actual_step_flat
            
        # 4. Calculate SIE for the entire batch in a vectorized way
        batch_predicted_sie = calculate_full_sie_in_kilometers(temp_predicted_grids, nCells_array)
        batch_actual_sie = calculate_full_sie_in_kilometers(temp_actual_grids, nCells_array)
        
        # 5. Append the large batch arrays to the lists
        all_predicted_sic_data.append(temp_predicted_grids)
        all_actual_sic_data.append(temp_actual_grids)
        all_predicted_sie_data.extend(batch_predicted_sie)
        all_actual_sie_data.extend(batch_actual_sie)

    print(len(all_sic_dates))
    print(len(all_sie_dates))
    
    # --- Create and Save Xarray Datasets to Zarr ---
    print("\nCreating and saving Zarr stores...")
    logging.info("\nCreating and saving Zarr stores...")
    
    # Zarr path for SIC degradation data
    sic_zarr_path = f'{model_version}_sic.zarr'
    
    if all_sic_dates:
        # We need to reshape the SIC data from a list of 1D arrays to a single 2D array
        # Let's assume all_predicted_sic_data has shape (num_time_steps, num_cells)
        predicted_sic_stack = np.stack(all_predicted_sic_data)
        actual_sic_stack = np.stack(all_actual_sic_data)

        # Create the xarray Dataset for SIC
        sic_ds = xr.Dataset(
            {
                'predicted': (('date_forecast', 'cell'), predicted_sic_stack),
                'actual': (('date_forecast', 'cell'), actual_sic_stack),
            },
            coords={
                'date': (('date_forecast'), all_sic_dates),
                'forecast_step': (('date_forecast'), all_sic_forecast_steps),
                'cell': (('cell'), np.arange(predicted_sic_stack.shape[1])),
            }
        )
        
        # Save to Zarr
        sic_ds.to_zarr(sic_zarr_path, mode='w', compute=True)
        print(f"Saved SIC performance degradation data to {sic_zarr_path}")
        logging.info(f"Saved SIC performance degradation data to {sic_zarr_path}")
        
    # Zarr path for SIE degradation data
    sie_zarr_path = f'{model_version}_sie.zarr'
    
    if all_sie_dates:
        # SIE data is simpler, already 1D lists
        sie_ds = xr.Dataset(
            {
                'predicted_sie_km': (('date_forecast'), np.array(all_predicted_sie_data)),
                'actual_sie_km': (('date_forecast'), np.array(all_actual_sie_data)),
            },
            coords={
                'date': (('date_forecast'), all_sie_dates),
                'forecast_step': (('date_forecast'), all_sie_forecast_steps),
            }
        )
        
        # Save to Zarr
        sie_ds.to_zarr(sie_zarr_path, mode='w', compute=True)
        print(f"Saved SIE performance degradation data to {sie_zarr_path}")
        logging.info(f"Saved SIE performance degradation data to {sie_zarr_path}")

    end_full_evaluation = time.perf_counter()
    total_eval_time = end_full_evaluation - start_full_evaluation
    print(f"\nTotal evaluation and metric calculation took: {total_eval_time:.2f} seconds.")
    logging.info(f"\nTotal evaluation and metric calculation took: {total_eval_time:.2f} seconds.")
    
    end_time_test_eval = time.perf_counter()
    print(f"Elapsed time for saving SIC and SIE performance degradation csvs: {end_time_test_eval - start_time_test_eval:.2f} seconds")


# In[ ]:


if EVALUATING_ON:
    start_full_evaluation = time.perf_counter()
    # Create a string buffer to capture output
    captured_output = io.StringIO()
    
    # Redirect stdout to the buffer
    sys.stdout = captured_output

    # --- CALL METRIC FUNCTIONS HERE ---
    
    logging.info("\nCalling SIC plotting functions...")
    # 1. Calculate and Log Overall Spatial Errors (Overall SIC's MAE, MSE, RMSE)
    # This function calculates and prints overall spatial error metrics.
    logging.info("\nErrors Overall...")
    start_time_spatial_errors = time.perf_counter()
    if not sic_ds.empty:
        calculate_and_log_spatial_errors(sic_ds, title_suffix=" (Overall)")
    end_time_spatial_errors = time.perf_counter()
    print(f"Elapsed time for Overall Spatial Error Calculation: {end_time_spatial_errors - start_time_spatial_errors:.2f} seconds")

    # 2. Plot Temporal Degradation (SIC Over Each Forecast Day)
    # This function plots MAE and RMSE degradation over the forecast horizon for SIC.
    logging.info("\nTemporal Degradation (SIC) over forecast days ...")
    start_time_sic_temporal_degradation = time.perf_counter()
    if not sic_ds.empty:
        plot_SIC_temporal_degradation(sic_ds, model_version, patching_strategy_abbr)
    end_time_sic_temporal_degradation = time.perf_counter()
    print(f"Elapsed time for SIC Temporal Degradation Plot: {end_time_sic_temporal_degradation - start_time_sic_temporal_degradation:.2f} seconds")

    # 3. Plot Actual vs. Predicted SIC Distribution (Overall SIC)
    # This function plots overlapping histograms of actual vs. predicted SIC values.
    logging.info("\nDistance between actual and predicted ...")
    start_time_actual_vs_predicted_sic_dist = time.perf_counter()
    if not sic_ds.empty:
        plot_actual_vs_predicted_sic_distribution(sic_ds, model_version, patching_strategy_abbr, num_bins=50, title_suffix=" (Overall)")
    end_time_actual_vs_predicted_sic_dist = time.perf_counter()
    print(f"Elapsed time for Overall Actual vs Predicted SIC histogram: {end_time_actual_vs_predicted_sic_dist - start_time_actual_vs_predicted_sic_dist:.2f} seconds")

    logging.info("\nCalling SIE plotting functions...")
    # 4. Log Classification Report (SIE as a binary value of SIC with 15% threshold)
    # This function provides a classification report for Sea Ice Extent (SIE).
    logging.info("\nClassification Report for SIE ...")
    start_time_classification_report = time.perf_counter()
    if not sic_ds.empty:
        sie_threshold = 0.15 # Define sie_threshold
        log_classification_report(sic_ds, threshold=sie_threshold)
    end_time_classification_report = time.perf_counter()
    print(f"Elapsed time for Overall Classification Report: {end_time_classification_report - start_time_classification_report:.2f} seconds")
    
    # 5. Plot Overall SIE Confusion Matrix (SIE here is derived as a binary value of SIC with 15% threshold)
    # This function generates a confusion matrix plot for SIE classification.
    logging.info("\nConfusion Matrix for SIE ...")
    start_time_confusion_matrix = time.perf_counter()
    if not sic_ds.empty:
        plot_sie_confusion_matrix(sic_ds, threshold=sie_threshold, model_version=model_version, patching_strategy_abbr=patching_strategy_abbr, forecast_day=None) # None for overall
    end_time_confusion_matrix = time.perf_counter()
    print(f"Elapsed time for Overall Confusion Matrix Plot: {end_time_confusion_matrix - start_time_confusion_matrix:.2f} seconds")
    
    # 6. Plot Overall ROC Curve (SIE here is derived as a binary value of SIC with 15% threshold)
    # This function plots the Receiver Operating Characteristic (ROC) curve and calculates AUC.
    logging.info("\nROC Curve for SIE ...")
    start_time_roc_curve = time.perf_counter()
    if not sic_ds.empty:
        plot_roc_curve(sic_ds, model_version=model_version, patching_strategy_abbr=patching_strategy_abbr, threshold=sie_threshold, forecast_day=None) # None for overall
    end_time_roc_curve = time.perf_counter()
    print(f"Elapsed time for Overall ROC Curve Plot: {end_time_roc_curve - start_time_roc_curve:.2f} seconds")

    # 7. Plot F1-Score Degradation (SIE here is derived as a binary value of SIC with 15% threshold)
    # This function plots the F1-score degradation for SIE classification.
    logging.info("\nF1 Score for SIE ...")
    start_time_f1_degradation = time.perf_counter()
    if not sic_ds.empty: # F1-score uses the same SIC temporal data
        plot_SIE_f1_score_degradation(sic_ds, model_version, patching_strategy_abbr, threshold=sie_threshold)
    end_time_f1_degradation = time.perf_counter()
    print(f"Elapsed time for F1-Score Degradation Plot: {end_time_f1_degradation - start_time_f1_degradation:.2f} seconds")

    # 8. Plot SIE Degradation (SIE as the area that is ice in km^2)
    # This function plots MAE and RMSE degradation over the forecast horizon for SIE in km^2.
    logging.info("\nDegradation of SIE (as square kilometers) over the forecast window ...")
    start_time_sie_kilometers_degradation = time.perf_counter()
    if not sie_ds.empty:
        plot_SIE_Kilometers_degradation(sie_ds, model_version, patching_strategy_abbr)
    end_time_sie_kilometers_degradation = time.perf_counter()
    print(f"Elapsed time for SIE Kilometers Degradation Plot: {end_time_sie_kilometers_degradation - start_time_sie_kilometers_degradation:.2f} seconds")

        if PLOT_DAY_BY_DAY_METRICS:
        
        logging.info("\nPlotting per-day forecast analysis...")
        # --- Optional: Per-Day Analysis for Classification Metrics and Distributions ---
        # This section iterates through each forecast day to provide detailed metrics and plots.
        print("\n############################################")
        print("\n#   PER-DAY FORECAST ANALYSIS (Optional)   #")
        print("\n############################################")
        start_time_per_day_analysis = time.perf_counter()
        for day in range(1, FORECAST_HORIZON + 1): # Loop through each forecast day
            df_day = sic_ds[sic_ds['forecast_step'] == day]
            if not df_day.empty:
                print(f"\n--- Metrics for Forecast Day {day} ---")
                
                # Log Classification Report for specific day
                log_classification_report(df_day['actual'].values, df_day['predicted'].values, threshold=sie_threshold)
                
                # Plot SIC distribution for specific day
                plot_actual_vs_predicted_sic_distribution(
                    df_day['actual'].values, df_day['predicted'].values, model_version, patching_strategy_abbr, num_bins=50, title_suffix=f" (Day {day})"
                )
    
                # Plot Confusion Matrix for specific day
                plot_sie_confusion_matrix(sic_ds, sie_threshold, model_version, patching_strategy_abbr, forecast_day=day)
                
                # Plot ROC Curve for specific day
                plot_roc_curve(sic_ds, model_version, patching_strategy_abbr, sie_threshold, forecast_day=day)
                
            end_time_per_day_analysis = time.perf_counter()
        print(f"Elapsed time for Per-Day Forecast Analysis: {end_time_per_day_analysis - start_time_per_day_analysis:.2f} seconds")

    # END OF EVALUATION
    end_full_evaluation = time.perf_counter()
    print(f"Elapsed time for FULL EVALUATION: {end_full_evaluation - start_full_evaluation:.2f} seconds")
    
    print("\nEvaluation complete.")

    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Now, write the captured output to the file
    with open(f'{model_version}_Metrics.txt', 'w') as f:
        f.write(captured_output.getvalue())
    
    print(f"Metrics saved as {model_version}_Metrics.txt")


# # Make a Single Prediction

# In[ ]:


if EVALUATING_ON:
    if MAP_WITH_CARTOPY_ON:
        # Load one batch
        data_iter = iter(test_loader)
        sample_x, sample_y, start_idx, end_idx, target_start, target_end = next(data_iter)
        
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
            predicted_y_patches = loaded_model(sample_x_reshaped)
        
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
        
        for day_idx in range(loaded_model.forecast_horizon):
            predicted_day = predicted_y_patches[:, day_idx, :, :].cpu()
            all_predicted_ice_areas.append(predicted_day)
        
            actual_day = sample_y[:, day_idx, :, :].cpu()
            all_actual_ice_areas.append(actual_day)
        
            print(f"Processing forecast day {day_idx}: Predicted shape {predicted_day.shape}, Actual shape {actual_day.shape}")
        
            # Save each day's prediction/actual data if needed
            # np.save(f'patches/ice_area_patches_predicted_day{day_idx}.npy', predicted_day)
            # np.save(f'patches/ice_area_patches_actual_day{day_idx}.npy', actual_day)


# # Recover nCells from Patches for Visualization

# In[ ]:


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

