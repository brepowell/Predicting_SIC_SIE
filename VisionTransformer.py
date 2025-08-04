#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
print('Xarray version', xr.__version__)


# In[2]:


import numpy as np
print('Numpy version', np.__version__)


# In[3]:


from perlmutterpath import *  # Contains the data_dir and mesh_dir variables
NUM_FEATURES = 2              # C: Number of features per cell (ex., Freeboard, Ice Area)


# In[4]:


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

# In[5]:


# --- Run Settings:
#PATCHIFY_TO_USE = "latlon_spillover"   # Change this to use other patching techniques
PATCHIFY_TO_USE = os.environ.get("SLURM_PATCHIFY_TO_USE", "rows") # for SLURM

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
FORECAST_HORIZON = 7          # Number of future time steps to predict (ex. 1 day for next time step)

# --- Model Hyperparameters:
D_MODEL = 128                 # d_model: Dimension of the transformer's internal representations (embedding dimension)
N_HEAD = 8                    # nhead: Number of attention heads
NUM_TRANSFORMER_LAYERS = 4    # num_layers: Number of TransformerEncoderLayers
BATCH_SIZE = 8                # 16 for context/forecast of 7 and 3; lower for longer range
NUM_EPOCHS = 10

# --- Performance-Related Variables:
NUM_WORKERS = 8   # changed from 64

# --- Feature-Related Variables:
MAX_FREEBOARD_FOR_NORMALIZATION = 1    # Only works when you set MAX_FREEBOARD_ON too; bad results

# --- Space-Related Variables:
LATITUDE_THRESHOLD = 40          # Determines number of cells and patches (could use -90 to use the entire dataset).
CELLS_PER_PATCH = 256            # L: Number of cells within each patch (based on ViT paper 16 x 16 = 256)


# ## Other Variables Dependent on Those Above ^

# In[6]:


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


# In[7]:


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
    f"{model_version}_Dataset.zarr"
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

# In[8]:


import sys
print('System Version:', sys.version)


# In[9]:


#print(sys.executable) # for troubleshooting kernel issues
#print(sys.path)


# In[10]:


import os
#print(os.getcwd())


# In[11]:


import pandas as pd
print('Pandas version', pd.__version__)


# In[12]:


import matplotlib
import matplotlib.pyplot as plt
print('Matplotlib version', matplotlib.__version__)


# In[13]:


import torch
from torch.utils.data import Dataset, DataLoader

print('PyTorch version', torch.__version__)


# In[14]:


import seaborn as sns
print('Seaborn version', sns.__version__)


# # Hardware Details

# In[15]:


if TRAINING and not torch.cuda.is_available():
    raise ValueError("There is a problem with Torch not recognizing the GPUs")
else:
    print(torch.cuda.device_count()) # check the number of available CUDA devices
    # will print 1 on login node; 4 on GPU exclusive node; 1 on shared GPU node


# In[16]:


#print(torch.cuda.get_device_properties(0)) #provides information about a specific GPU
#total_memory=40326MB, multi_processor_count=108, L2_cache_size=40MB


# In[17]:


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



# # Pre-processing + Freeboard calculation functions

# In[18]:


# Constants (adjust if you use different units)
D_WATER = 1023  # Density of seawater (kg/m^3)
D_ICE = 917     # Density of sea ice (kg/m^3)
D_SNOW = 330    # Density of snow (kg/m^3)

MIN_AREA = 1e-6

def compute_freeboard(area: np.ndarray, 
                      ice_volume: np.ndarray, 
                      snow_volume: np.ndarray) -> np.ndarray:
    """
    Compute sea ice freeboard from ice and snow volume and area.
    
    Parameters
    ----------
    area : np.ndarray
        Sea ice concentration / area (same shape as ice_volume and snow_volume).
    ice_volume : np.ndarray
        Sea ice volume per grid cell.
    snow_volume : np.ndarray
        Snow volume per grid cell.
    
    Returns
    -------
    freeboard : np.ndarray
        Freeboard height for each cell, same shape as inputs.
    """
    # Initialize arrays
    height_ice = np.zeros_like(ice_volume)
    height_snow = np.zeros_like(snow_volume)

    # Valid mask: avoid dividing by very small or zero area
    valid = area > MIN_AREA

    # Safely compute heights where valid
    height_ice[valid] = ice_volume[valid] / area[valid]
    height_snow[valid] = snow_volume[valid] / area[valid]

    # Compute freeboard using the physical formula
    freeboard = (
        height_ice * (D_WATER - D_ICE) / D_WATER +
        height_snow * (D_WATER - D_SNOW) / D_WATER
    )

    return freeboard


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
    data_dir : str
        Directory containing NetCDF files
    transform : Callable | None
        Optional - transform applied to the data tensor *only*.
    latitude_threshold
        The minimum latitude to use for Arctic data
    context_length
        The number of days to fetch for input in the prediction step
    forecast_horizon
        The number of days to predict in the future
    plot_outliers_and_imbalance
        Optional - check outliers and imbalance on the variables Ice Area and Freeboard
    trial_run
        Optional - use the data in the trial directory instead of the full dataset
        Specify the name of the trial director in perlmutterpath.py
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
        data_dir: str = data_dir,
        mesh_dir: str = mesh_dir,
        transform: Callable = None,
        latitude_threshold: int = LATITUDE_THRESHOLD,
        context_length: int = CONTEXT_LENGTH,
        forecast_horizon: int = FORECAST_HORIZON,
        normalize_on: bool = NORMALIZE_ON,
        plot_outliers_and_imbalance: bool = PLOT_DATA_FULL_DISTRIBUTION, # set FALSE NOW THAT I HAVE IT FOR FULL DATASET
        trial_run: bool = TRIAL_RUN, # Use the trial data directory
        num_patches: int = NUM_PATCHES,
        cells_per_patch: int = CELLS_PER_PATCH,
        patchify_func: Callable = DEFAULT_PATCHIFY_METHOD_FUNC, # Default patchify function
        patchify_func_key: str = PATCHIFY_TO_USE, # Key to look up specific params
        max_freeboard_for_normalization: int = MAX_FREEBOARD_FOR_NORMALIZATION,
        max_freeboard_on: bool = MAX_FREEBOARD_ON,
        processed_data_path: str = PROCESSED_DATA_DIR,       
    
    ):

        """ __init__ needs to 

        Handle the raw data:
        1) Gather the sorted daily data from each netCDF file (1 file = 1 month of daily data)
            The netCDF files contain nCells worth of data per day for each feature (ice area, ice volume, etc.)
            nCells = 465044 with the IcoswISC30E3r5 mesh
        2) Load the mesh and initialize the cell mask
        3) Store a list of datetimes from each file 
        4) Extract raw data
        
        Perform pre-processing:
        5) Apply a mask to nCells to look just at regions in certain latitudes
            nCells >= 40 degrees is 53973 cells
            nCells >= 50 degrees is 35623 cells
        6) Derive Freeboard from ice area, snow volume, and ice volume
        7) Custom patchify and store patch_ids so the data loader can use them
        8) Optional: Plot the outliers and data imbalance for Ice Area and Freeboard
        9) Optional: Normalize the data (Ice area is already between 0 and 1; Freeboard is not) """

        start_time = time.perf_counter()
        self.data_dir = data_dir
        self.transform = transform
        self.latitude_threshold = latitude_threshold
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.normalize_on = normalize_on
        self.plot_outliers_and_imbalance = plot_outliers_and_imbalance
        self.trial_run = trial_run
        self.num_patches = num_patches
        self.cells_per_patch = cells_per_patch
        self.patchify_func = patchify_func # Store the specified patchify function
        self.patchify_func_key = patchify_func_key # Store the key for looking up specific params
        self.max_freeboard_for_normalization = max_freeboard_for_normalization
        self.max_freeboard_on = max_freeboard_on
        self.processed_data_path = processed_data_path # Store the processed data path
        
        # --- Check for pre-processed data first ---
        if self.processed_data_path and os.path.exists(self.processed_data_path):
            logging.info(f"Loading pre-processed data from Zarr store: {self.processed_data_path}")
            try:

                processed_ds = xr.open_zarr(self.processed_data_path)
                self.times = processed_ds["time"].values
                self.ice_area = processed_ds["ice_area"].values
                self.freeboard = processed_ds["freeboard"].values
                self.cell_mask = processed_ds["cell_mask"].values
                
                # Convert string representations back to dicts using eval()
                # Be cautious with eval() if the source of the Zarr store is untrusted.
                self.full_to_masked = eval(processed_ds["full_to_masked"].item())
                self.masked_to_full = eval(processed_ds["masked_to_full"].item())
                self.patch_latlons = processed_ds["patch_latlons"].values
                self.algorithm = processed_ds["algorithm"].item()
                self.num_raw_files = processed_ds["num_raw_files"].item()
                
                # Reconstruct the list of lists (jagged array) from the flattened indices and offsets
                flattened_indices = processed_ds["flattened_indices"].values
                patch_offsets = processed_ds["patch_offsets"].values
                self.indices_per_patch_id = []
                start_idx = 0

                # Convert array of arrays back to list of lists/arrays
                # Ensure it's a list of lists of ints
                for offset in patch_offsets:
                    self.indices_per_patch_id.append(flattened_indices[start_idx:offset])
                    start_idx = offset
                
                logging.info(f"Loaded pre-processed data in {time.time() - start_time:.2f} seconds.")
                print(f"Loaded pre-processed data in {time.time() - start_time:.2f} seconds.")
                return # Exit __init__ if data loaded successfully
            except Exception as e:
                logging.error(f"Error loading pre-processed data from {self.processed_data_path}: {e}")
                print(f"Error loading pre-processed data: {e}. Falling back to raw NetCDF loading.")
                # Fall through to raw NetCDF loading if pre-processed load fails

        logging.info("No valid pre-processed data found. Loading from raw NetCDF files.")

        
        # --- 1. Gather files (sorted for deterministic order) ---------
        if trial_run:
            # Use glob pattern for open_mfdataset
            file_pattern = os.path.join(self.data_dir, "v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.202*.nc")
        else:
            # Use glob pattern for open_mfdataset for the full dataset
            file_pattern = os.path.join(self.data_dir, "v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.*.nc")

        # Check if any files match the pattern using glob
        file_paths = sorted(glob.glob(file_pattern))
        if not file_paths:
            raise FileNotFoundError(f"No *.nc files found matching the pattern in {self.data_dir}")

        self.num_raw_files = len(file_paths) # Store the count of raw files
        logging.info(f"Found {self.num_raw_files} NetCDF files matching the file pattern")
                
        # --- 2. Load the mesh file. Latitudes and Longitudes are in radians. ---
        mesh = xr.open_dataset(mesh_dir)
        self.latCell = np.degrees(mesh["latCell"].values)
        self.lonCell = np.degrees(mesh["lonCell"].values)
        mesh.close()
        
        # Initialize the cell mask
        self.cell_mask = self.latCell >= latitude_threshold        
        masked_ncells_size = np.count_nonzero(self.cell_mask)
        logging.info(f"Mask size: {masked_ncells_size}")

        self.full_to_masked = {
            full_idx: new_idx
            for new_idx, full_idx in enumerate(np.where(self.cell_mask)[0])
        }

        # Also store reverse mapping: masked -> full for recovery of data later
        self.masked_to_full = {
            v: k for k, v in self.full_to_masked.items()
        }

        logging.info(f"=== Extracting raw data and times using xarray.open_mfdataset === ")

        # Use open_mfdataset for efficient parallel loading
        # Specify 'timeDaily_avg_iceAreaCell', 'timeDaily_avg_iceVolumeCell', 'timeDaily_avg_snowVolumeCell'
        # as the variables to load.
        # Use combine='nested' and concat_dim='Time' because the NetCDF files do not have explicit time coordinates,
        # but the 'Time' dimension exists and files are sorted by name.
        # parallel=True enables Dask for parallel file opening and processing.
        with xr.open_mfdataset(
            file_pattern,
            combine='nested',
            concat_dim='Time',
            parallel=True,  # Might cause an error
            data_vars=['timeDaily_avg_iceAreaCell', 'timeDaily_avg_iceVolumeCell', 'timeDaily_avg_snowVolumeCell', 'xtime_startDaily'],
            decode_times=True # Let xarray handle time decoding
        ) as combined_ds:
            # Extract times from byte string format and convert to datetime64[s]
            xtime_byte_array = combined_ds["xtime_startDaily"].values
            xtime_unicode_array = xtime_byte_array.astype(str)
            xtime_cleaned_array = np.char.replace(xtime_unicode_array, "_", " ")
            self.times = np.asarray(xtime_cleaned_array, dtype='datetime64[s]')

            # Extract raw data and apply mask directly to Dask arrays, then compute to NumPy
            # .compute() triggers the actual loading into memory
            # .values gives you NumPy arrays
            self.ice_area = combined_ds["timeDaily_avg_iceAreaCell"][:, self.cell_mask].compute().values
            ice_volume_combined = combined_ds["timeDaily_avg_iceVolumeCell"][:, self.cell_mask].compute().values
            snow_volume_combined = combined_ds["timeDaily_avg_snowVolumeCell"][:, self.cell_mask].compute().values

        # Stats on how many dates there are
        logging.info(f"Total days collected: {len(self.times)}")
        logging.info(f"Unique days: {len(np.unique(self.times))}")
        logging.info(f"First 35 days: {self.times[:35]}")
        logging.info(f"Last 35 days: {self.times[-35:]}")

        logging.info(f"Shape of combined ice_area array: {self.ice_area.shape}")
        logging.info(f"Shape of combined ice_volume_combined array: {ice_volume_combined.shape}")
        logging.info(f"Shape of combined snow_volume_combined array: {snow_volume_combined.shape}")

        logging.info(f"Elapsed time for data file reading: {time.perf_counter() - start_time} seconds")
        
        # --- 6. Derive Freeboard from ice area, snow volume and ice volume
        logging.info(f"=== Calculating Freeboard === ")

        # Convert the xarray objects using .values so the freeboard calculation gets numpy arrays
        self.freeboard = compute_freeboard(self.ice_area, ice_volume_combined, snow_volume_combined)
        logging.info(f"Elapsed time for freeboard calculation: {time.perf_counter() - start_time} seconds")
        
        logging.info(f"=== Patchifying === ")

        # Get the parameters for the patchification function
        patchify_call_params = COMMON_PARAMS.copy()
        
        # Retrieve only the specific parameters for the chosen patchify function
        patchify_call_params.update(SPECIFIC_PARAMS.get(self.patchify_func_key, {}))
        
        # --- 7. Use the dynamic patchify function
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
        logging.info(f"Elapsed time for patchifying with the {self.algorithm} algorithm: {time.perf_counter() - start_time} seconds")

        # --- 8. Optional --- OUTLIER DETECTION AND DATA IMBALANCE CHECK ---
        if self.trial_run:
            status = "trial"
        else:
            status = "prod"
            
        if self.plot_outliers_and_imbalance:
            logging.info(f"=== Plotting Outliers and Imbalance === ")
            check_and_plot_freeboard(self.freeboard, self.times, f"{status}_fb_pre_norm")
            analyze_ice_area_imbalance(self.ice_area)
            plot_ice_area_imbalance(self.ice_area, status)
            logging.info(f"Elapsed time for plotting the outliers and imbalance {time.perf_counter() - start_time} seconds")

        # --- 9. Optional --- Normalize the data (Area is already between 0 and 1; Freeboard is not)
        if self.normalize_on:
            logging.info(f"=== Normalizing Freeboard with Scikit-learn MinMaxScaler === ")
    
            # Reshape freeboard data from (days, cells) to (total_samples, 1) for the scaler
            freeboard_reshaped = self.freeboard.reshape(-1, 1)

            # Determine the min and max values for the scaler based on the flag
            if self.max_freeboard_on:
                logging.info(f"Using custom max freeboard value: {self.max_freeboard_for_normalization}")
                min_val = 0
                max_val = self.max_freeboard_for_normalization
            else:
                logging.info(f"Using min and max values from the data.")
                min_val = freeboard_reshaped.min()
                max_val = freeboard_reshaped.max()

            # Initialize and fit the scaler with the determined range
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit([[min_val], [max_val]]) 

            # Transform the freeboard data and reshape it back to the original shape
            self.freeboard = scaler.transform(freeboard_reshaped).reshape(self.freeboard.shape)
            
            # Apply clipping to ensure all values are within the 0 to 1 range
            self.freeboard = np.clip(self.freeboard, 0, 1)

            logging.info(f"Freeboard min (pre-norm): {min_val} meters" )
            logging.info(f"Freeboard max (pre-norm): {max_val} meters")
            logging.info(f"Freeboard min (post-norm): {self.freeboard.min()}" )
            logging.info(f"Freeboard max (post-norm): {self.freeboard.max()}")

            if self.plot_outliers_and_imbalance:
                logging.info(f"Elapsed time for normalizing the Freeboard: {time.perf_counter() - start_time} seconds")
                check_and_plot_freeboard(self.freeboard, self.times, f"{status}_fb_post_norm")
        
        # --- Save processed data if a path is provided and it's not a trial run ---
        #if self.processed_data_path and not self.trial_run:
        if self.processed_data_path:
            
            logging.info(f"Attempting to save processed data to Zarr store: {self.processed_data_path}")
            try:

                # Flatten the list of lists into a single 1D array
                flattened_indices = np.concatenate(self.indices_per_patch_id)
                
                # Create an array of cumulative sums to mark the start and end of each patch
                patch_offsets = np.cumsum([len(p) for p in self.indices_per_patch_id])
                
                # Create a new xarray dataset
                # Create a new xarray dataset
                processed_ds_to_save = xr.Dataset(
                    {
                        "ice_area": (("time", "nCells_masked"), self.ice_area),
                        "freeboard": (("time", "nCells_masked"), self.freeboard),
                        "cell_mask": ("nCells_full", self.cell_mask),
                        "full_to_masked": ((), str(self.full_to_masked)),
                        "masked_to_full": ((), str(self.masked_to_full)),
                        "algorithm": ((), self.algorithm),
                        "flattened_indices": ("nCells_in_patches", flattened_indices),
                        "patch_offsets": ("patch_idx", patch_offsets),
                        "patch_latlons": (("patch_idx", "latlon_coord"), self.patch_latlons),
                        "num_raw_files": ((), self.num_raw_files),
                    },
                    coords={
                        "time": self.times,
                        "nCells_masked": np.arange(self.ice_area.shape[1]),
                        "nCells_full": np.arange(len(self.latCell)),
                        "patch_idx": np.arange(len(self.indices_per_patch_id)),
                        "latlon_coord": ["latitude", "longitude"]
                    }
                )
                
                # Save to Zarr format
                # Use compute=True to ensure all Dask arrays are written to disk immediately
                processed_ds_to_save.to_zarr(self.processed_data_path, mode='w', compute=True)
                logging.info(f"Processed data successfully saved to {self.processed_data_path}")
                print(f"Processed data successfully saved to {self.processed_data_path}")
                
            except Exception as e:
                logging.error(f"Failed to save processed data to Zarr: {e}")
                print(f"Failed to save processed data to Zarr: {e}")

        logging.info("End of __init__")
        end_time = time.perf_counter()
        logging.info(f"Elapsed time for DailyNetCDFDataset __init__: {end_time - start_time} seconds")
        print(f"Elapsed time for DailyNetCDFDataset __init__: {end_time - start_time} seconds")
        print(f"In minutes:                {(end_time - start_time)//60} minutes")

    def __len__(self):
        """
        Returns the total number of possible starting indices (idx) for a valid sequence.
        A valid sequence needs `self.context_length` days for input and `self.forecast_horizon` days for target.
        
        ex) If the total number of days is 365, the context_length is 7 and the forecast_horizon is 3, then
        
        365 - (7 + 3) + 1 = 365 - 10 + 1 = 356 valid starting indices
        return len(self.freeboard) - (self.context_length + self.forecast_horizon)
        """
        required_length = self.context_length + self.forecast_horizon
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
        end_idx = idx + self.context_length
        target_start = end_idx

        # the target sequence ends after forecast horizon
        target_end = end_idx + self.forecast_horizon

        if target_end > len(self.freeboard):
            raise IndexError(
                f"Requested time window exceeds dataset. "
                f"Problematic idx: {idx}, "
                f"Context Length: {self.context_length}, "
                f"Forecast Horizon: {self.forecast_horizon}, "
                f"Calculated target_end: {target_end}, "
                f"Actual dataset length (len(self.freeboard)): {len(self.freeboard)}"
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
            f"{len(self.freeboard[0])} cells/day, "
            f"{self.num_raw_files} files loaded, "
            f"Patchify Algorithm: {self.algorithm}>" # What patchify algorithm was used
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


from torch.utils.data import DataLoader
from torch.utils.data import Subset

print(f"===== Making the Dataset Class: TRIAL_RUN MODE IS {TRIAL_RUN} ===== ")

# load all the data from one folder
dataset = DailyNetCDFDataset(data_dir)

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
    # --- Custom Splitting by Year ---
    
    # Convert dataset.times to pandas DatetimeIndex for easier year-based filtering
    all_times_pd = pd.to_datetime(dataset.times)

    # Define the start and end years for each set - keep this for the full dataset
    train_start_year = 1850
    train_end_year = 2011   
    val_start_year = 2012
    val_end_year = 2017
    test_start_year = 2018
    test_end_year = 2024
    
    # Get the boolean masks for each set
    train_mask = (all_times_pd.year >= train_start_year) & (all_times_pd.year <= train_end_year)
    val_mask = (all_times_pd.year >= val_start_year) & (all_times_pd.year <= val_end_year)
    test_mask = (all_times_pd.year >= test_start_year) & (all_times_pd.year <= test_end_year)

    # Get the integer indices where the masks are True
    train_indices = np.where(train_mask)[0].tolist()
    val_indices = np.where(val_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()
    
    # Create Subsets using the obtained indices
    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)
    test_set  = Subset(dataset, test_indices)

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

    # Get the global indices of the first and last elements in the subset
    first_global_idx = dataset_subset.indices[0]
    last_global_idx = dataset_subset.indices[-1]

    # Note: For the training, validation, and testing sets, each item (idx) represents the *start*
    # of a `context_length + forecast_horizon` window.
    # So, the start date of a set is the `dataset.times` value at the global index of its first item.
    start_date = dataset.times[first_global_idx] 

    # The last "day" considered in the last sample of the subset
    # is the `dataset.times` value at the global index of its last item
    # PLUS the `context_length + forecast_horizon - 1` days to get to the end of that last window.
    end_date_idx_for_last_sample = last_global_idx + dataset.context_length + dataset.forecast_horizon - 1
    end_date_technical = dataset.times[end_date_idx_for_last_sample]
    end_date = dataset.times[last_global_idx]

    print(f"{set_name} set start date: {start_date}")
    print(f"{set_name} set end date: {end_date} (technically ends {end_date_technical} -- last window for prediction)")
    logging.info(f"{set_name} set start date: {start_date}")
    logging.info(f"{set_name} set end date: {end_date} (technically ends {end_date_technical} -- last window for prediction)")
    
    return str(start_date), str(end_date)

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
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, drop_last=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, drop_last=True)

print("input_tensor should be of shape (context_length, num_patches, num_features, patch_size)")
print(f"actual input_tensor.shape = {input_tensor.shape}")
print("target_tensor should be of shape (forecast_horizon, num_patches, patch_size)")
print(f"actual target_tensor.shape = {target_tensor.shape}")


# # Transformer Class

# In[21]:


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
        Defaults to 7.
    forecast_horizon : int, optional
        The number of future days to predict for each patch.
        Defaults to 1.
    d_model : int, optional
        The dimension of the model's hidden states (embedding dimension).
        This is the size of the vectors that flow through the Transformer encoder.
        Defaults to 128.
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

# In[22]:


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


# TODO OPTION: Try temporal attention only (ex., Informer, Time Series Transformer).
# 
# # Save the Model

# In[23]:


# Define the path where to save or load the model
PATH = f"{model_version}_model.pth"

if TRAINING:
    
    # Save the model's state_dict
    torch.save(model.state_dict(), PATH)
    print(f"Saved model at {PATH}")


# # === BELOW - CAN BE USED ANY TIME FROM A .PTH FILE
# 
# Make sure and run the cells that contain constants or run all, but comment out the "save" and the training loop cell.

# # Re-Load the Model

# In[24]:


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


# # Metrics

# In[25]:


if EVALUATING_ON:
    import io
    import time # Import the time module for timing
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
    print(f"DEBUG: train_end: {train_end}")
    print(f"DEBUG: val_end: {val_end}")
    print(f"DEBUG: range for test_set: {range(val_end, total_days)}")
    print("==================")
    
    # --- Data Accumulators ---
    all_abs_errors_spatial = [] # To store absolute errors for each cell in each patch (for spatial maps)
    all_mse_errors_spatial = [] # To store MSE for each cell in each patch (for spatial maps)
    
    all_predicted_values_flat = [] # To accumulate all flattened predicted SIC values
    all_actual_values_flat = []    # To accumulate all flattened actual SIC values

    all_sic_temporal_results_dfs = [] # For SIC temporal degradation
    all_sie_temporal_results_dfs = [] # For SIE temporal degradation

    print("\n--- Running Test Set Evaluation for Temporal Degradation ---")
    start_time_test_eval = time.perf_counter()

    for i, (sample_x, sample_y, start_idx, end_idx, target_start_idx, target_end_idx) in enumerate(test_loader):
        # Move data to device
        sample_x = sample_x.to(device)
        sample_y = sample_y.to(device)
    
        # Reshape for model input
        B_sample, T_context, P_sample, C_sample, L_sample = sample_x.shape
        sample_x_reshaped = sample_x.view(B_sample, T_context, P_sample, C_sample * L_sample)
    
        with torch.no_grad():
            predicted_y_patches = loaded_model(sample_x_reshaped)
    
        # Ensure output shape matches target shape
        if predicted_y_patches.shape != sample_y.shape:
            print(f"Shape mismatch: Predicted {predicted_y_patches.shape}, Actual {sample_y.shape}")
            continue
    
        # Accumulate errors for spatial error calculation (these are still 4D tensors)
        diff = predicted_y_patches - sample_y
        all_abs_errors_spatial.append(torch.abs(diff).cpu())
        # Ensure the operation is done on the tensor, then move to CPU
        all_mse_errors_spatial.append((diff ** 2.0).cpu()) 
        
        # Accumulate flattened values for overall SIC distribution and classification
        all_predicted_values_flat.append(predicted_y_patches.cpu().numpy().flatten())
        all_actual_values_flat.append(sample_y.cpu().numpy().flatten())
    
        # --- Logic for temporal degradation analysis (for SIC and SIE) ---
        if dataset.times is not None:
            batch_size, forecast_horizon = predicted_y_patches.shape[0], predicted_y_patches.shape[1]
            
            for b in range(batch_size):
                base_date = dataset.times[target_start_idx[b]]
                
                for h in range(forecast_horizon):
                    step_date = base_date + pd.Timedelta(days=h)
                    
                    # Flatten the data for this forecast step for SIC temporal analysis
                    predicted_sic_step_flat = predicted_y_patches[b, h, :, :].cpu().numpy().flatten()
                    actual_sic_step_flat = sample_y[b, h, :, :].cpu().numpy().flatten()
                    
                    # Calculate SIE for this forecast step (squeeze to remove batch/time dim for calculate_sie)
                    # predicted_sic_2d_for_sie has shape (NUM_PATCHES, CELLS_PER_PATCH)
                    # actual_sic_2d_for_sie has shape (NUM_PATCHES, CELLS_PER_PATCH)
                    predicted_sic_2d_for_sie = predicted_y_patches[b, h, :, :].cpu().numpy()
                    actual_sic_2d_for_sie = sample_y[b, h, :, :].cpu().numpy()
                    
                    # --- Reconstruct full grid from patches for SIE calculation ---
                    # Initialize full grid arrays with NaNs, using the total number of cells from the dataset's mesh
                    total_grid_cells = len(dataset.latCell) # Assuming dataset.latCell holds the full number of cells
                    recovered_predicted_grid = np.full(total_grid_cells, np.nan)
                    recovered_actual_grid = np.full(total_grid_cells, np.nan)

                    # Populate the full grid using the patch data and mapping
                    # dataset.indices_per_patch_id contains masked indices
                    # dataset.masked_to_full maps masked indices to full indices
                    for patch_idx in range(P_sample): # P_sample is the number of patches in the batch
                        masked_cell_indices_in_patch = dataset.indices_per_patch_id[patch_idx]

                        # Get the values for this patch from the predicted/actual 2D arrays
                        predicted_values_in_patch = predicted_sic_2d_for_sie[patch_idx]
                        actual_values_in_patch = actual_sic_2d_for_sie[patch_idx]

                        # Map these values back to the full grid using the masked_to_full mapping
                        for i, masked_idx in enumerate(masked_cell_indices_in_patch):
                            full_idx = dataset.masked_to_full[masked_idx]
                            recovered_predicted_grid[full_idx] = predicted_values_in_patch[i]
                            recovered_actual_grid[full_idx] = actual_values_in_patch[i]

                    # Now pass the reconstructed full grids to the SIE calculation
                    predicted_sie_km = calculate_full_sie_in_kilometers(recovered_predicted_grid, nCells_array)
                    actual_sie_km = calculate_full_sie_in_kilometers(recovered_actual_grid, nCells_array)

                    # Store SIC results for temporal degradation
                    all_sic_temporal_results_dfs.append(pd.DataFrame({
                        'predicted': predicted_sic_step_flat,
                        'actual': actual_sic_step_flat,
                        'date': step_date,
                        'forecast_step': h + 1
                    }))
                    
                    # Store SIE results for temporal degradation
                    all_sie_temporal_results_dfs.append(pd.DataFrame({
                        'predicted_sie_km': [predicted_sie_km],
                        'actual_sie_km': [actual_sie_km],
                        'date': [step_date],
                        'forecast_step': [h + 1]
                    }))
    
    # Concatenate all temporal results into single DataFrames for analysis
    degradation_sic_df = pd.concat(all_sic_temporal_results_dfs, ignore_index=True) if all_sic_temporal_results_dfs else pd.DataFrame()
    degradation_sie_df = pd.concat(all_sie_temporal_results_dfs, ignore_index=True) if all_sie_temporal_results_dfs else pd.DataFrame()

    logging.info("\nSaving csv's...")
    # Save degradation data
    # The SIC per cell file is GB size; the SIE one is small (SIE as km^2 area averaged over all cells)
    if not degradation_sic_df.empty:
        degradation_sic_df.to_csv(f'{model_version}_sic_degradation.csv', index=False)
        print(f"Saved SIC performance degradation data as {model_version}_sic_degradation.csv")
    if not degradation_sie_df.empty:
        degradation_sie_df.to_csv(f'{model_version}_sie_degradation.csv', index=False)
        print(f"Saved SIE performance degradation data as {model_version}_sie_degradation.csv")

    end_time_test_eval = time.perf_counter()
    print(f"Elapsed time for saving SIC and SIE performance degradation csvs: {end_time_test_eval - start_time_test_eval:.2f} seconds")
    
    # --- Conditional Data collection for Training and Validation Sets ---
    if PLOT_DATA_SPLIT_DISTRIBUTION:
        print("--- Collecting ground truth data from training and validation sets ---")
        start_time_data_collection = time.perf_counter()
        
        all_train_actual_values = []
        all_val_actual_values = []
    
        for i, (sample_x, sample_y, *_) in enumerate(train_loader):
            all_train_actual_values.append(sample_y.cpu().numpy().flatten())
        
        for i, (sample_x, sample_y, *_) in enumerate(val_loader):
            all_val_actual_values.append(sample_y.cpu().numpy().flatten())
    
        final_train_values = np.concatenate(all_train_actual_values)
        final_val_values = np.concatenate(all_val_actual_values)
        
        end_time_data_collection = time.perf_counter()
        elapsed_time_data_collection = end_time_data_collection - start_time_data_collection
    else:
        # Define empty arrays if not plotting distribution to avoid NameError later
        final_train_values = np.array([])
        final_val_values = np.array([])

    logging.info("\nConcatenating values...")
    # --- Final Data Concatenation for overall metrics ---
    final_predicted_values = np.concatenate(all_predicted_values_flat) if all_predicted_values_flat else np.array([])
    final_actual_values = np.concatenate(all_actual_values_flat) if all_actual_values_flat else np.array([])
    
    if PLOT_DATA_SPLIT_DISTRIBUTION:
        print(f"Time for training/validation data collection: {elapsed_time_data_collection:.2f} seconds")
    
    # --- CALL FUNCTIONS HERE ---
    
    logging.info("\nCalling SIC plotting functions...")
    # 1. Calculate and Log Overall Spatial Errors (Overall SIC's MAE, MSE, RMSE)
    # This function calculates and prints overall spatial error metrics.
    start_time_spatial_errors = time.perf_counter()
    calculate_and_log_spatial_errors(degradation_sic_df, model_version, title_suffix=" (Overall)")
    end_time_spatial_errors = time.perf_counter()
    print(f"Elapsed time for Overall Spatial Error Calculation: {end_time_spatial_errors - start_time_spatial_errors:.2f} seconds")

    # 2. Plot Temporal Degradation (SIC Over Each Forecast Day)
    # This function plots MAE and RMSE degradation over the forecast horizon for SIC.
    start_time_sic_temporal_degradation = time.perf_counter()
    if not degradation_sic_df.empty:
        plot_SIC_temporal_degradation(degradation_sic_df, model_version, patching_strategy_abbr)
    end_time_sic_temporal_degradation = time.perf_counter()
    print(f"Elapsed time for SIC Temporal Degradation Plot: {end_time_sic_temporal_degradation - start_time_sic_temporal_degradation:.2f} seconds")

    # 3. Plot Actual vs. Predicted SIC Distribution (Overall SIC)
    # This function plots overlapping histograms of actual vs. predicted SIC values.
    start_time_actual_vs_predicted_sic_dist = time.perf_counter()
    plot_actual_vs_predicted_sic_distribution(final_actual_values, final_predicted_values, model_version, patching_strategy_abbr, num_bins=50, title_suffix=" (Overall)")
    end_time_actual_vs_predicted_sic_dist = time.perf_counter()
    print(f"Elapsed time for Overall Actual vs Predicted SIC histogram: {end_time_actual_vs_predicted_sic_dist - start_time_actual_vs_predicted_sic_dist:.2f} seconds")

    logging.info("\nCalling SIE plotting functions...")
    # 4. Log Classification Report (SIE as a binary value of SIC with 15% threshold)
    # This function provides a classification report for Sea Ice Extent (SIE).
    start_time_classification_report = time.perf_counter()
    # Assuming sie_threshold is defined elsewhere, e.g., sie_threshold = 0.15
    sie_threshold = 0.15 # Define sie_threshold if not already defined
    log_classification_report(final_actual_values, final_predicted_values, threshold=sie_threshold)
    end_time_classification_report = time.perf_counter()
    print(f"Elapsed time for Overall Classification Report: {end_time_classification_report - start_time_classification_report:.2f} seconds")
    
    # 5. Plot Overall SIE Confusion Matrix (SIE as a binary value of SIC with 15% threshold)
    # This function generates a confusion matrix plot for SIE classification.
    start_time_confusion_matrix = time.perf_counter()
    plot_sie_confusion_matrix(degradation_sic_df, threshold=sie_threshold, model_version=model_version, patching_strategy_abbr=patching_strategy_abbr, forecast_day=None) # None for overall
    end_time_confusion_matrix = time.perf_counter()
    print(f"Elapsed time for Overall Confusion Matrix Plot: {end_time_confusion_matrix - start_time_confusion_matrix:.2f} seconds")
    
    # 6. Plot Overall ROC Curve (SIE as a binary value of SIC with 15% threshold)
    # This function plots the Receiver Operating Characteristic (ROC) curve and calculates AUC.
    start_time_roc_curve = time.perf_counter()
    plot_roc_curve(degradation_sic_df, model_version=model_version, patching_strategy_abbr=patching_strategy_abbr, threshold=sie_threshold, forecast_day=None) # None for overall
    end_time_roc_curve = time.perf_counter()
    print(f"Elapsed time for Overall ROC Curve Plot: {end_time_roc_curve - start_time_roc_curve:.2f} seconds")

    # 7. Plot F1-Score Degradation (SIE as a binary value of SIC with 15% threshold)
    # This function plots the F1-score degradation for SIE classification.
    start_time_f1_degradation = time.perf_counter()
    if not degradation_sic_df.empty: # F1-score uses the same SIC temporal data
        plot_SIE_f1_score_degradation(degradation_sic_df, model_version, patching_strategy_abbr, threshold=sie_threshold)
    end_time_f1_degradation = time.perf_counter()
    print(f"Elapsed time for F1-Score Degradation Plot: {end_time_f1_degradation - start_time_f1_degradation:.2f} seconds")

    # 8. Plot SIE Degradation (SIE as the area that is ice in km^2)
    # This function plots MAE and RMSE degradation over the forecast horizon for SIE in km^2.
    start_time_sie_kilometers_degradation = time.perf_counter()
    if not degradation_sie_df.empty:
        plot_SIE_Kilometers_degradation(degradation_sie_df, model_version, patching_strategy_abbr)
    end_time_sie_kilometers_degradation = time.perf_counter()
    print(f"Elapsed time for SIE Kilometers Degradation Plot: {end_time_sie_kilometers_degradation - start_time_sie_kilometers_degradation:.2f} seconds")

    # 9. Conditional Plotting All Three SIC Distributions (Train, Val, Test Sets)
    # This block plots the distribution of SIC values across training, validation, and test sets.
    if PLOT_DATA_SPLIT_DISTRIBUTION:
        logging.info("\nPlotting data distributions (Train, Val, Test)...")
        print("\n--- Plotting data distributions (Train, Val, Test) ---")
        start_time_plot_data_dist = time.perf_counter()
        plot_sic_distribution_bars(
            train_data=final_train_values,
            val_data=final_val_values,
            test_data=final_actual_values,
            start_date=train_set_start_year,
            end_date=test_set_end_year,
            num_bins=10
        )
        end_time_plot_data_dist = time.perf_counter()
        print(f"Elapsed time for plotting data distribution comparison: {end_time_plot_data_dist - start_time_plot_data_dist:.2f} seconds")

        # Calculate Pairwise Jensen-Shannon Distances for Data Splits
        # This calculates the Jensen-Shannon Distance between the distributions of the data splits.
        print("\n--- Pairwise Jensen-Shannon Distances for Data Splits ---")
        start_time_jsd_pairwise = time.perf_counter()
        distributions_for_jsd = {
            'train': final_train_values,
            'validation': final_val_values,
            'test': final_actual_values
        }
        jsd_bins = np.linspace(0, 1, 10 + 1) # 10 bins for JSD calculation
        pairwise_jsd_results = jensen_shannon_distance_pairwise(distributions_for_jsd, jsd_bins)
        
        for pair, jsd_val in pairwise_jsd_results.items():
            print(f"JSD ({pair}): {jsd_val:.4f}")
        end_time_jsd_pairwise = time.perf_counter()
        print(f"Elapsed time for Jensen Shannon Pairwise Calculation: {end_time_jsd_pairwise - start_time_jsd_pairwise:.2f} seconds")

    if PLOT_DAY_BY_DAY_METRICS:
        
        logging.info("\nPlotting per-day forecast analysis...")
        # --- Optional: Per-Day Analysis for Classification Metrics and Distributions ---
        # This section iterates through each forecast day to provide detailed metrics and plots.
        print("\n############################################")
        print("\n#   PER-DAY FORECAST ANALYSIS (Optional)   #")
        print("\n############################################")
        start_time_per_day_analysis = time.perf_counter()
        for day in range(1, FORECAST_HORIZON + 1): # Loop through each forecast day
            df_day = degradation_sic_df[degradation_sic_df['forecast_step'] == day]
            if not df_day.empty:
                print(f"\n--- Metrics for Forecast Day {day} ---")
                
                # Log Classification Report for specific day
                log_classification_report(df_day['actual'].values, df_day['predicted'].values, threshold=sie_threshold)
                
                # Plot SIC distribution for specific day
                plot_actual_vs_predicted_sic_distribution(
                    df_day['actual'].values, df_day['predicted'].values, model_version, patching_strategy_abbr, num_bins=50, title_suffix=f" (Day {day})"
                )
    
                # Plot Confusion Matrix for specific day
                plot_sie_confusion_matrix(degradation_sic_df, sie_threshold, model_version, patching_strategy_abbr, forecast_day=day)
                
                # Plot ROC Curve for specific day
                plot_roc_curve(degradation_sic_df, model_version, patching_strategy_abbr, sie_threshold, forecast_day=day)
        
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

# In[26]:


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

# In[27]:


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
    # Assuming the model predicts ice area, which is the second feature (index 1)
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

