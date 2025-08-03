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
    "latitude_spillover_redo": patchify_with_spillover,
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
    "latlon_spillover": "LLSO",
    "latitude_neighbors": "LAT",
    "latitude_simple": "LSIM",
    "latitude_spillover_redo": "PSO", # Uses PSO (Patchify SpillOver)
    "lon_spilldown": "LSD",
    "rows": "ROW",
    "staggered_polar_descent": "SPD",
}


# # Variables for the Model
# 
# Check over these CAREFULLY!
# 
# Note that if you use the login node for training (even for the trial dataset that is much smaller), you run the risk of getting the error: # OutOfMemoryError: CUDA out of memory.

# In[5]:


# --- Time-Related Variables:
CONTEXT_LENGTH = 7            # T: Number of historical time steps used for input
FORECAST_HORIZON = 3          # Number of future time steps to predict (ex. 1 day for next time step)

# --- Model Hyperparameters
D_MODEL = 128                 # d_model: Dimension of the transformer's internal representations (embedding dimension)
N_HEAD = 8                    # nhead: Number of attention heads
NUM_TRANSFORMER_LAYERS = 4    # num_layers: Number of TransformerEncoderLayers
BATCH_SIZE = 16
NUM_EPOCHS = 10

# --- Performance-Related Variables:
NUM_WORKERS = 64

# --- Feature-Related Variables:
MAX_FREEBOARD_FOR_NORMALIZATION = 1    # Only works when you set MAX_FREEBOARD_ON too; bad results

# --- Space-Related Variables:
LATITUDE_THRESHOLD = 40          # Determines number of cells and patches (could use -90 to use the entire dataset).
CELLS_PER_PATCH = 256            # L: Number of cells within each patch

#PATCHIFY_TO_USE = "lon_spilldown"   # Change this to use other patching techniques
PATCHIFY_TO_USE = os.environ.get("SLURM_PATCHIFY_TO_USE", "lon_spilldown")

# --- Run Settings:
TRIAL_RUN =              False   # SET THIS TO USE THE PRACTICE SET (MUCH FASTER AND SMALLER, for debugging)
PLOT_DATA_DISTRIBUTION = True   # SET THIS TO PLOT THE OUTLIERS (Results are independent of patchify used)
NORMALIZE_ON =           True    # SET THIS TO USE NORMALIZATION ON FREEBOARD (Results are independent of patchify used)
TRAINING =               True    # SET THIS TO RUN THE TRAINING LOOP (Use on full dataset for results)
EVALUATING_ON =          True    # SET THIS TO RUN THE METRICS AT THE BOTTOM (Use on full dataset for results)

MAX_FREEBOARD_ON =       False   # To normalize with a pre-defined maximum for outlier handling
MAP_WITH_CARTOPY_ON =    False   # Make sure the Cartopy library is included in the kernel


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

print(model_version)


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



# # Example of one netCDF file with xarray

# In[18]:


# ds = xr.open_dataset("train/v3.LR.DTESTM.pm-cpu-10yr.mpassi.hist.am.timeSeriesStatsDaily.0010-01-01.nc")

from perlmutterpath import * # has the path to the data on Perlmutter
ds = xr.open_dataset(full_data_dir_sample, decode_times=True)


# In[19]:


ds.data_vars


# In[20]:


day_counter = ds["timeDaily_counter"].shape[0]
print(day_counter)


# In[21]:


print(ds["xtime_startDaily"])


# In[22]:


print(ds["xtime_startDaily"].values)


# In[23]:


ice_area = ds["timeDaily_avg_iceAreaCell"]
print(ds['timeDaily_avg_iceAreaCell'].attrs['long_name'])
print(f"Shape of ice area variable: {ice_area.shape}")


# In[24]:


ice_volume = ds["timeDaily_avg_iceVolumeCell"]
print(ds['timeDaily_avg_iceVolumeCell'].attrs['long_name'])
print(f"Shape of ice area variable: {ice_volume.shape}")


# In[25]:


print(ds.coords)
print(ds.dims)


# In[26]:


print(ds)
ds.close()


# # Example of Mesh File

# In[27]:


mesh = xr.open_dataset("NC_FILE_PROCESSING/mpassi.IcoswISC30E3r5.20231120.nc")


# In[28]:


mesh.data_vars


# In[29]:


print(mesh["latCell"].attrs['long_name'])
print(mesh["lonCell"].attrs['long_name'])


# In[30]:


cellsOnCell = mesh["cellsOnCell"].values
print(mesh["cellsOnCell"].attrs['long_name'])
print(mesh["cellsOnCell"].values)


# In[31]:


print(cellsOnCell.shape[1])


# In[32]:


print(mesh["cellsOnCell"].max().values)
print(mesh["cellsOnCell"].min().values)


# In[33]:


#np.save('cellsOnCell.npy', cellsOnCell) 


# In[34]:


#landIceMask = mesh["landIceMask"].values
#np.save('landIceMask.npy', landIceMask)


# In[35]:


print(mesh.coords)
print(mesh.dims)


# In[36]:


print(mesh)


# In[37]:


mesh.close()


# # Pre-processing + Freeboard calculation functions

# In[38]:


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


# In[39]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

def check_and_plot_freeboard(freeboard_data: np.ndarray, times_array: np.ndarray = None, status: str = "", y_limits: tuple = None):
    """
    Combines outlier checking and distribution plotting for a large freeboard dataset.
    Performs all calculations and then plots a bar chart and a custom boxplot for performance.
    
    Parameters
    ----------
    freeboard_data : np.ndarray
        The NumPy array of freeboard values (can be multi-dimensional).
    times_array : np.ndarray, optional
        The NumPy array of datetime objects corresponding to the time dimension
        of freeboard_data. Required to pinpoint dates of extremes.
    status : str
        A string indicating the normalization state (e.g., "pre_norm", "post_norm").
    """
    logging.info(f"--- Processing Freeboard Outliers and Distribution ({status}) ---")

    flat_freeboard = freeboard_data.flatten()
    data_shape = freeboard_data.shape # (Time, nCells)
    total_elements = len(flat_freeboard)
    
    logging.info(f"Data Shape: {data_shape}, Samples: {total_elements}")

    # --- IQR Outlier Detection (First, as we need these values for the plot) ---
    Q1 = np.percentile(flat_freeboard, 25)
    median = np.percentile(flat_freeboard, 50)
    Q3 = np.percentile(flat_freeboard, 75)
    IQR = Q3 - Q1
    
    lower_bound_theoretical = Q1 - 1.5 * IQR
    upper_bound_theoretical = Q3 + 1.5 * IQR
    
    # Correctly find the whisker end points (the lowest/highest data points within the bounds)
    lower_whisker = np.min(flat_freeboard[flat_freeboard >= lower_bound_theoretical]) if np.any(flat_freeboard >= lower_bound_theoretical) else flat_freeboard.min()
    upper_whisker = np.max(flat_freeboard[flat_freeboard <= upper_bound_theoretical]) if np.any(flat_freeboard <= upper_bound_theoretical) else flat_freeboard.max()
    
    outliers_low = flat_freeboard[flat_freeboard < lower_whisker]
    outliers_high = flat_freeboard[flat_freeboard > upper_whisker]
    # --- Log the Metrics ---
    norm_state = "Post-Norm" if "post_norm" in status else "Pre-Norm"
    trial_state = "Trial Dataset (2020 - 2024)" if "trial" in status else "Full Dataset (1850 - 2024)"
    logging.info(f"Freeboard Absolute Minimum Value ({norm_state}): {flat_freeboard.min():.4f}")
    logging.info(f"Freeboard Absolute Maximum Value ({norm_state}): {flat_freeboard.max():.4f}")

    if times_array is not None and data_shape[0] == len(times_array):
        
        # Find unique year extremes
        # Reshape data to be a long-form DataFrame for easy grouping by year
        df = pd.DataFrame({
            'value': flat_freeboard,
            'date': np.repeat(times_array, data_shape[1])
        })
        
        df['year'] = df['date'].dt.year

        # Find min/max values for each year
        annual_extremes = df.groupby('year').agg(
            min_value=('value', 'min'),
            min_date=('date', lambda x: x[x.idxmin()]),
            max_value=('value', 'max'),
            max_date=('date', lambda x: x[x.idxmax()])
        ).reset_index()

        # Find the overall top 10 from these annual extremes
        min_extremes = annual_extremes.nsmallest(10, 'min_value').rename(columns={'min_value': 'value', 'min_date': 'date'})
        max_extremes = annual_extremes.nlargest(10, 'max_value').rename(columns={'max_value': 'value', 'max_date': 'date'})

        logging.info("\nTop 10 Extreme Minimum Freeboard Values (unique years):")
        for i, row in min_extremes.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Rank {i+1}: Value={row['value']:.4f}, Date={date_str}")
        
        logging.info("\nTop 10 Extreme Maximum Freeboard Values (unique years):")
        for i, row in max_extremes.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Rank {i+1}: Value={row['value']:.4f}, Date={date_str}")
    
    else:
        logging.warning("Cannot pinpoint dates of extremes: times_array not provided or shape mismatch.")

    num_outliers = len(outliers_low) + len(outliers_high)
    logging.info(f"{norm_state}")
    logging.info(f"Freeboard Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
    logging.info(f"Freeboard Lower Bound: {lower_bound_theoretical:.4f}, Upper Bound: {upper_bound_theoretical:.4f}")
    logging.info(f"Total outliers: {num_outliers} ({num_outliers / total_elements * 100:.2f}% of total)")

    if num_outliers > 0:
        logging.warning("Potential outliers detected!")

    # --- Plotting Section ---
    if "post_norm" in status.lower():
        unit_label = "Freeboard Value (Normalized)"
    else:
        unit_label = "Freeboard (m)"
        
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left Plot: Bar Chart Distribution
    bins = [0, 0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    hist_counts, bin_edges = np.histogram(flat_freeboard, bins=bins)
    plot_labels = [f"{bin_edges[i]:.3f}-{bin_edges[i+1]:.3f}" for i in range(len(bin_edges)-1)]
    axes[0].bar(plot_labels, hist_counts, color='skyblue', edgecolor='black')
    axes[0].set_title(f'Freeboard Value Distribution ({status})')
    axes[0].set_xlabel(unit_label)
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Right Plot: Manual Box and Whisker Plot
    x_pos = 1
    box_width = 0.5
    
    # The Box (IQR)
    box = patches.Rectangle((x_pos - box_width/2, Q1), box_width, IQR, 
                            facecolor='lightcoral', edgecolor='black')
    axes[1].add_patch(box)

    # Median
    axes[1].plot([x_pos - box_width/2, x_pos + box_width/2], [median, median], 
                 color='black', linewidth=2)

    # Whiskers
    axes[1].plot([x_pos, x_pos], [lower_whisker, Q1], color='gray', linestyle='--')
    axes[1].plot([x_pos, x_pos], [Q3, upper_whisker], color='gray', linestyle='--')
    
    # Caps
    axes[1].plot([x_pos - box_width/4, x_pos + box_width/4], [lower_whisker, lower_whisker], color='gray', linewidth=2)
    axes[1].plot([x_pos - box_width/4, x_pos + box_width/4], [upper_whisker, upper_whisker], color='gray', linewidth=2)

    # Note: We are intentionally not plotting outliers for performance reasons.
    axes[1].set_title(f'Boxplot of Freeboard ({status})')
    axes[1].set_ylabel(unit_label)
    axes[1].set_xticks([])
    axes[1].set_xlim(0, 2)

    # Use the provided y_limits, otherwise set a dynamic scale based on the data
    if y_limits:
        axes[1].set_ylim(y_limits)
    else:
        axes[1].set_ylim(min(0, lower_whisker), max(upper_whisker, 0.3) * 1.1)

    plt.suptitle(f'Freeboard Analysis {trial_state} ({norm_state})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = os.path.join(os.getcwd(), f"FB_analysis_{status}.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    logging.info(f"Freeboard analysis plot saved as {filename}")


# In[40]:


def analyze_ice_area_imbalance(ice_area_data: np.ndarray, status = ""):
    """
    Measures and logs the percentage of ice_area data points that are 0, 1, or between 0 and 1.
    
    Parameters
    ----------
    ice_area_data : np.ndarray
        The NumPy array of ice_area values (can be multi-dimensional).
    """
    logging.info(f"--- Analyzing Ice Area Imbalance {status} ---")

    flat_ice_area = ice_area_data.flatten()
    total_elements = len(flat_ice_area)

    if total_elements == 0:
        logging.warning("Ice Area data is empty, cannot analyze imbalance.")
        return

    count_zero = np.sum(flat_ice_area == 0)
    count_one = np.sum(flat_ice_area == 1)
    count_between = np.sum((flat_ice_area > 0) & (flat_ice_area < 1))

    percent_zero = (count_zero / total_elements) * 100
    percent_one = (count_one / total_elements) * 100
    percent_between = (count_between / total_elements) * 100

    logging.info(f"Total Ice Area data points: {total_elements}")
    logging.info(f"Percentage of values == 0: {percent_zero:.2f}% ({count_zero} points)")
    logging.info(f"Percentage of values == 1: {percent_one:.2f}% ({count_one} points)")
    logging.info(f"Percentage of values between 0 and 1 (exclusive): {percent_between:.2f}% ({count_between} points)")

    
    # Optional check for values outside [0, 1] range, if any
    count_invalid = np.sum((flat_ice_area < 0) | (flat_ice_area > 1))
    if count_invalid > 0:        
        logging.warning(f"Found {count_invalid} ice_area values outside the [0, 1] range!")
        print(f"Found {count_invalid} ice_area values outside the [0, 1] range!")
        
        logging.info(f"Minimum ice area: {flat_ice_area.min()}")
        logging.info(f"Maximum ice area: {flat_ice_area.max()}")

def plot_ice_area_imbalance(ice_area_data: np.ndarray, status: str = ""):
    """
    Creates a bar chart to visualize the imbalance of ice_area values (0, 1, or between 0-1).
    Saves the chart as a PNG file.
    
    Parameters
    ----------
    ice_area_data : np.ndarray
        The NumPy array of ice_area values to plot (can be multi-dimensional).
    save_path : str
        The directory where the plot PNG file will be saved.
    """
    logging.info(f"--- Plotting Ice Area Imbalance Chart {status} ---")

    flat_ice_area = ice_area_data.flatten()
    total_elements = len(flat_ice_area)

    if total_elements == 0:
        logging.warning("Ice Area data is empty, cannot plot imbalance.")
        return

    # Calculate counts and percentages for each category
    count_zero = np.sum(flat_ice_area == 0)
    count_00_to_25_percent = np.sum((flat_ice_area > 0) & (flat_ice_area < 0.25))
    count_25_to_50_percent = np.sum((flat_ice_area >= 0.25) & (flat_ice_area < 0.5))
    count_50_to_75_percent = np.sum((flat_ice_area >= 0.5) & (flat_ice_area < 0.75))
    count_75_to_99_percent = np.sum((flat_ice_area >= 0.75) & (flat_ice_area < 1))
    count_one = np.sum(flat_ice_area == 1)
    
    categories = ['Exactly 0', '0 < x < 0.25', '0.25 <= x < 0.5', '0.5 <= x < 0.75', '0.75 <= x < 1', 'Exactly 1']
    counts = [count_zero, count_00_to_25_percent, count_25_to_50_percent, count_50_to_75_percent, count_75_to_99_percent, count_one]
    percentages = [(c / total_elements) * 100 for c in counts]
    
    # Create a DataFrame for cleaner plotting with seaborn
    df_imbalance = pd.DataFrame({
        'Category': categories,
        'Percentage': percentages
    })

    logging.info("--- Creating the subplots ---")
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Plot with Seaborn ---
    sns.barplot(data=df_imbalance, x='Category', y='Percentage', hue='Category', palette='viridis', ax=ax, legend=False)
    
    ax.set_title(f'Distribution of Ice Area Values for {status.capitalize()} Dataset', fontsize=16)
    ax.set_xlabel('Value Category', fontsize=12)
    ax.set_ylabel('Percentage of Data (%)', fontsize=12)
    ax.set_ylim(0, max(percentages) * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels on top of the bars
    for index, row in df_imbalance.iterrows():
        ax.text(index, row['Percentage'] + 1, f"{row['Percentage']:.2f}%",
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save to the current working directory
    current_directory = os.getcwd()
    filename = os.path.join(current_directory, f"{model_mode}_{norm}_SIC_imbalance_{status}.png")
    
    plt.savefig(filename, dpi=300) # dpi=300 for high-quality image
    plt.close(fig) 
    logging.info(f"Ice Area imbalance chart saved as {model_mode}_{norm}_SIC_imbalance_{status}.png")


# # Custom Pytorch Dataset
# Example from NERSC of using ERA5 Dataset:
# 
# https://github.com/NERSC/dl-at-scale-training/blob/main/utils/data_loader.py

# # __ init __ - masks and loads the data into tensors

# In[41]:


import os
import time
from datetime import datetime
from datetime import timedelta

from torch.utils.data import Dataset
from typing import List, Union, Callable, Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler

from NC_FILE_PROCESSING.patchify_utils import *
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
        plot_outliers_and_imbalance: bool = PLOT_DATA_DISTRIBUTION, # set FALSE FOR FINAL
        trial_run: bool = TRIAL_RUN, # Use the trial data directory
        num_patches: int = NUM_PATCHES,
        cells_per_patch: int = CELLS_PER_PATCH,
        patchify_func: Callable = DEFAULT_PATCHIFY_METHOD_FUNC, # Default patchify function
        patchify_func_key: str = PATCHIFY_TO_USE, # Key to look up specific params
        max_freeboard_for_normalization: int = MAX_FREEBOARD_FOR_NORMALIZATION,
        max_freeboard_on: bool = MAX_FREEBOARD_ON,
    
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

        start_time = time.time()
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
        
        # --- 1. Gather files (sorted for deterministic order) ---------
        if self.trial_run:
            
            # USE THIS FOR PRACTICE (SMALLER CHUNK OF DATA)-
            self.file_paths = sorted(
                [
                    os.path.join(self.data_dir, f)
                    for f in os.listdir(self.data_dir)

                    # GET 4 YEAR SUBSET 2020 - 2024
                    if f.startswith("v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.202") and f.endswith(".nc")
                ]
            )

        else:
            # USE THE FULL DATASET 
            self.data_dir = data_dir
            self.file_paths = sorted(
                [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)

                    # GET ALL - 1850 TO 2024
                    if f.startswith("v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.") and f.endswith(".nc")
                ]
            )
        
        logging.info(f"Found {len(self.file_paths)} NetCDF files:")
        if not self.file_paths:
            raise FileNotFoundError(f"No *.nc files found in {data_dir!r}")
        
        # --- 2. Load the mesh file. Latitudes and Longitudes are in radians. ---
        mesh = xr.open_dataset(mesh_dir)
        latCell = np.degrees(mesh["latCell"].values)
        lonCell = np.degrees(mesh["lonCell"].values)
        mesh.close()
        
        # Initialize the cell mask
        self.cell_mask = latCell >= latitude_threshold        
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

        logging.info(f"=== Extracting raw data and times in a single loop === ")

        all_times_list = []
        ice_area_all_list = []
        ice_volume_all_list = []
        snow_volume_all_list = []
        
        for i, path in enumerate(self.file_paths):
            ds = xr.open_dataset(path)

            # --- 3. Store a list of datetimes from each file -> helps with retrieving 1 day's data later
            # Extract times from byte string format
            xtime_byte_array = ds["xtime_startDaily"].values
            xtime_unicode_array = xtime_byte_array.astype(str)
            xtime_cleaned_array = np.char.replace(xtime_unicode_array, "_", " ")
            times_array = np.asarray(xtime_cleaned_array, dtype='datetime64[s]')
            all_times_list.append(times_array)

            # --- 4. Extract raw data
            ice_area = ds["timeDaily_avg_iceAreaCell"].values
            ice_volume = ds["timeDaily_avg_iceVolumeCell"].values
            snow_volume = ds["timeDaily_avg_snowVolumeCell"].values

            # --- 5. Apply a mask to the nCells
            ice_area = ice_area[:, self.cell_mask]
            ice_volume = ice_volume[:, self.cell_mask]
            snow_volume = snow_volume[:, self.cell_mask]

            # Append masked data to lists
            ice_area_all_list.append(ice_area)
            ice_volume_all_list.append(ice_volume)
            snow_volume_all_list.append(snow_volume)

            ds.close() # Close dataset after processing

        # --- Concatenate all collected data into single NumPy arrays after the loop
        self.times = np.concatenate(all_times_list, axis=0)
        self.ice_area = np.concatenate(ice_area_all_list, axis=0)
        ice_volume_combined = np.concatenate(ice_volume_all_list, axis=0)
        snow_volume_combined = np.concatenate(snow_volume_all_list, axis=0)

        # Checking the dates
        logging.info(f"Parsed {len(self.times)} total dates")
        logging.info(f"First few: {str(self.times[:5])}")

        # Stats on how many dates there are
        logging.info(f"Total days collected: {len(self.times)}")
        logging.info(f"Unique days: {len(np.unique(self.times))}")
        logging.info(f"First 35 days: {self.times[:35]}")
        logging.info(f"Last 35 days: {self.times[-35:]}")

        logging.info(f"Shape of combined ice_area array: {self.ice_area.shape}")
        logging.info(f"Elapsed time for combined data/time loading: {time.time() - start_time} seconds")
        
        # --- 6. Derive Freeboard from ice area, snow volume and ice volume
        logging.info(f"=== Calculating Freeboard === ")
        self.freeboard = compute_freeboard(self.ice_area, ice_volume_combined, snow_volume_combined)
        logging.info(f"Elapsed time for freeboard calculation: {time.time() - start_time} seconds")
        
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
        logging.info(f"Elapsed time for patchifying with the {self.algorithm} algorithm: {time.time() - start_time} seconds")

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
            logging.info(f"Elapsed time for plotting the outliers and imbalance {time.time() - start_time} seconds")

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
                logging.info(f"Elapsed time for normalizing the Freeboard: {time.time() - start_time} seconds")
                check_and_plot_freeboard(self.freeboard, self.times, f"{status}_fb_post_norm")
        
        logging.info("End of __init__")
        end_time = time.time()
        logging.info(f"Elapsed time for DailyNetCDFDataset __init__: {end_time - start_time} seconds")
        print(f"Elapsed time for DailyNetCDFDataset __init__: {end_time - start_time} seconds")
        print(f"In minutes:                {(end_time - start_time)//60} minutes")

    def __len__(self):
        """
        Returns the total number of possible starting indices (idx) for a valid sequence.
        A valid sequence needs `self.context_length` days for input and `self.forecast_horizon` days for target.
        
        ex) If the total number of days is 365, the context_length is 7 and the forecast_horizon is 3, then
        
        365 - (7 + 3) + 1 = 365 - 10 + 1 = 356 valid starting indices
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
            Shape: (num_patches, num_features, patch_size)
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
            f"<DailyNetCDFDataset: {len(self)} days, "
            f"{len(self.freeboard[0])} cells/day, "
            f"{len(self.file_paths)} files loaded, "
            f"Patchify Algorithm: {self.algorithm}>" # Added algorithm to repr
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


# In[42]:





# # DataLoader

# In[ ]:


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
    #end_date_idx_for_last_sample = last_global_idx + dataset.context_length + dataset.forecast_horizon - 1
    #end_date = dataset.times[end_date_idx_for_last_sample]
    end_date = dataset.times[last_global_idx]

    print(f"{set_name} set start date: {start_date}")
    print(f"{set_name} set end date: {end_date}")
    logging.info(f"{set_name} set start date: {start_date}")
    logging.info(f"{set_name} set end date: {end_date}")
    
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
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=2, drop_last=True)

print("input_tensor should be of shape (context_length, num_patches, num_features, patch_size)")
print(f"actual input_tensor.shape = {input_tensor.shape}")
print("target_tensor should be of shape (forecast_horizon, num_patches, patch_size)")
print(f"actual target_tensor.shape = {target_tensor.shape}")


# # Transformer Class

# In[ ]:


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
        start_time = time.time()

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

        end_time = time.time()
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


# In[ ]:





# # Training Loop

# In[ ]:


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
    start_time = time.time()
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
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logging.info("===============================================")
    logging.info(f"Elapsed time for TRAINING: {elapsed_time:.2f} seconds")
    logging.info("===============================================")
    print("===============================================")
    print(f"Elapsed time for TRAINING: {elapsed_time:.2f} seconds")
    print("===============================================")


# In[ ]:





# TODO OPTION: Try temporal attention only (ex., Informer, Time Series Transformer).
# 
# # Save the Model

# In[ ]:


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

# In[ ]:


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

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import itertools # To help with pairwise combinations

# Jensen-Shannon Distance function
def jensen_shannon_distance(p, q):
    """Calculates the Jensen-Shannon distance between two probability distributions."""
    # Ensure distributions sum to 1
    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (p + q)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    jsd = 0.5 * (entropy(p + epsilon, m + epsilon) + entropy(q + epsilon, m + epsilon))
    return np.sqrt(jsd) # JSD is the square root of JS divergence
    
def jensen_shannon_distance_pairwise(distributions: dict, bins: np.ndarray):
    """
    Calculates the pairwise Jensen-Shannon Distance between multiple probability distributions.

    Parameters
    ----------
    distributions : dict
        A dictionary where keys are distribution names (e.g., 'train', 'val', 'test')
        and values are 1D NumPy arrays of data.
    bins : np.ndarray
        The bin edges to use for discretizing the distributions into histograms.

    Returns
    -------
    dict
        A dictionary of pairwise JSDs, with keys like 'dist1_vs_dist2'.
    """
    jsd_results = {}
    
    # First, convert all distributions to normalized histograms
    histograms = {}
    for name, data in distributions.items():
        # Compute histogram for the given data and bins
        hist, _ = np.histogram(data, bins=bins, density=True)
        # Normalize to sum to 1 (important for JSD)
        histograms[name] = hist / hist.sum()
    
    # Calculate JSD for all unique pairs
    for (name1, hist1), (name2, hist2) in itertools.combinations(histograms.items(), 2):
        # Add a small epsilon to avoid log(0) in entropy calculation
        epsilon = 1e-10
        m = 0.5 * (hist1 + hist2)
        
        jsd = 0.5 * (entropy(hist1 + epsilon, m + epsilon) + entropy(hist2 + epsilon, m + epsilon))
        jsd_results[f"{name1}_vs_{name2}"] = np.sqrt(jsd) # JSD is the square root of JS divergence
        
    return jsd_results

def plot_sic_distribution_bars(train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray, start_date: str, end_date: str, num_bins: int = 10):
    """
    Plots the distribution of Sea Ice Concentration for the training, validation,
    and testing sets using side-by-side bar plots, with zeros plotted in a
    separate subplot for clarity. The Y-axis represents the percentage of
    data points for each bin within each dataset.

    Parameters
    ----------
    train_data : np.ndarray
        Flattened array of ground-truth SIC values from the training set.
    val_data : np.ndarray
        Flattened array of ground-truth SIC values from the validation set.
    test_data : np.ndarray
        Flattened array of ground-truth SIC values from the test set.
    start_date : str
        The start date of the dataset (e.g., "1850" for the full dataset).
    end_date : str
        The end date of the dataset (e.g., "2024" for the full dataset).
    num_bins : int
        The number of bins for the histogram (excluding the 0 bin).
    """
    logging.info(f"--- Plotting SIC Distribution Comparison with Separate Zeros ({num_bins} bins) ---")
    
    # Define bins from 0 to 1
    bins = np.linspace(0, 1, num_bins + 1)
    
    # Calculate histograms (counts) for each dataset
    train_counts, _ = np.histogram(train_data, bins=bins)
    val_counts, _ = np.histogram(val_data, bins=bins)
    test_counts, _ = np.histogram(test_data, bins=bins)
    
    # --- Convert counts to percentages for a fair comparison ---
    train_total = len(train_data)
    val_total = len(val_data)
    test_total = len(test_data)

    train_percentages = (train_counts / train_total) * 100
    val_percentages = (val_counts / val_total) * 100
    test_percentages = (test_counts / test_total) * 100
    
    # --- Separate the zero-value percentages from the rest ---
    train_zeros_pc = train_percentages[0]
    val_zeros_pc = val_percentages[0]
    test_zeros_pc = test_percentages[0]
    
    train_non_zeros_pc = train_percentages[1:]
    val_non_zeros_pc = val_percentages[1:]
    test_non_zeros_pc = test_percentages[1:]

    # Create bin labels for the non-zero values, now as percentages
    non_zero_bin_labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}%" for i in range(1, len(bins)-1)]
    
    # Create DataFrames for easy plotting with Seaborn
    df_zeros = pd.DataFrame({
        'Dataset': ['Training', 'Validation', 'Testing'],
        'Percentage': [train_zeros_pc, val_zeros_pc, test_zeros_pc]
    })
    
    df_non_zeros = pd.DataFrame({
        'Bin': non_zero_bin_labels * 3,
        'Dataset': ['Training'] * len(train_non_zeros_pc) + ['Validation'] * len(val_non_zeros_pc) + ['Testing'] * len(test_non_zeros_pc),
        'Percentage': np.concatenate([train_non_zeros_pc, val_non_zeros_pc, test_non_zeros_pc])
    })
    
    # --- Create the figure with two subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 4]})
    
    # Left Subplot: Zeros only
    sns.barplot(data=df_zeros, x='Dataset', y='Percentage', hue='Dataset', palette='deep', ax=axes[0], legend=False)
    axes[0].set_title('Percentage of Zero-Value Data Points', fontsize=14)
    axes[0].set_xlabel('Dataset', fontsize=12)
    axes[0].set_ylabel('Percentage (%)', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Right Subplot: Non-zeros
    sns.barplot(data=df_non_zeros, x='Bin', y='Percentage', hue='Dataset', palette='deep', ax=axes[1])
    axes[1].set_title(f'Distribution of Non-Zero SIC Values ({num_bins} Bins)', fontsize=14)
    axes[1].set_xlabel('Sea Ice Concentration Value Bins', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Overall plot title
    plot_title = f"Sea Ice Concentration Distribution - Train, Valid, Test Sets ({start_date}-{end_date})"
    plt.suptitle(plot_title, fontsize=16)
    
    # Save the plot
    filename = os.path.join(os.getcwd(), f"SIC_Distribution_Comparison_Bars_SeparateZeros_Percentages_{start_date}_{end_date}_{num_bins}bins.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    logging.info(f"SIC distribution comparison bar plot with separate zeros saved as {filename}")


# In[ ]:


import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay # <<< Added for classification metrics

if EVALUATING_ON:

    start_time = time.time()
    
    import io
    
    # Create a string buffer to capture output
    captured_output = io.StringIO()
    
    # Redirect stdout to the buffer
    sys.stdout = captured_output
    
    from scipy.stats import entropy
    
    # Accumulators for errors
    all_abs_errors = [] # To store absolute errors for each cell in each patch
    all_mse_errors = [] # To store MSE for each cell in each patch
    
    # Accumulators for histogram data
    all_predicted_values_flat = []
    all_actual_values_flat = []

    start_time_test_eval = time.perf_counter()
    
    print("\nStarting evaluation and metric calculation...")
    print("==================")
    print(f"DEBUG: Batch Size: {BATCH_SIZE} Days")
    print(f"DEBUG: Context Length: {CONTEXT_LENGTH} Days")
    print(f"DEBUG: Forecast Horizon: {FORECAST_HORIZON} Days")
    print(f"DEBUG: Number of batches in test_loader (with drop_last=True): {len(test_loader)} Batches")
    print("==================")
    print(f"DEBUG: len(test_set): {len(test_set)} Days")
    print(f"DEBUG: len(dataset) for splitting: {len(dataset)} Days")
    print(f"DEBUG: train_end: {train_end}")
    print(f"DEBUG: val_end: {val_end}")
    print(f"DEBUG: range for test_set: {range(val_end, total_days)}")
    print("==================")
    
    # --- Test Set Evaluation (including model inference) ---
    print("\n--- Running Test Set Evaluation ---")
    start_time_test_eval = time.perf_counter()
    
    # Iterate over the test_loader
    # (B, forecast_horizon, P, CELLS_PER_PATCH) to match the model's output.
    for i, (sample_x, sample_y, start_idx, end_idx, target_start, target_end) in enumerate(test_loader):
        print(f"Processing batch {i+1}/{len(test_loader)}")
        
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

    # --- Conditional Data collection for Training and Validation Sets ---
    if PLOT_DATA_DISTRIBUTION:
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
    
    # --- Final Data Concatenation ---
    if all_abs_errors and all_mse_errors:
        combined_abs_errors = torch.cat(all_abs_errors, dim=0)
        combined_mse_errors = torch.cat(all_mse_errors, dim=0)
        mean_abs_error_per_cell_patch = combined_abs_errors.mean(dim=(0, 1)).numpy()
        mean_mse_per_cell_patch = combined_mse_errors.mean(dim=(0, 1)).numpy()
        
    if all_predicted_values_flat and all_actual_values_flat:
        final_predicted_values = np.concatenate(all_predicted_values_flat)
        final_actual_values = np.concatenate(all_actual_values_flat)
    
    # --- Printing and Saving Metrics ---
    end_time_test_eval = time.perf_counter()
    elapsed_time_test_eval = end_time_test_eval - start_time_test_eval
    print(f"\nTime for test set evaluation (with inference): {elapsed_time_test_eval:.2f} seconds")
    
    if PLOT_DATA_DISTRIBUTION:
        print(f"Time for training/validation data collection: {elapsed_time_data_collection:.2f} seconds")
 
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
    
    # --- Conditional Plotting All Three SIC Distributions ---
    if PLOT_DATA_DISTRIBUTION:
        print("\n--- Plotting distributions ---")
        start_time_plot = time.perf_counter()

        plot_sic_distribution_bars(
            train_data=final_train_values,
            val_data=final_val_values,
            test_data=final_actual_values,
            start_date=train_set_start_year, 
            end_date=test_set_end_year,    
            num_bins=10 
        )
        
        end_time_plot = time.perf_counter()
        elapsed_time_plot = end_time_plot - start_time_plot
        print(f"Time for plotting distribution comparison: {elapsed_time_plot:.2f} seconds")

        # --- Calculate Pairwise Jensen-Shannon Distances ---
        print("\n--- Pairwise Jensen-Shannon Distances for Data Splits ---")
        distributions_for_jsd = {
            'train': final_train_values,
            'validation': final_val_values,
            'test': final_actual_values
        }
        # Use the same bins as the histogram for consistency
        jsd_bins = np.linspace(0, 1, 10 + 1) # 10 bins for JSD calculation
        pairwise_jsd_results = jensen_shannon_distance_pairwise(distributions_for_jsd, jsd_bins)
        
        for pair, jsd_val in pairwise_jsd_results.items():
            print(f"JSD ({pair}): {jsd_val:.4f}")
            
    # Define bins for the histogram (e.g., for ice concentration between 0 and 1)
    # Adjust bins based on the expected range of your data
    bins = np.linspace(0, 1, 51) # 50 bins from 0 to 1

    # Compute histograms
    hist_predicted, _ = np.histogram(final_predicted_values, bins=bins, density=True)
    hist_actual, _ = np.histogram(final_actual_values, bins=bins, density=True)

    # Normalize histograms to sum to 1 (they are already density=True, but re-normalize for safety)
    hist_predicted = hist_predicted / hist_predicted.sum()
    hist_actual = hist_actual / hist_actual.sum()

    # Calculate Jensen-Shannon Distance
    jsd = jensen_shannon_distance(hist_actual, hist_predicted)
    print(f"\nJensen-Shannon Distance between actual and predicted histograms: {jsd:.4f}")

    # Plotting Histograms
    plt.figure(figsize=(10, 6))
    plt.hist(final_actual_values, bins=bins, alpha=0.7, label='Actual Data', color='skyblue', density=True)
    plt.hist(final_predicted_values, bins=bins, alpha=0.7, label='Predicted Data', color='salmon', density=True)
    plt.title(f'Actual vs. Predicted SIC for {patching_strategy_abbr} Patchify')
    plt.xlabel('Ice Concentration Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f"{model_version}_SIC_xy.png")
    plt.close()

    print("\nEvaluation complete.")

    end_time = time.time()
    print(f"Elapsed time for evaluation: {end_time - start_time:.2f} seconds")
    
    # When reading the histograms, look for overlap:
    # High Overlap: predictions are close to actual values. Decent model.
    # Low Overlap: predictions differ from actual values, issues with the model. 

    # Restore stdout
    sys.stdout = sys.__stdout__
    
    # Now, write the captured output to the file
    with open(f'{model_version}_Metrics.txt', 'w') as f:
        f.write(captured_output.getvalue())
    
    print(f"Metrics saved as {model_version}_Metrics.txt")



# # Make a Single Prediction

# In[ ]:


if EVALUATING_ON:
    
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


# In[ ]:




