# preprocess_data_variables.py
import os
import time
import glob
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import zarr

# Import your custom modules
from perlmutterpath import * # Contains data_dir, mesh_dir, PROCESSED_DATA_DIR, etc.
from NC_FILE_PROCESSING.metrics_and_plots import compute_freeboard # Assuming this is available
from VisionTransformer import *

if TRIAL_RUN:
    model_mode = "tr" # Training Dataset
else:
    model_mode = "fd" # Full Dataset

import logging

logging.basicConfig(filename='variable_preprocessing.log', filemode='w', level=logging.INFO)

# Constants (adjust if you use different units)
D_WATER = 1023  # Density of seawater (kg/m^3)
D_ICE = 917     # Density of sea ice (kg/m^3)
D_SNOW = 330    # Density of snow (kg/m^3)

MIN_SIC = 1e-6

def compute_freeboard(sea_ice_concentration: np.ndarray, 
                      ice_volume: np.ndarray, 
                      snow_volume: np.ndarray) -> np.ndarray:
    """
    Compute sea ice freeboard from ice and snow grid cell averaged thickness and sea_ice_concentration.
    
    Parameters
    ----------
    sea_ice_concentration : np.ndarray
        Sea ice concentration in percent (called timeDaily_avg_iceAreaCell in E3SM)
    ice_volume : np.ndarray
        Grid cell averaged ice thickness in meters (called timeDaily_avg_iceVolumeCell in E3SM)
    snow_volume : np.ndarray
        Grid cell averaged snow thickness in meters (called timeDaily_avg_snowVolumeCell in E3SM)
    All arrays are the same shape
    
    Returns
    -------
    freeboard : np.ndarray
        Freeboard height for each cell, same shape as inputs.
    """
    # Initialize the freeboard array with zeros
    freeboard = np.zeros_like(sea_ice_concentration)
    
    # Create a boolean mask for valid areas to prevent division by zero
    valid_mask = sea_ice_concentration > MIN_SIC
    
    # Calculate freeboard only for the valid cells in a single vectorized step
    # This avoids creating intermediate height_ice and height_snow arrays.
    if np.any(valid_mask):
        valid_area = sea_ice_concentration[valid_mask]
        valid_ice_avg_thickness = ice_volume[valid_mask]
        valid_snow_avg_thickness = snow_volume[valid_mask]

        freeboard[valid_mask] = (
            (valid_ice_avg_thickness / valid_area) * (D_WATER - D_ICE) / D_WATER +
            (valid_snow_avg_thickness / valid_area) * (D_WATER - D_SNOW) / D_WATER
        )
    
    return freeboard

def preprocess_and_save_variables_to_zarr(
    data_dir: str,
    mesh_dir: str,
    latitude_threshold: int,
    normalize_on: bool,
    max_freeboard_for_normalization: int,
    max_freeboard_on: bool,
    trial_run: bool, # Optional - use the data in the trial directory instead of the full dataset
    output_dir: str = "./"
):
    """
    Performs data loading and freeboard calculation, saving the results to Zarr.

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
        7) Optional: Normalize the data (Ice area is already between 0 and 1; Freeboard is not) 

    Parameters
    ----------
    data_dir : str
        Directory containing NetCDF files
    latitude_threshold
        The minimum latitude to use for Arctic data

    """
    start_time = time.perf_counter()
    
    # 1. Gather files
    file_pattern = os.path.join(data_dir, "v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.202*.nc") if trial_run else os.path.join(data_dir, "v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.*.nc")
    file_paths = sorted(glob.glob(file_pattern))
    if not file_paths:
        raise FileNotFoundError(f"No *.nc files found matching the pattern in {data_dir}")
    
    num_raw_files = len(file_paths)
    logging.info(f"Found {num_raw_files} NetCDF files.")
    
    # 2. Load mesh and create mask
    mesh = xr.open_dataset(mesh_dir)
    latCell = np.degrees(mesh["latCell"].values)
    cell_mask = latCell >= latitude_threshold
    mesh.close()
    
    masked_ncells_size = np.count_nonzero(cell_mask)
    logging.info(f"Mask size: {masked_ncells_size}")

    # Create mappings for full to masked indices
    full_to_masked = {full_idx: new_idx for new_idx, full_idx in enumerate(np.where(cell_mask)[0])}
    masked_to_full = {v: k for k, v in full_to_masked.items()}

    # 3. Load raw data with xarray.open_mfdataset
    # Specify 'timeDaily_avg_iceAreaCell', 'timeDaily_avg_iceVolumeCell', 'timeDaily_avg_snowVolumeCell'
    # as the variables to load.
    # Use combine='nested' and concat_dim='Time' because the NetCDF files do not have explicit time coordinates,
    # but the 'Time' dimension exists and files are sorted by name.
    # parallel=True enables Dask for parallel file opening and processing.   
    with xr.open_mfdataset(
        file_pattern,
        combine='nested',
        concat_dim='Time',
        parallel=True,
        data_vars=['timeDaily_avg_iceAreaCell', 'timeDaily_avg_iceVolumeCell', 'timeDaily_avg_snowVolumeCell', 'xtime_startDaily'],
        decode_times=True
    ) as combined_ds:
        
        xtime_byte_array = combined_ds["xtime_startDaily"].values
        xtime_unicode_array = xtime_byte_array.astype(str)
        xtime_cleaned_array = np.char.replace(xtime_unicode_array, "_", " ")
        times = np.asarray(xtime_cleaned_array, dtype='datetime64[s]')
        
        ice_area = combined_ds["timeDaily_avg_iceAreaCell"][:, cell_mask].compute().values
        ice_volume_combined = combined_ds["timeDaily_avg_iceVolumeCell"][:, cell_mask].compute().values
        snow_volume_combined = combined_ds["timeDaily_avg_snowVolumeCell"][:, cell_mask].compute().values

    logging.info(f"Elapsed time for raw data loading: {time.perf_counter() - start_time:.2f} seconds.")

    # 4. Derive Freeboard
    logging.info("Calculating Freeboard...")
    freeboard_raw = compute_freeboard(ice_area, ice_volume_combined, snow_volume_combined)
    logging.info("Freeboard calculation completed.")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Save the un-normalized data to a Zarr store
    zarr_path_unnormalized = os.path.join(output_dir, "{model_mode}_preprocessed_data_unnormalized.zarr")
    logging.info(f"Saving un-normalized data as {model_mode}_preprocessed_data_unnormalized.zarr")
    
    ds_unnormalized = xr.Dataset(
        {
            "ice_area": (("time", "nCells_masked"), ice_area),
            "freeboard": (("time", "nCells_masked"), freeboard_raw),
            "cell_mask": ("nCells_full", cell_mask),
            "full_to_masked": ((), str(full_to_masked)),
            "masked_to_full": ((), str(masked_to_full)),
            "times": ("time", times),
            "num_raw_files": ((), num_raw_files),
        },
        coords={
            "time": times,
            "nCells_masked": np.arange(ice_area.shape[1]),
            "nCells_full": np.arange(len(latCell)),
        }
    )
    ds_unnormalized.to_zarr(zarr_path_unnormalized, mode='w', compute=True)
    logging.info("Un-normalized data saved.")

    # 6. Normalize and save the second version
    # Normalize the Freeboard (Area is already between 0 and 1; Freeboard is not)
    if normalize_on:
        logging.info(f"=== Normalizing Freeboard with Scikit-learn MinMaxScaler === ")
        freeboard_normalized = freeboard_raw.copy()
        freeboard_reshaped = freeboard_normalized.reshape(-1, 1)

        if max_freeboard_on:
            min_val = 0
            max_val = max_freeboard_for_normalization
        else:
            min_val = freeboard_reshaped.min()
            max_val = freeboard_reshaped.max()
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit([[min_val], [max_val]])

         # Transform the freeboard data and reshape it back to the original shape
        freeboard_normalized = scaler.transform(freeboard_reshaped).reshape(freeboard_raw.shape)
        freeboard_normalized = np.clip(freeboard_normalized, 0, 1)

        zarr_path_normalized = os.path.join(output_dir, "{model_mode}_preprocessed_data_normalized.zarr")
        logging.info(f"Saving normalized data as {model_mode}_preprocessed_data_normalized.zarr")
        
        ds_normalized = ds_unnormalized.copy(deep=True) # Start with a copy
        ds_normalized['freeboard'] = (("time", "nCells_masked"), freeboard_normalized)
        
        ds_normalized.to_zarr(zarr_path_normalized, mode='w', compute=True)
        logging.info("Normalized data saved.")
    
    end_time = time.perf_counter()
    logging.info(f"Total preprocessing time: {(end_time - start_time):.2f} seconds.")

if __name__ == "__main__":

    # Define and pass your parameters
    preprocess_and_save_variables_to_zarr(
        data_dir=data_dir,
        mesh_dir=mesh_dir,
        output_dir=PROCESSED_DATA_DIR,
        latitude_threshold=LATITUDE_THRESHOLD,
        normalize_on=NORMALIZE_ON,
        max_freeboard_for_normalization=MAX_FREEBOARD_FOR_NORMALIZATION,
        max_freeboard_on=MAX_FREEBOARD_ON,
        trial_run=TRIAL_RUN
    )


    # plot_outliers_and_imbalance
    #     Optional - check outliers and imbalance on the variables Ice Area and Freeboard

    # if self.plot_outliers_and_imbalance:
    #         logging.info(f"=== Plotting Outliers and Imbalance === ")
    #         check_and_plot_freeboard(self.freeboard, self.times, f"{status}_fb_pre_norm")
    #         analyze_ice_area_imbalance(self.ice_area)
    #         plot_ice_area_imbalance(self.ice_area, status)
    #         logging.info(f"Elapsed time for plotting the outliers and imbalance {time.perf_counter() - start_time} seconds")

    #         # Reshape freeboard data from (days, cells) to (total_samples, 1) for the scaler
    #         freeboard_reshaped = self.freeboard.reshape(-1, 1)

    #         # Determine the min and max values for the scaler based on the flag
    #         if self.max_freeboard_on:
    #             logging.info(f"Using custom max freeboard value: {self.max_freeboard_for_normalization}")
    #             min_val = 0
    #             max_val = self.max_freeboard_for_normalization
    #         else:
    #             logging.info(f"Using min and max values from the data.")
    #             min_val = freeboard_reshaped.min()
    #             max_val = freeboard_reshaped.max()

    # if self.plot_outliers_and_imbalance:
    #             logging.info(f"Elapsed time for normalizing the Freeboard: {time.perf_counter() - start_time} seconds")
    #             check_and_plot_freeboard(self.freeboard, self.times, f"{status}_fb_post_norm")
        
