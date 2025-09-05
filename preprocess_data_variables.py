# preprocess_data_variables.py
import os
import time
import glob
import xarray as xr
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import zarr

# Import your custom modules
from perlmutterpath import * # Contains data_dir, mesh_dir, etc.

import logging

logging.basicConfig(filename='monthly_variable_preprocessing.log', filemode='w', level=logging.INFO)

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
    trial_run: bool,
    output_dir: str = "./",
    model_mode: str = None,
    norm: str = None
):
    """
    Performs data loading and freeboard calculation, saving the results to Zarr.

        Handle the raw data:
        1) Gather the sorted monthly data from each netCDF file (1 file = 1 month of monthly data)
            The netCDF files contain nCells worth of data per month for each feature (ice area, ice volume, etc.)
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
    print(f"Starting data preprocessing with model_mode={model_mode}, norm={norm}")
    logging.info(f"Starting data preprocessing with model_mode={model_mode}, norm={norm}")

    # Define paths for the Zarr stores
    zarr_path_unnormalized = os.path.join(output_dir, f"Monthly_{model_mode}_{norm}_preprocessed_data.zarr")
    zarr_path_normalized = os.path.join(output_dir, f"Monthly_{model_mode}_{norm}_normalized_data.zarr")

    # This is the new, conditional logic block
    if os.path.exists(zarr_path_unnormalized):
        logging.info(f"Found existing un-normalized data at '{zarr_path_unnormalized}'. Loading...")
        ds_unnormalized = xr.open_zarr(zarr_path_unnormalized)
        logging.info("Existing data loaded successfully.")
    else:
        logging.info("No existing un-normalized data found. Starting from scratch.")
        
        # 1. Gather files
        file_pattern = os.path.join(data_dir, "v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsMonthly.202*.nc") if trial_run else os.path.join(data_dir, "v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsMonthly.*.nc")
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

        full_to_masked = {full_idx: new_idx for new_idx, full_idx in enumerate(np.where(cell_mask)[0])}
        masked_to_full = {v: k for k, v in full_to_masked.items()}
        
        # 3. Load raw data with xarray.open_mfdataset
        logging.info("Loading raw data with xarray.open_mfdataset...")
        with xr.open_mfdataset(
            file_pattern,
            combine='nested',
            concat_dim='Time',
            parallel=False,
            data_vars=['timeMonthly_avg_iceAreaCell', 'timeMonthly_avg_iceVolumeCell', 'timeMonthly_avg_snowVolumeCell', 'xtime_startMonthly'],
            decode_times=True,
            engine='netcdf4',
            chunks={'Time': 30}
        ) as combined_ds:
            xtime_byte_array = combined_ds["xtime_startMonthly"].values
            xtime_unicode_array = xtime_byte_array.astype(str)
            xtime_cleaned_array = np.char.replace(xtime_unicode_array, "_", " ")
            times = np.asarray(xtime_cleaned_array, dtype='datetime64[s]')
            
            # Use .compute() to load into memory after masking
            ice_area = combined_ds["timeMonthly_avg_iceAreaCell"][:, cell_mask].compute().values
            ice_volume_combined = combined_ds["timeMonthly_avg_iceVolumeCell"][:, cell_mask].compute().values
            snow_volume_combined = combined_ds["timeMonthly_avg_snowVolumeCell"][:, cell_mask].compute().values

        logging.info(f"Elapsed time for raw data loading and masking: {time.perf_counter() - start_time:.2f} seconds.")

        # 4. Derive Freeboard
        logging.info("Calculating Freeboard...")
        freeboard_raw = compute_freeboard(ice_area, ice_volume_combined, snow_volume_combined)
        logging.info("Freeboard calculation completed.")
        
        # Create the un-normalized dataset
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

        # 5. Save the un-normalized data to a Zarr store
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Saving un-normalized data as '{zarr_path_unnormalized}'...")
        ds_unnormalized.to_zarr(zarr_path_unnormalized, mode='w', compute=True)
        logging.info("Un-normalized data saved.")

    # Now, whether the data was loaded or newly created, ds_unnormalized is available.
    
    # 6. Optional: Normalize and save
    if normalize_on:
        # Check if normalized data already exists
        if os.path.exists(zarr_path_normalized):
            logging.info(f"Normalized data already exists at '{zarr_path_normalized}'. Skipping normalization.")
        else:
            logging.info(f"=== Normalizing Freeboard with Scikit-learn MinMaxScaler === ")
            # Use a fresh copy of the unnormalized data to prevent modifying the original
            freeboard_raw = ds_unnormalized['freeboard'].values 
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

            freeboard_normalized = scaler.transform(freeboard_reshaped).reshape(freeboard_raw.shape)
            freeboard_normalized = np.clip(freeboard_normalized, 0, 1)

            ds_normalized = ds_unnormalized.copy(deep=True)
            ds_normalized['freeboard'] = (("time", "nCells_masked"), freeboard_normalized)
            
            logging.info(f"Saving normalized data as '{zarr_path_normalized}'...")
            ds_normalized.to_zarr(zarr_path_normalized, mode='w', compute=True)
            logging.info("Normalized data saved.")
    else:
        logging.info("Normalization is turned off. No normalized data will be created.")

    end_time = time.perf_counter()
    logging.info(f"Total preprocessing time: {(end_time - start_time):.2f} seconds.")

if __name__ == "__main__":

    LATITUDE_THRESHOLD = 40
    TRIAL_RUN = False
    NORMALIZE_ON = True
    MAX_FREEBOARD_ON = False
    MAX_FREEBOARD_FOR_NORMALIZATION = 1

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

    # Define and pass your parameters
    preprocess_and_save_variables_to_zarr(
        data_dir=data_dir,
        mesh_dir=mesh_dir,
        latitude_threshold=LATITUDE_THRESHOLD,
        normalize_on=NORMALIZE_ON,
        max_freeboard_for_normalization=MAX_FREEBOARD_FOR_NORMALIZATION,
        max_freeboard_on=MAX_FREEBOARD_ON,
        trial_run=TRIAL_RUN,
        model_mode=model_mode,
        norm=norm,
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
        
