import numpy as np
import matplotlib.pyplot as plt
import os
import itertools # To help with pairwise combinations
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


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
    dataset.times : np.ndarray, optional
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
    logging.info(f"Freeboard analysis plot saved as FB_analysis_{status}.png")


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

####################################
# DATASET POST-PROCESSING METRICS  #
####################################

# WORKS
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
    print(f"--- Plotting SIC Distribution Comparison with Separate Zeros ({num_bins} bins) ---")

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
    filename = os.path.join(os.getcwd(), f"SIC_Distribution_Comparison_Bars_SeparateZeros_Percentages_{start_date}_{end_date}_{num_bins}_bins.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"SIC distribution comparison bar plot with separate zeros saved as SIC_Distribution_Comparison_Bars_SeparateZeros_Percentages_{start_date}_{end_date}_{num_bins}_bins.png")


#######################
# PREDICTION METRICS  #
#######################

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


########
# SIC  #
########
# WORKS
def plot_SIC_temporal_degradation(df: pd.DataFrame, model_version: str, patching_strategy: str):
    """
    Calculates and plots error metrics (MAE, RMSE) as a function of forecast horizon
    and optionally by month/year.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'predicted', 'actual', 'date', and 'forecast_step' columns.
    model_version : str
        The version/name of the model for saving files.
    patching_strategy : str
        The patching strategy for file naming.
    """
    if df.empty:
        logging.warning("No data provided for temporal degradation analysis.")
        return

    print("--- Calculating Temporal Degradation Metrics ---")
    
    # Calculate errors for each data point
    df['abs_error'] = np.abs(df['predicted'] - df['actual'])
    df['squared_error'] = (df['predicted'] - df['actual'])**2

    # Group by forecast step and calculate metrics
    degradation_by_step = df.groupby('forecast_step').agg(
        MAE=('abs_error', 'mean'),
        RMSE=('squared_error', lambda x: np.sqrt(x.mean()))
    ).reset_index()

    # --- Plot Degradation Curve (MAE and RMSE vs. Forecast Step) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(degradation_by_step['forecast_step'], degradation_by_step['MAE'], label='MAE', marker='o')
    ax.plot(degradation_by_step['forecast_step'], degradation_by_step['RMSE'], label='RMSE', marker='x')
    
    ax.set_title(f"Error Degradation over Forecast Horizon for {patching_strategy}")
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True)
    
    plot_filename = f"{model_version}_temporal_degradation.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"Temporal degradation plot saved as {model_version}_temporal_degradation.png")

    # --- Optional: Analyze Degradation by Month or Year ---
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year

    # Calculate metrics by month and forecast step
    degradation_by_month = df.groupby(['month', 'forecast_step']).agg(
        MAE=('abs_error', 'mean')
    ).reset_index()

    # Plot a heatmap of MAE by month and forecast step
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap_data = degradation_by_month.pivot(index='month', columns='forecast_step', values='MAE')
    sns.heatmap(heatmap_data, cmap='viridis', ax=ax, cbar_kws={'label': 'MAE'})
    ax.set_title(f"MAE Degradation by Month and Forecast Horizon for {patching_strategy}")
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("Month")
    plt.tight_layout()
    plt.savefig(f"{model_version}_monthly_degradation.png")
    plt.close()
    print(f"Monthly degradation heatmap saved.")  

########
# SIE  #
########

def calculate_full_sie_in_kilometers(sic_array: np.ndarray, cell_area: [float, np.ndarray] = 25 * 25):
    """
    Calculates the Sea Ice Extent (SIE) from a 2D Sea Ice Concentration (SIC) array.
    
    Parameters
    ----------
    sic_array : np.ndarray
        A 2D array of Sea Ice Concentration values (0-1).
    cell_area : float or np.ndarray
        The area of a single grid cell in square kilometers. Can be a float (if all cells
        are the same size) or a NumPy array (if cell areas vary per grid cell).
        Default is 25x25 km.
    
    Returns
    -------
    float
        The total Sea Ice Extent in square kilometers.
    """
    # Create a boolean mask for ice (SIC >= 15%)
    ice_mask = (sic_array >= 0.15)
    
    if isinstance(cell_area, (int, float)):
        # If cell_area is a single value, multiply the count of ice cells by this area
        sie_area = np.sum(ice_mask) * cell_area
    elif isinstance(cell_area, np.ndarray):
        # If cell_area is an array, perform element-wise multiplication with the mask
        # and then sum the areas of the ice cells
        if ice_mask.shape != cell_area.shape:
            raise ValueError(f"Shape mismatch: sic_array mask {ice_mask.shape} and cell_area array {cell_area.shape} must be the same.")
        sie_area = np.sum(ice_mask * cell_area)
    else:
        raise TypeError("cell_area must be a float, int, or numpy.ndarray.")
        
    return sie_area

# WORKS
# --- SIE Degradation Plotting ---
def plot_SIE_Kilometers_degradation(df: pd.DataFrame, model_version: str, patching_strategy: str):
    """
    Calculates and plots error metrics (MAE, RMSE) for SIE as a function of
    forecast horizon.
    """
    if df.empty:
        logging.warning("No SIE data provided for temporal degradation analysis.")
        return

    print("--- Calculating SIE Temporal Degradation Metrics ---")
    
    df['abs_error'] = np.abs(df['predicted_sie_km'] - df['actual_sie_km'])
    df['squared_error'] = (df['predicted_sie_km'] - df['actual_sie_km'])**2

    degradation_by_step = df.groupby('forecast_step').agg(
        MAE=('abs_error', 'mean'),
        RMSE=('squared_error', lambda x: np.sqrt(x.mean()))
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(degradation_by_step['forecast_step'], degradation_by_step['MAE'], label='MAE', marker='o')
    ax.plot(degradation_by_step['forecast_step'], degradation_by_step['RMSE'], label='RMSE', marker='x')
    
    ax.set_title(f"SIE Error Degradation over Forecast Horizon for {patching_strategy}")
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("Error (km²)")
    ax.legend()
    ax.grid(True)
    
    plot_filename = f"{model_version}_sie_KM_temporal_degradation.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"SIE temporal degradation plot saved as {model_version}_sie_KM_temporal_degradation.png")

# WORKS
# --- F1-Score Degradation Plotting ---
def plot_SIE_f1_score_degradation(df: pd.DataFrame, model_version: str, patching_strategy: str, threshold: float = 0.15):
    """
    Calculates and plots the F1-score for sea ice classification as a function of
    the forecast horizon.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'predicted' and 'actual' SIC values.
    model_version : str
        The version/name of the model for saving files.
    patching_strategy : str
        The patching strategy for file naming.
    threshold : float
        The SIC threshold (e.15 for 15%) to classify a pixel as "ice".
    """
    if df.empty:
        logging.warning("No data provided for F1-score degradation analysis.")
        return

    print("--- Calculating F1-Score Degradation Metrics ---")
    
    # Classify each pixel based on the threshold
    df['actual_class'] = np.where(df['actual'] > threshold, 1, 0)
    df['predicted_class'] = np.where(df['predicted'] > threshold, 1, 0)

    # Calculate F1-score for each forecast step
    # Added include_groups=False to silence the FutureWarning
    f1_scores_by_step = df.groupby('forecast_step').apply(
        lambda x: f1_score(x['actual_class'], x['predicted_class'], zero_division=0),
        include_groups=False
    ).reset_index(name='f1_score')

    # --- Plot the F1-score Degradation Curve ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(f1_scores_by_step['forecast_step'], f1_scores_by_step['f1_score'], label='F1-Score', marker='o', color='forestgreen')
    
    ax.set_title(f"F1-Score Degradation over Forecast Horizon ({threshold*100:.0f}% threshold)")
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("F1-Score")
    ax.legend()
    ax.grid(True)
    
    plot_filename = f"{model_version}_f1_score_degradation.png"
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()
    print(f"F1-score degradation plot saved as {model_version}_f1_score_degradation.png")

def calculate_and_log_spatial_errors(data_df: pd.DataFrame, model_version: str, title_suffix: str = ""):
    """
    Calculates and logs spatial error metrics (MAE, MSE, RMSE)
    and saves the per-cell/per-patch error arrays.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing 'predicted' and 'actual' values (e.g., degradation_sic_df or a filtered version).
    model_version : str
        The version/name of the model for saving files.
    title_suffix : str, optional
        Additional text to append to the title for flexibility (e.g., " (Day 5)").
    """
    
    if data_df.empty:
        logging.warning(f"No data provided for spatial error analysis{title_suffix}.")
        return

    # Calculate errors from the DataFrame
    abs_errors = np.abs(data_df['predicted'].values - data_df['actual'].values)
    mse_errors = (data_df['predicted'].values - data_df['actual'].values)**2

    # These are now 1D arrays, so we just take their mean directly
    mean_abs_error = np.mean(abs_errors)
    mean_mse = np.mean(mse_errors)

    print(f"\n--- Error Metrics (Averaged per Cell per Patch{title_suffix}) ---")
    print(f"Overall Mean Absolute Error: {mean_abs_error:.4f}")

    mse = mean_mse
    print(f"Overall Mean Squared Error: {mse:.4f}")
    rmse = np.sqrt(mse)
    print(f"Overall Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Determine filename suffix based on title_suffix
    file_suffix = ""
    if "overall" in title_suffix.lower():
        file_suffix = "_overall"
    elif "day" in title_suffix.lower():
        # Extract day number if present, e.g., " (Day 5)" -> "_Day5"
        try:
            day_num = int(''.join(filter(str.isdigit, title_suffix)))
            file_suffix = f"_Day{day_num}"
        except ValueError:
            file_suffix = "_custom" # Fallback if day number not found
    else:
        file_suffix = "_custom" # Default if no specific suffix in title

def log_classification_report(final_actual_values: np.ndarray, final_predicted_values: np.ndarray, threshold: float = 0.15):
    """
    Applies a threshold to SIC values and logs the classification report.

    Parameters
    ----------
    final_actual_values : np.ndarray
        Flattened array of actual SIC values.
    final_predicted_values : np.ndarray
        Flattened array of predicted SIC values.
    threshold : float
        The SIC threshold (e.g., 0.15) to classify a pixel as "ice".
    """
    print(f"\n--- Sea Ice Extent (SIE) Metrics (Threshold > {threshold}) ---")
    sie_actual_flat = np.where(final_actual_values > threshold, 1, 0)
    sie_predicted_flat = np.where(final_predicted_values > threshold, 1, 0)

    print("\nClassification Report:")
    report = classification_report(sie_actual_flat, sie_predicted_flat, target_names=['No Ice', 'Ice'], zero_division=0)
    print(report)

# WORKS
def plot_sie_confusion_matrix(df_sic_temporal: pd.DataFrame, threshold: float, model_version: str, patching_strategy_abbr: str, forecast_day: int = None):
    """
    Calculates and plots the confusion matrix for SIE classification with percentages.
    Can plot for overall data or a specific forecast day.

    Parameters
    ----------
    df_sic_temporal : pd.DataFrame
        DataFrame containing 'predicted', 'actual', and 'forecast_step' columns.
    threshold : float
        The SIC threshold (e.g., 0.15) to classify a pixel as "ice".
    model_version : str
        The version/name of the model for saving files.
    patching_strategy_abbr : str
        The patching strategy abbreviation for plot titles/filenames.
    forecast_day : int, optional
        If provided, plots the confusion matrix for this specific forecast day.
        Otherwise, uses all data (overall performance).
    """
    plot_suffix = ""
    if forecast_day is not None:
        df_filtered = df_sic_temporal[df_sic_temporal['forecast_step'] == forecast_day]
        plot_suffix = f" (Day {forecast_day})"
    else:
        df_filtered = df_sic_temporal
        plot_suffix = " (Overall)"
    
    if df_filtered.empty:
        logging.warning(f"No data for confusion matrix for forecast day {forecast_day if forecast_day is not None else 'overall'}.")
        return

    sie_actual_flat = np.where(df_filtered['actual'] > threshold, 1, 0)
    sie_predicted_flat = np.where(df_filtered['predicted'] > threshold, 1, 0)

    cm = confusion_matrix(sie_actual_flat, sie_predicted_flat)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_labels = np.array([
        [f"{cm[0, 0]:,}\n({cm_percent[0, 0]:.2%})", f"{cm[0, 1]:,}\n({cm_percent[0, 1]:.2%})"],
        [f"{cm[1, 0]:,}\n({cm_percent[1, 0]:.2%})", f"{cm[1, 1]:,}\n({cm_percent[1, 1]:.2%})"]
    ])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues', xticklabels=['No Ice', 'Ice'], yticklabels=['No Ice', 'Ice'], ax=ax)
    ax.set_title(f'Sea Ice Extent Confusion Matrix (Threshold > {threshold}){plot_suffix} for {patching_strategy_abbr}', fontsize=16)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    
    filename_suffix = f"_Day{forecast_day}" if forecast_day is not None else "_Overall"
    plt.savefig(f"{model_version}_SIE_Confusion_Matrix{filename_suffix}.png")
    plt.close()
    print(f"Confusion matrix plot saved as {model_version}_SIE_Confusion_Matrix{filename_suffix}.png")

# WORKS
def plot_roc_curve(df_sic_temporal: pd.DataFrame, model_version: str, patching_strategy_abbr: str, threshold: float = 0.15, forecast_day: int = None):
    """
    Calculates and plots the ROC curve and AUC score.
    Can plot for overall data or a specific forecast day.

    Parameters
    ----------
    df_sic_temporal : pd.DataFrame
        DataFrame containing 'predicted', 'actual', and 'forecast_step' columns.
    model_version : str
        The version/name of the model for saving files.
    patching_strategy_abbr : str
        The patching strategy abbreviation for plot titles/filenames.
    threshold : float
        The SIC threshold (e.g., 0.15) used for binary classification (for y_true_binary).
    forecast_day : int, optional
        If provided, plots the ROC curve for this specific forecast day.
        Otherwise, uses all data (overall performance).
    """
    plot_suffix = ""
    if forecast_day is not None:
        df_filtered = df_sic_temporal[df_sic_temporal['forecast_step'] == forecast_day]
        plot_suffix = f" (Day {forecast_day})"
    else:
        df_filtered = df_sic_temporal
        plot_suffix = " (Overall)"

    if df_filtered.empty:
        logging.warning(f"No data for ROC curve for forecast day {forecast_day if forecast_day is not None else 'overall'}.")
        return

    print(f"\n--- ROC Curve and AUC Metrics{plot_suffix} ---")

    y_true_binary = np.where(df_filtered['actual'] > threshold, 1, 0)
    y_scores = df_filtered['predicted']

    # Calculate ROC curve data
    fpr, tpr, thresholds_roc = roc_curve(y_true_binary, y_scores)

    # Calculate AUC score
    auc = roc_auc_score(y_true_binary, y_scores)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve{plot_suffix} {patching_strategy_abbr}')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    filename_suffix = f"_Day{forecast_day}" if forecast_day is not None else "_Overall"
    plt.savefig(f"{model_version}_ROC_Curve{filename_suffix}.png")
    plt.close()
    
    print(f"\nArea Under the Curve (AUC): {auc:.4f}")
    print(f"ROC curve plot saved as {model_version}_ROC_Curve{filename_suffix}.png")

# WORKS    
def plot_actual_vs_predicted_sic_distribution(final_actual_values: np.ndarray, final_predicted_values: np.ndarray, model_version: str, patching_strategy_abbr: str, num_bins: int = 50, title_suffix: str = ""):
    """
    Plots overlapping histograms of actual vs. predicted SIC values and calculates JSD.

    Parameters
    ----------
    final_actual_values : np.ndarray
        Flattened array of actual SIC values.
    final_predicted_values : np.ndarray
        Flattened array of predicted SIC values.
    model_version : str
        The version/name of the model for saving files.
    patching_strategy_abbr : str
        The patching strategy abbreviation for plot titles/filenames.
    num_bins : int
        Number of bins for the histograms.
    title_suffix : str, optional
        Additional text to append to the title for flexibility (e.g., " (Day 5)").
    """
    bins = np.linspace(0, 1, num_bins + 1)
    hist_predicted, _ = np.histogram(final_predicted_values, bins=bins, density=True)
    hist_actual, _ = np.histogram(final_actual_values, bins=bins, density=True)

    # Normalize histograms to sum to 1
    hist_predicted = hist_predicted / hist_predicted.sum()
    hist_actual = hist_actual / hist_actual.sum()

    # Calculate Jensen-Shannon Distance
    jsd = jensen_shannon_distance(hist_actual, hist_predicted)
    print(f"\nJensen-Shannon Distance between Actual vs. Predicted SIC for {patching_strategy_abbr} patchify {title_suffix}') {jsd:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_actual_values, bins=bins, alpha=0.7, label='Actual Data', color='skyblue', density=True)
    plt.hist(final_predicted_values, bins=bins, alpha=0.7, label='Predicted Data', color='salmon', density=True)
    plt.title(f'Actual vs. Predicted SIC for {patching_strategy_abbr} Patchify{title_suffix}')
    plt.xlabel('Ice Concentration Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    filename_suffix = ""
    if "overall" in title_suffix.lower():
        filename_suffix = "_Overall"
    elif "day" in title_suffix.lower():
        try:
            day_num = int(''.join(filter(str.isdigit, title_suffix)))
            filename_suffix = f"_Day{day_num}"
        except ValueError:
            filename_suffix = "_Custom"
    else:
        filename_suffix = "_Custom" # Default if no specific suffix in title

    plt.savefig(f"{model_version}_SIC_xy{filename_suffix}.png")
    plt.close()
    print(f"Actual vs. Predicted SIC histogram saved as {model_version}_SIC_xy{filename_suffix}.png")
