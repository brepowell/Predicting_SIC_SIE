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

####################
# DATASET METRICS  #
####################

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

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import itertools # To help with pairwise combinations

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


#######################
# PREDICTION METRICS  #
#######################

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import itertools
from scipy.stats import entropy
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score
)

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

def get_predicted_actual_arrays(ds: xr.Dataset):
    """
    Helper function to get flattened predicted and actual values from an xarray Dataset.
    """
    predicted = ds['predicted'].values.flatten()
    actual = ds['actual'].values.flatten()
    return predicted, actual

def plot_actual_vs_predicted_sic_distribution(
    predicted_flat: np.ndarray, 
    actual_flat: np.ndarray, 
    model_version: str, 
    patching_strategy_abbr: str, 
    num_bins: int = 50, 
    title_suffix: str = ""
):
    """
    Plots overlapping histograms of actual vs. predicted SIC values and calculates JSD.

    Parameters
    ----------
    predicted_flat : np.ndarray
        1D array of all predicted Sea Ice Concentration values.
    actual_flat : np.ndarray
        1D array of all actual Sea Ice Concentration values.
    model_version : str
        The version/name of the model for saving files.
    patching_strategy_abbr : str
        The patching strategy abbreviation for plot titles/filenames.
    num_bins : int
        Number of bins for the histograms.
    title_suffix : str, optional
        Additional text to append to the title.
    """

    bins = np.linspace(0, 1, num_bins + 1)
    hist_predicted, _ = np.histogram(predicted_flat, bins=bins, density=True)
    hist_actual, _ = np.histogram(actual_flat, bins=bins, density=True)

    # Normalize histograms to sum to 1 (Required for JSD to work correctly with probabilities)
    # The sum might be close to 1 due to density=True, but explicit normalization is safer.
    hist_predicted = hist_predicted / hist_predicted.sum()
    hist_actual = hist_actual / hist_actual.sum()

    # Calculate Jensen-Shannon Distance
    jsd = jensen_shannon_distance(hist_actual, hist_predicted)
    print(f"\nJensen-Shannon Distance between Actual vs. Predicted SIC for {patching_strategy_abbr} patchify {title_suffix}: {jsd:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(actual_flat, bins=bins, alpha=0.7, label='Actual Data', color='skyblue', density=True)
    plt.hist(predicted_flat, bins=bins, alpha=0.7, label='Predicted Data', color='salmon', density=True)
    plt.title(f'Actual vs. Predicted SIC for {patching_strategy_abbr} Patchify{title_suffix}\nJSD: {jsd:.4f}')
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