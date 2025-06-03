# Utility functions needed for .nc files
from config import *
import netCDF4                      # For opening .nc files for numpy
import numpy as np
from datetime import datetime, timedelta 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import collections

import matplotlib.pyplot as plt

import os

#######################
# DIRECTORY FUNCTIONS #
#######################

def gather_files(useFullPath = True, path = FULL_PATH):
    """ Use the subdirectory specified in the config file. 
    Get all netCDF files in that folder. Return a list of .nc files sorted alphabetically. """
    all_nc_files_in_folder = []
    print("Path to files is ", path)

    if useFullPath:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                if name.endswith('.nc'):
                    all_nc_files_in_folder.append(os.path.join(root, name))

    else:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                if name.endswith('.nc'):
                    all_nc_files_in_folder.append(name)

    print("Read this many files: ", len(all_nc_files_in_folder))

    # Sort the list alphabetically before returning
    return sorted(all_nc_files_in_folder)

def validate_path(directory_or_full_path, file_name=""):
    """ Take in a directory or full path and a file name. Check if the path leads to a .nc file. Return the full path. """
    
    if directory_or_full_path == "":
        raise ValueError("The first parameter in load_mesh needs to be a directory or an .nc file")
    
    full_path = directory_or_full_path + file_name

    if not full_path.endswith('.nc'):
        raise ValueError("This path does not go to an .nc file")
    
    if not os.path.exists(full_path):  # Check if file exists
        raise FileNotFoundError(f"File not found: {full_path}")

    return full_path

#######################
#    LOADING FILES    #
#######################

def load_mesh(path_to_nc_file, mesh_file_name="", print_read_statement=True):
    """ Load the mesh from an .nc file. 
    The mesh must have the same resolution as the output file. 
    Return the latCell and lonCell variables. """

    full_path = validate_path(path_to_nc_file, mesh_file_name)
        
    if print_read_statement:
        print('======= Read Mesh: ', full_path)

    dataset = netCDF4.Dataset(full_path)
    latCell = np.degrees(dataset.variables['latCell'][:]) 
    lonCell = np.degrees(dataset.variables['lonCell'][:])

    return latCell, lonCell

def load_data(path_to_nc_file, output_file_name="", print_read_statement=True):
    """ Load the data from an .nc output file. """

    #TODO ADD A CHECK FOR ACCIDENTALLY HAVING SOMETHING OTHER THAN A STRING FOR THE 2ND PARAMETER
    
    full_path = validate_path(path_to_nc_file, output_file_name)
    
    if print_read_statement:
        print('======= Read Output: ', full_path)

    return netCDF4.Dataset(full_path)

#########################
#    FILE NAME UTILS    #
#########################

def get_date_string_from_file_name(path_to_nc_file):
    """Checks if a string is in the format YYYY-MM-DD.
    Assumes files are named with the format *YYYY-MM-DD.nc
    """
    date_string = path_to_nc_file[-13:-3]
    print(date_string)

    try:
        datetime.strptime(date_string, "%Y-%m-%d")
        return date_string
    except ValueError:
        return None

def is_valid_ymd(date_string):
    """
    Checks if a string is in the format YYYY-MM-DD and represents a valid date.

    Args:
        date_string: The string to check.

    Returns:
        True if the string is a valid date in YYYY-MM-DD format, False otherwise.
    """
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_string):
        return False
    try:
        datetime.datetime.strptime(date_string, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def get_year_from_file_name(path_to_nc_file):
    """Returns a string with the year.
    Assumes files are named with the format *YYYY-MM-DD.nc
    """
    date_string = get_date_string_from_file_name(path_to_nc_file)

    try:
        is_valid_ymd(date_string)
        return date_string[0:4]
    except ValueError:
        return None
    
def get_year_from_date_string(date_string):
    """Get the year. Assumes the format *YYYY-MM-DD.nc"""
    return date_string[0:4]

def get_month_from_date_string(date_string):
    """Get the month. Assumes the format *YYYY-MM-DD.nc"""
    return date_string[5:7]

def get_day_from_date_string(date_string):
    """Get the day. Assumes the format *YYYY-MM-DD.nc"""
    return date_string[8:10]

#######################
#   TIME VARIABLES    #
#######################

def get_number_of_days(output, keyVariableToPlot=VARIABLETOPLOT):
    """ Find out how many days are in the simulation by looking at the netCDF file 
    and at the variable you have chosen to plot. """

    count = output.variables[NC_TIME_COUNT_VARIABLE]
    print("Count of days is ", count.shape)

    variableForAllDays = output.variables[keyVariableToPlot][:]
    print("Shape of variable to plot is", variableForAllDays.shape)

    return variableForAllDays.shape[0]

def print_date_time(output, timeStringVariable = TIMESTRINGVARIABLE, days = 1):
    """ Prints and returns the date from the .nc file's time string variable. 
    This assumes that the time needs to be decoded and is the format
    [b'0' b'0' b'0' b'1' b'-' b'0' b'1' b'-' b'0' b'2' b'_' b'0' b'0' b':' b'0' b'0' b':' b'0' b'0']
    """

    # Get all the time variables
    rawTime = output.variables[timeStringVariable][:days]
    rawTime = rawTime.ravel()
    rawTime = bytearray(rawTime).decode()
    timeStrings = rawTime.split('\x00')
    timeStrings[:] = [x for x in timeStrings if x]
    
    if len(timeStrings) == 1:
        print(timeStrings[0])
        return timeStrings[0]
    
    print(timeStrings)
    return timeStrings

def convert_date_bytes_to_string(bytesTime):
    """ Prints and returns the date from a byte string. 
    This assumes that the time needs to be decoded and is the format
    [b'0' b'0' b'0' b'1' b'-' b'0' b'1' b'-' b'0' b'2' b'_' b'0' b'0' b':' b'0' b'0' b':' b'0' b'0']
    """
    bytesTime = bytearray(bytesTime).decode()
    timeStrings = bytesTime.split('\x00')
    timeStrings[:] = [x for x in timeStrings if x]
    
    if len(timeStrings) == 1:
        print(timeStrings[0])
        return timeStrings[0]
    
    print(timeStrings)
    return timeStrings

def convert_time(timeToConvert):
    """ Convert time from proleptic_gregorian to a human-readable string."""
    base_date = datetime(2000, 1, 1)
    d = base_date + timedelta(hours=timeToConvert)
    timeString = d.strftime("%Y-%m-%d %H:%M:%S")
    print("Time converted", timeString)
    return timeString

def get_time_array_from_start_time(output, length):
    """ Pull the starting timestamp from the .nc file. 
    Populate an array with times. (These are approximate, not real). """
    start = float(output.variables["time"][:1])
    stop = output.variables[TIMEVARIABLE][:1] + length
    step = .00036 # How much time elapses between pulses (there are 10,000 pulses per second)
    return np.arange(start, stop, step)

###########
#  PRINT  #
###########

def print_all_available_nc_variables(output):
    """ See what variables you can use in this netCDF file. 
    Requires having loaded a netCDF file into an output variable. 
    This is an alternative to the ncdump command. 
    It's useful because it displays the shape of each variable as well."""
    print(output.variables) # See all variables available in the netCDF file

#############################
#  OTHER NETCDF VARIABLES   #
#############################

def get_cell_indices(output, cellVariable = CELLVARIABLE):
    """ Get only the indices that correspond to the E3SM mesh. """
    indices = output.variables[cellVariable][:1]
    return indices.ravel()

def get_Lat_Lon(output):
    """ Pull the latitude and longitude variables from an .nc file. """
    latCell = output.variables[LATITUDEVARIABLE][:1]
    latCell = latCell.ravel()
    lonCell = output.variables[LONGITUDEVARIABLE][:1]
    lonCell = lonCell.ravel()
    return latCell, lonCell


def cluster_patches(latCell, lonCell):
    """ Take lat and lon values from only 50 degrees north and cluster them into patches of 49
    cells per patch for 727 patches and 35623 mesh cells. Returns an array of all 465,044 mesh cells
    where -1 indicates that there was no cluster. """
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > 50                     # Boolean array, True for 35623 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask))
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f))  # shape (35623, 2)
    
    k = 727                                   # 35623 ÷ 49
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",     # default, good for speed / accuracy
        n_init="auto",        # auto-scales n_init in recent scikit-learn versions
        random_state=42,      # make runs reproducible
        algorithm="elkan"     # faster for low-dimension dense data
    ).fit(coords)
    
    centroids = kmeans.cluster_centers_       # array shape (727, 2)
    print("centroids", centroids.shape)
    labels_f = kmeans.labels_                 # length 35623
    print("labels", len(labels_f))
    
    # --- 3.  Re-insert labels into full-length array ---------------
    # Make an array filled with -1's in the shape of latCell
    labels_full = np.full(latCell.shape, -1, dtype=int)  # –1 marks “not clustered”
    labels_full[mask] = labels_f # Populate the array with the labels from the clustering

    print("Cluster sizes:", collections.Counter(labels_full[mask]))
    
    #plt.hist(labels_full[mask])
    counts, bins = np.histogram(labels_f)
    #plt.stairs(counts, bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig("histogram_kmeans_patches.png")
    plt.close()
    
    return labels_full


def get_rows_of_patches(latCell, lonCell):
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    indices = np.where(latCell > LAT_LIMIT)
    print("Non-zeroes", np.count_nonzero(indices[0]))

    labels_full = np.full(latCell.shape, -1, dtype=int)

    bucket = 0

    for i, index in enumerate(indices[0]):
        labels_full[index] = bucket
        if (i+1) % 49 == 0 and i > 0:
            bucket += 1

    print("Cluster sizes:", collections.Counter(labels_full[indices]))
    
    plt.hist(labels_full[indices])
    plt.savefig("histogram_patch_rows.png")
    plt.close()
    
    return labels_full

def get_clusters_dbscan(latCell, lonCell):

    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > 50                     # Boolean array, True for 35623 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask)) # prints 35623
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f)) # shape (35623, 2)
    print("Shape of coords", coords.shape)

    # make an elbow plot to see the best value for eps
    
    neighbors = NearestNeighbors(n_neighbors=49, algorithm='ball_tree', metric='haversine')
    neighbors_fit = neighbors.fit(coords)
    distances, indices = neighbors_fit.kneighbors(coords)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 48]  # index 48 = 49th nearest
    plt.plot(distances)
    plt.savefig("elbow_plot.png")
    plt.close()

    # The elbow plot has an elbow around 0.022
    
    # DBSCAN parameters
    
    eps = 0.022  # Epsilon (radius)
    min_samples = 20  # MinPts (minimum points within the radius)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    dbscan.fit(coords)

    # Get cluster labels
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Estimated number of clusters:", n_clusters)
    print("Estimated number of noise points:", n_noise)
    print("Cluster sizes:", collections.Counter(labels))
    
    # --- 3.  Re-insert labels into full-length array ---------------
    # Make an array filled with -1's in the shape of latCell
    labels_full = np.full(latCell.shape, -1, dtype=int)  # –1 marks “not clustered”
    labels_full[mask] = labels # Populate the array with the labels from the clustering

    print(len(labels_full[mask]))

    plt.hist(labels_full[mask])
    plt.savefig("histogram_dbscan_patches.png")
    plt.close()

    return labels_full

def get_clusters_kmeans_constrained(latCell, lonCell):
    """
    This method is special -- I did not want to mix it up with my regular environment, 
    so I'm running it in its own environment and then saving the results to use with the main environment.
    This algorithm takes a long time. Run it as a batch job in the future!
    
    conda create -n env_special_kmeans
    conda activate env_special_kmeans
    conda install pip
    conda install netCDF4
    conda install scikit-learn
    
    pip install k_means_constrained
    python cluster_k_means_constrained.py

    """

    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > 50                     # Boolean array, True for 35623 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask))
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f))  # shape (35623, 2)

    from k_means_constrained import KMeansConstrained
    
    kmeans = KMeansConstrained(
        n_clusters=727,
        size_min=49,
        size_max=49
    ).fit(coords)

    # Save labels to a file
    labels_f = kmeans.labels_
    print("labels", len(labels_f))
    np.save("patch_labels.npy", labels_f)
    print("== saved patches ==")

    print("Cluster sizes:", collections.Counter(labels_f))
    counts, bins = np.histogram(labels_f)
    plt.hist(bins[:-1], bins, weights=counts)
          
    plt.savefig("histogram_kmeans_constrained_patches.png")
    plt.close()
    
#######################
#  REDUCE DIMENSIONS  #
#######################

def reduce_to_one_dimension(output, keyVariableToPlot=VARIABLETOPLOT, dayNumber=0):
    """ Reduce the variable to one day's worth of data so we can plot 
    using each index per cell. The indices for each cell of the 
    variableToPlot1Day array coincide with the indices 
    of the latCell and lonCell. """
    
    variableForAllDays = output.variables[keyVariableToPlot][:]

    # Check if the variable is one-dimensional
    if variableForAllDays.ndim != 1:
        return variableForAllDays[dayNumber,:]
    else:
        return variableForAllDays[:]

###################
#  DOWNSAMPLING   #
###################

def downsample_data(latCell, lonCell, timeCell, variableToPlot1Day, factor=DEFAULT_DOWNSAMPLE_FACTOR):
    """ Downsample the data arrays by the given factor. """
    return latCell[::factor], lonCell[::factor], timeCell[::factor], variableToPlot1Day[::factor]

def downsample_data(variable, factor=DEFAULT_DOWNSAMPLE_FACTOR):
    """ Downsample the data arrays by the given factor. """
    return variable[::factor]
