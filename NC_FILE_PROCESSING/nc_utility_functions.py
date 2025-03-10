# Utility functions needed for .nc files
from config import *
import netCDF4                      # For opening .nc files for numpy
import numpy as np
from datetime import datetime, timedelta 

import os

#######################
# DIRECTORY FUNCTIONS #
#######################

def gather_files(useFullPath = True, path = FULL_PATH):
    """ Use the subdirectory specified in the config file. 
    Get all files in that folder. """
    filesToPlot = []
    print("Path to files is ", path)

    if useFullPath:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                if name.endswith('.nc'):
                    filesToPlot.append(os.path.join(root, name))

    else:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                if name.endswith('.nc'):
                    filesToPlot.append(name)

    print("Read this many files: ", len(filesToPlot))

    return filesToPlot

def check_if_nc_files_exist_in_directory(directory):

    if not os.path.isdir(directory):
        raise ValueError(f"Invalid directory: {directory}")

    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def check_if_Daily_or_Monthly(directory):
    """
    Check if all files in a directory contain either 'Daily' or 'Monthly' in their names.

    Args:
        directory (str): Path to the directory.

    Returns:
        str: "Daily" if all files contain "Daily", "Monthly" if all files contain "Monthly",
             False if there is a mix or neither.
    """

    files = check_if_nc_files_exist_in_directory(directory)
    
    if not files:
        print("No files found in the directory.")
        return False

    contains_daily = all("Daily" in f for f in files)
    contains_monthly = all("Monthly" in f for f in files)

    if contains_daily:
        return "Daily"
    elif contains_monthly:
        return "Monthly"
    else:
        return False  # Mixed or neither
    
#######################
#    LOADING FILES    #
#######################

def load_mesh(path_to_nc_file, mesh_file_name="", print_read_statement=True):
    """ Load the mesh from an .nc file. 
    The mesh must have the same resolution as the output file. 
    Return the latCell and lonCell variables. """

    if not os.path.isdir(path_to_nc_file):
        raise ValueError(f"Invalid directory: {path_to_nc_file}")
    
    if print_read_statement:
        print('======= Read Mesh: ', path_to_nc_file, mesh_file_name)

    dataset = netCDF4.Dataset(path_to_nc_file + mesh_file_name)
    latCell = np.degrees(dataset.variables['latCell'][:]) 
    lonCell = np.degrees(dataset.variables['lonCell'][:])

    return latCell, lonCell

def load_data(path_to_nc_file, output_file_name="", print_read_statement=True):
    """ Load the data from an .nc output file. """
    if not os.path.exists(path_to_nc_file):  # Check if file exists
        raise FileNotFoundError(f"File not found: {path_to_nc_file}")
    
    if print_read_statement:
        print('======= Read Output: ', path_to_nc_file)
    return netCDF4.Dataset(path_to_nc_file + output_file_name)

def load_all(path_to_nc_file, mesh_file_name, output_file_name):
    """ Load the mesh and data to plot. """

    files = check_if_nc_files_exist_in_directory(path_to_nc_file)

    if not files:
        print("No .nc files found in the directory.")
        raise FileNotFoundError(f"File not found: {path_to_nc_file}")

    latCell, lonCell    = load_mesh(path_to_nc_file, mesh_file_name)
    output              = load_data(path_to_nc_file, output_file_name) # TODO MAKE THIS DYNAMIC
    #days                = getNumberOfDays(output, keyVariableToPlot=VARIABLETOPLOT)

    # TODO: For the netCDF make this not hard coded
    days                = 1

    return latCell, lonCell, output, days

#######################
#   TIME VARIABLES    #
#######################

def get_number_of_days(output, keyVariableToPlot=VARIABLETOPLOT):
    """ Find out how many days are in the simulation by looking at the netCDF file 
    and at the variable you have chosen to plot. """

    count = output.variables[NC_TIME_COUNT_VARIABLE]
    print("Count is ", count.shape)

    variableForAllDays = output.variables[keyVariableToPlot][:]

    print("Shape is", variableForAllDays.shape)

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
