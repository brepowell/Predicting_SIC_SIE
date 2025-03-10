import os
import argparse
from NC_FILE_PROCESSING.nc_utility_functions import *

def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load data from a specified directory or file.")
    parser.add_argument("path", help="Path to the directory or file")

    args = parser.parse_args()

    # Determine whether the argument is a directory or a file
    if os.path.isdir(args.path):
        runDir = args.path
        
        #TODO - MAKE THIS ITERATE THROUGH ALL FILES IN A FOLDER
        output = loadData(runDir, fileName)
        days = getNumberOfDays(output)

    elif os.path.isfile(args.path):
        runDir, fileName = os.path.split(args.path)
        output = loadData(runDir, fileName)
        days = getNumberOfDays(output)
    else:
        raise ValueError("Provided path is neither a valid file nor a directory.")

    # Get list of all days / time values to plot that exist in one .nc file
    timeList = printDateTime(output, timeStringVariable=START_TIME_VARIABLE, days=days)

if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print("Error:", e)