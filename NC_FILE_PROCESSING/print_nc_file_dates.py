import os
import argparse
import sys
import os

# Get the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Now you can import the modules
from nc_utility_functions import *
from config import *

def main():
    """ This requires a path to a file that is in the same directory. 
    TODO: MAKE THIS WORK ON ANY PATH ON PERLMUTTER
    """
    parser = argparse.ArgumentParser(description="Load data from a specified file.")
    parser.add_argument("path", help="Path to the file")
    args = parser.parse_args()

    print(f"Received file path: {args.path}")  # Debug print

    if args.path.endswith('.nc'):
        output = load_data(args.path)
        days = get_number_of_days(output)
    else:
        raise ValueError("Provided path is not an .nc file.")
        
    # Get list of all days / time values to plot that exist in one .nc file
    timeList = print_date_time(output, timeStringVariable=START_TIME_VARIABLE, days=days)

if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print("Error:", e)