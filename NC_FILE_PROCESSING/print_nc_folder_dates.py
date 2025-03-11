import os
import argparse
from nc_utility_functions import *
from config import *

def main():
    """ This requires a path to a file that is in the same directory. 
    TODO: MAKE THIS WORK ON ANY PATH ON PERLMUTTER
    """
    parser = argparse.ArgumentParser(description="Load nc files from a specified folder.")
    parser.add_argument("path", help="Path the folder (if current directory, use . ")
    args = parser.parse_args()

    print(f"Received file path: {args.path}")  # Debug print

    # This only retrieves files that end with ".nc"
    all_nc_files = gather_files(True, args.path)

    for file in all_nc_files:
        output = load_data("", file) # Use "" for files in current directory
        days = get_number_of_days(output)
        timeList = print_date_time(output, timeStringVariable=START_TIME_VARIABLE, days=days)
        print("file completed: ", file)
        
if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print("Error:", e)