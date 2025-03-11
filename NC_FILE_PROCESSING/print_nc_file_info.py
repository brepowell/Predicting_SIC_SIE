import os
import argparse
from nc_utility_functions import *

def main():
    """ This requires a path to a file that is in the same directory. """
    
    parser = argparse.ArgumentParser(description="Load data from a specified file.")
    parser.add_argument("path", help="Path to the file")
    args = parser.parse_args()

    print(f"Received file path: {args.path}")  # Debug print

    if args.path.endswith('.nc'):
        output = load_data(args.path)
    else:
        raise ValueError("Provided path is not an .nc file.")

    print("Called Load Data")
    
    print_all_available_nc_variables(output)

if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print("Error:", e)