from NC_FILE_PROCESSING.nc_utility_functions import *
from config import *
from perlmutterpath import *

import math

def find_factors_optimized(number):
    factors = set()
    for i in range(1, int(math.sqrt(number)) + 1):
        if number % i == 0:
            factors.add(i)
            factors.add(number // i)
    return sorted(list(factors))

def is_perfect_square(n):
    if n < 0:
        return False
    root = math.sqrt(n)
    return root.is_integer()

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)

    print("How many cells would there be if I mask the data down to certain latitudes?")

    indices = np.where(latCell > 50)
    cellCount = len(indices[0])
    print("Cell count when lat_limit is 50 degrees:", cellCount)
    result = find_factors_optimized(cellCount)
    print(f"The factors of {cellCount} are: {result}")
    
    # indices = np.where(latCell > 65)
    # print("Cell count when lat_limit is 65 degrees:", len(indices[0])) 

    # indices = np.where(latCell > 80)
    # print("Cell count when lat_limit is 80 degrees:", len(indices[0]))
    
    labels_full = cluster_patches(latCell, lonCell)
    #print(labels_full)

    indices = np.where(labels_full > -1)
    print("how many cells have labels", len(indices[0]))

    print("max index", labels_full.max())

    indices = np.where(labels_full == 700)
    print("Latitudes in patch 700")
    print(latCell[indices])
    print("Longitudes in patch 700")
    print(lonCell[indices])
    print("How many cells are in patch 700", len(indices[0]))

    print("lat min", latCell[indices].min())
    print("lat max", latCell[indices].max())
    print("lon min", lonCell[indices].min())
    print("lon max", lonCell[indices].max())

    diameter_latitudes  = np.radians(latCell[indices].max()) - np.radians(latCell[indices].min())
    diameter_longitudes = np.radians(lonCell[indices].max()) - np.radians(lonCell[indices].min())

    print("radius for lats", diameter_latitudes / 2)
    print("radius for lons", diameter_longitudes / 2)

    
    


    

if __name__ == "__main__":
    main()