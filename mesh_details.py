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

    latitudes_to_try = [40,41,42,43,44,45,46,47,48,50,55,60,65,70,75,80,85,89]

    print("How many cells would there be if I mask the data down to certain latitudes?")
    
    for latitude_limit in latitudes_to_try:
        indices = np.where(latCell >= latitude_limit)
        cellCount = len(indices[0])

        print("=========================================")
        print("Lat limit:         ", latitude_limit)
        print("Cell count:        ", cellCount)
        result = find_factors_optimized(cellCount)
        print(f"The factors of {cellCount} are: {result}")
        print("Is perfect square? ", is_perfect_square(cellCount))
        print("Divisible by 8       ", 8 in result)
        print("Divisible by 16      ", 16 in result)
        print("Divisible by 128     ", 128 in result)
        print("Divisible by 256     ", 256 in result)
        print("Divisible by 640     ", 640 in result)
        print("smallest index", indices[0][0])
        print("largest index", indices[0][-1])
    

if __name__ == "__main__":
    main()