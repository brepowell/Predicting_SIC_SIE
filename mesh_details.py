from NC_FILE_PROCESSING.nc_utility_functions import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from config import *

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh()

    indices = np.where(latCell > 50)
    print("Cell count when lat_limit is 50 degrees:", len(indices))

    indices = np.where(latCell > 65)
    print("Cell count when lat_limit is 65 degrees:", len(indices)) 

    indices = np.where(latCell > 80)
    print("Cell count when lat_limit is 80 degrees:", len(indices))

    

if __name__ == "__main__":
    main()