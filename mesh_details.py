from NC_FILE_PROCESSING.nc_utility_functions import *
from config import *
from perlmutterpath import *
import matplotlib.pyplot as plt

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)

    print("How many cells would there be if I mask the data down to certain latitudes?")

    indices = np.where(latCell > 50)
    print("Cell count when lat_limit is 50 degrees:", len(indices[0]))

    # indices = np.where(latCell > 65)
    # print("Cell count when lat_limit is 65 degrees:", len(indices[0])) 

    # indices = np.where(latCell > 80)
    # print("Cell count when lat_limit is 80 degrees:", len(indices[0]))
    
    labels_full = cluster_patches(latCell, lonCell)
    print(labels_full)

    indices = np.where(labels_full > -1)
    print("how many cells have labels", len(indices[0]))

    print("max index", labels_full.max())

    plt.hist(labels_full[indices])
    plt.savefig("histogram.png")
    plt.close()

if __name__ == "__main__":
    main()