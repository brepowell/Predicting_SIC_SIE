from NC_FILE_PROCESSING.nc_utility_functions import *
#from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
#from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
#from MAP_ANIMATION_GENERATION.map_animation_utility_functions import *
from config import *

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)

    print("Number of Cells", len(latCell))

    '''
    # Creates a map of the mesh all in one color (hot pink) so I can see the mesh
    fig, northMap = generate_axes_north_pole()
    map_lats_lons_one_color(fig, latCell, lonCell, northMap)

    
    fig, northMap = generate_axes_north_pole() 
    map_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap)

    # Creates a map of the mesh where larger indices are darker, so I can see how the mesh is indexed.
    fig, [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]  = generate_axes_all_projections()
    map_all_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap) 

    # Creates a map of the mesh where every 50k grid cells are color-coded so I can see how the mesh is indexed.
    fig, [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]  = generate_axes_all_projections()
    map_all_lats_lons_binned_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap) 
    '''

    # Two different patching techniques
    patches = cluster_patches(latCell, lonCell) # Patching in clusters, like k-means
    #patches = get_rows_of_patches(latCell, lonCell) # Patching in rows or stripes
    #patches = get_clusters_dbscan(latCell, lonCell) # Patching from dbscan
    #patches = get_clusters_kmeans_constrained(latCell, lonCell)

    # Load the labels
    #labels = np.load("patch_labels.npy")
    # Or if using CSV:
    # import pandas as pd
    # labels = pd.read_csv("cluster_labels.csv", header=None).values.flatten()
    
    # Insert into full label array
    #patches = np.full(latCell.shape, -1, dtype=int)
    #patches[latCell > 50] = labels

    # mask = latCell > 50
    # fig, northMap = generate_axes_north_pole()
    # map_patches_by_index(fig, latCell, lonCell, patches, northMap)
    
    # distinct_count = len(set(patches[mask]))
    # print("number of patches:" , distinct_count) # Output: 727

    # Check the number of cells per patch
    #from collections import Counter
    #counts = Counter(patches[mask])

    # What patch has the largest number of cells or least number of cells?
    # print(counts.most_common()[-1]) #gives the tuple with the smallest count.
    # print(counts.most_common(1)[0]) #gives the tuple with the largest count.

    #fig, northMap = generate_axes_north_pole()
    #map_patches_by_index_binned(fig, latCell, lonCell, patches, northMap)

    #animate_patch_reveal(latCell, lonCell, patches)
    
if __name__ == "__main__":
    main()