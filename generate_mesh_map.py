from NC_FILE_PROCESSING.nc_utility_functions import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
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
    #patches = cluster_patches(latCell, lonCell)
    patches = get_rows_of_patches(latCell, lonCell)

    # mask = latCell > 50
    # fig, northMap = generate_axes_north_pole()
    # map_patches_by_index(fig, latCell, lonCell, patches, northMap)
    
    # distinct_count = len(set(patches[mask]))
    # print("number of patches:" , distinct_count) # Output: 727

    # from collections import Counter
    
    # counts = Counter(patches[mask])
    # print(counts.most_common()[-1]) #gives the tuple with the smallest count.
    # print(counts.most_common(1)[0]) #gives the tuple with the largest count.

    fig, northMap = generate_axes_north_pole()
    map_patches_by_index_binned(fig, latCell, lonCell, patches, northMap)
    
    
if __name__ == "__main__":
    main()