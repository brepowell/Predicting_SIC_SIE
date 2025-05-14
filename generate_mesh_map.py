from NC_FILE_PROCESSING.nc_utility_functions import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from config import *

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)

    fig, northMap = generate_axes_north_pole() 
    map_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap)

    fig, northMap = generate_axes_north_pole()
    map_lats_lons_one_color(fig, latCell, lonCell, northMap)

    fig, [northMap, southMap, plateCar, rotPole]  = generate_axes_all_projections()
    map_all_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, rotPole)


if __name__ == "__main__":
    main()