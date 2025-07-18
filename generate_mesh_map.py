from NC_FILE_PROCESSING.patchify_utils import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from MAP_ANIMATION_GENERATION.map_animation_utility_functions import * 
from config import *

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)

    print("Number of Cells", len(latCell))
    
    '''
    labels_full = np.load("labels_full.npy")
    mask = labels_full == -1
    
    # Creates a map of the mesh all in one color (hot pink) so I can see the mesh
    fig, northMap = generate_axes_north_pole()
    map_lats_lons_one_color(fig, latCell[mask], lonCell[mask], northMap)

    
    fig, northMap = generate_axes_north_pole() 
    map_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap)

    # Creates a map of the mesh where larger indices are darker, so I can see how the mesh is indexed.
    fig, [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]  = generate_axes_all_projections()
    map_all_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, 
                                        orthoMap, robinMap, rotPolMap) 

    # Creates a map of the mesh where every 50k grid cells are color-coded so I can see how the mesh is indexed.
    fig, [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]  = generate_axes_all_projections()
    map_all_lats_lons_binned_by_index(fig, latCell, lonCell, northMap, southMap, 
                                    plateCar, orthoMap, robinMap, rotPolMap) 
    '''

    lat_threshold = 50
    n_patches = 140
    cells_per_patch = 256
    
    # Different patching techniques
    #labels_full = cluster_patches_kmeans(latCell, lonCell, lat_threshold) # Patching in clusters, like k-means
    #labels_full = get_rows_of_patches(latCell, lonCell, lat_threshold) # Patching in rows or stripes
    #labels_full = get_clusters_dbscan(latCell, lonCell, lat_threshold, cells_per_patch)
    #labels_full = compute_knn_patches(latCell, lonCell, lat_threshold, cells_per_patch, n_patches, seed=42)
    #labels_full = compute_disjoint_knn_patches(latCell, lonCell, lat_threshold, cells_per_patch, n_patches, seed=42)
    #labels_full = compute_agglomerative_patches(latCell, lonCell, lat_threshold, n_patches)
    #labels_full = patchify_by_latitude(latCell, lonCell, patch_size=cells_per_patch, 
    #                            min_lat=lat_threshold, max_lat=90, step_deg=3, seed=42)
    # labels_full = patchify_by_latitude_simple(latCell, patch_size=cells_per_patch, 
    #                                       min_lat=lat_threshold, max_lat=90, step_deg=3, seed=42)

    #cellsOnCell = np.load("cellsOnCell.npy")
    
    # labels_full = patchify_with_spillover(latCell, patch_size=cells_per_patch, 
    #                         min_lat=lat_threshold, max_lat=90, 
    #                         step_deg=3, max_patches=n_patches, seed=42)

    #labels_full, patches, patch_latlons = patchify_by_latlon_spillover(latCell, lonCell, k=cells_per_patch, max_patches=n_patches, lat_threshold=lat_threshold)

    labels_full, patches, patch_latlons = patchify_by_lon_spilldown(latCell, lonCell, k=cells_per_patch, max_patches=n_patches, lat_threshold=lat_threshold)

    #labels_full, patches, patch_latlons = patchify_staggered_polar_descent(latCell, lonCell, k=cells_per_patch, max_patches=n_patches, lat_threshold=lat_threshold)

    # ----------- breadth-first method ------
    # mesh = xr.open_dataset("NC_FILE_PROCESSING/mpassi.IcoswISC30E3r5.20231120.nc")

    # cellsOnCell = mesh["cellsOnCell"].values
    # cellsOnCell[cellsOnCell == 0] = -1  # reinterpret 0s as missing
    # cellsOnCell -= 1  # Convert from 1-based to 0-based indexing (MPAS mesh convention)
    # cellsOnCell = mesh["cellsOnCell"].values - 1  # Convert from 1-based to 0-based index
    
    # valid_mask = np.degrees(mesh["latCell"].values) > lat_threshold # Optional mask (e.g. exclude southern hemisphere)
    
    # labels_full = build_patches_from_seeds(cellsOnCell, patch_size=49, n_patches=727, valid_mask=valid_mask)

    # np.save('patches_using_breadth_first.npy', labels_full) 

    # ----------------------------------
    # Load the labels
    # labels_full = np.load("patches_using_breadth_first.npy")

    # fig, northMap = generate_axes_north_pole()
    # map_patches_by_index(fig, latCell, lonCell, labels_full, northMap)

    fig, northMap = generate_axes_north_pole()
    map_patches_by_index_binned(fig,
                                latCell, lonCell, labels_full,   # 1-D numpy arrays
                                northMap,
                                n_patches, cells_per_patch,
                                algorithm = "lon_spilldown", 
                                color_map="flag", 
                                )

    animate_patch_reveal(latCell, lonCell, labels_full, gif_path="patch_animation_lon_spilldown")
    
if __name__ == "__main__":
    main()