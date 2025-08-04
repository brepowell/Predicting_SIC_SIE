from NC_FILE_PROCESSING.patchify_utils import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from MAP_ANIMATION_GENERATION.map_animation_utility_functions import * 
from config import *

def plot_mesh_indices(latCell, lonCell):
    
    fig, northMap = generate_axes_north_pole() 
    map_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap)

    # Creates a map of the mesh where larger indices are darker, so I can see how the mesh is indexed.
    # This doesn't work as well as the next one below.
    fig, [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]  = generate_axes_all_projections()
    map_all_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, 
                                        orthoMap, robinMap, rotPolMap) 

    # Creates a map of the mesh where every 50k grid cells are color-coded so I can see how the mesh is indexed.
    # This is the better option to see how the mesh indexes on the globe
    fig, [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]  = generate_axes_all_projections()
    map_all_lats_lons_binned_by_index(fig, latCell, lonCell, northMap, southMap, 
                                    plateCar, orthoMap, robinMap, rotPolMap) 

def plot_land_ice():
    landIceMask = np.load("landIceMask.npy")
    landIceMask = landIceMask[0]

    print("Land mask Size", len(landIceMask))
    print(f"Land mask non-zero{np.count_nonzero(landIceMask)}")
    
    fig, northMap, southMap = generate_axes_north_and_south_pole()
    generate_maps_north_and_south(fig, northMap, southMap, latCell, lonCell, landIceMask, "mesh_land_ice_mask")

def plot_mesh(latCell, lonCell, mask):
    
    # Creates a map of the mesh all in one color (hot pink) so I can see the mesh
    fig, northMap = generate_axes_north_pole()
    map_lats_lons_one_color(fig, latCell[mask], lonCell[mask], northMap)

def main():
    
    # --- Define Patchify Functions Dictionary ---
    PATCHIFY_FUNCTIONS = {
        "latlon_spillover": patchify_by_latlon_spillover,
        #"staggered_polar_descent": patchify_staggered_polar_descent,
        "lon_spilldown": patchify_by_lon_spilldown,
        "latitude_spillover_redo": patchify_with_spillover,
        #"latitude_simple": patchify_by_latitude_simple,
        #"latitude_neighbors": patchify_by_latitude,
        #"breadth_first_improved_padded": build_patches_from_seeds_improved_padded,
        #"breadth_first_bfs_basic": build_patches_from_seeds_bfs_basic,
        #"agglomerative": compute_agglomerative_patches,
        #"knn_disjoint": compute_disjoint_knn_patches,
        #"knn_basic": compute_knn_patches,
        #"kmeans": cluster_patches_kmeans,
        "rows": get_rows_of_patches,
        #"dbscan": get_clusters_dbscan
    }

    cellsOnCell = np.load(f'cellsOnCell.npy')
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)
    print("nCells:             ", len(latCell))
    
    latitude_threshold = 40
    print("latitude_threshold: ", latitude_threshold)
    
    mask = latCell >= latitude_threshold
    masked_ncells_size = np.count_nonzero(mask)
    print("mask size:          ", masked_ncells_size)
    
    cells_per_patch = 256
    num_patches = masked_ncells_size // cells_per_patch

    print("cells_per_patch:    ", cells_per_patch)
    print("num_patches:          ", num_patches)

    print("=== SET INITIAL VARIABLES === ")
    
    # --- Common Parameters for all functions ---
    common_params = {
        "latCell": latCell,
        "lonCell": lonCell,
        "cells_per_patch": cells_per_patch, 
        "num_patches": num_patches,
        "latitude_threshold": latitude_threshold,
        "seed": 42
    }

    # --- Function-specific Parameters (if any) ---
    specific_params = {
        "latitude_spillover_redo": {"step_deg": 5, "max_lat": 90},
        "latitude_simple": {"step_deg": 5, "max_lat": 90},
        "latitude_neighbors": {"step_deg": 5, "max_lat": 90},
        "breadth_first_improved_padded": {"cellsOnCell": cellsOnCell, "pad_to_exact_size": True},
        "breadth_first_bfs_basic": {"cellsOnCell": cellsOnCell},
        "agglomerative": {"n_neighbors": 5},
    }

    # --- Run Tests ---
    for name, func in PATCHIFY_FUNCTIONS.items():
        print(f"\n--- Testing '{name}' ---")
        params = common_params.copy()
        params.update(specific_params.get(name, {})) # Add function-specific params

        try:

            #     Each patchify function returns 3 things:
            # full_nCells_patch_ids : np.ndarray
            #     Array of shape (nCells,) giving patch ID or -1 if unassigned.
            # indices_per_patch_id : List[np.ndarray]
            #     List of patches, each a list of cell indices (np.ndarray of ints) that correspond with nCells array.
            # patch_latlon : np.ndarray
            #     Array of shape (n_patches, 2) containing (latitude, longitude) for one
            #     representative cell per patch (the first cell added to the patch)
            full_nCells_patch_ids, cell_indices_per_patch_id, patch_latlon, algorithm_name = func(**params)
            print(f"  SUCCESS: {algorithm_name} produced {len(cell_indices_per_patch_id)} patches.")
            print(f"  First patch indices: {cell_indices_per_patch_id[0] if cell_indices_per_patch_id else 'N/A'}")
            print(f"  First patch lat/lon: {patch_latlon[0] if patch_latlon.shape[0] > 0 else 'N/A'}")
            print(f"  Labels full shape: {full_nCells_patch_ids.shape}")

            DOT_SIZE = 1
            HIGHLIGHT_SIZE = DOT_SIZE * 50
            fig, northMap = generate_axes_north_pole()
            map_patches_by_index_binned_select_patches(fig, latCell, lonCell, full_nCells_patch_ids, 
                                                       patch_latlon, northMap,
                                                       cells_per_patch=cells_per_patch,
                                                       dot_size=DOT_SIZE, algorithm=algorithm_name,
                                                       color_map="flag", plot_every_n_patches=10,
                                                       highlight_cell_index=0, highlight_color='yellow',
                                                       highlight_size=HIGHLIGHT_SIZE)

            #Multicolored Patches
            # fig, northMap = generate_axes_north_pole()
            # map_patches_by_index_binned(fig,
            #                             latCell, lonCell, full_nCells_patch_ids,   # 1-D numpy arrays
            #                             northMap,
            #                             num_patches, cells_per_patch,
            #                             algorithm = algorithm_name, 
            #                             color_map="flag", 
            #                     )
            
            # animate_patch_reveal(latCell, lonCell, full_nCells_patch_ids, gif_path=f"patch_animation_{algorithm_name}.gif")

        except Exception as e:
            print(f"  FAILURE: {name} failed with error: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

    print("\n--- All Patchify Function Tests Completed ---")
    
if __name__ == "__main__":
    main()