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
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)
    print("nCells:             ", len(latCell))
    
    latitude_threshold = 40
    print("latitude_threshold: ", latitude_threshold)
    
    mask = latCell >= latitude_threshold
    masked_ncells_size = np.count_nonzero(mask)
    print("mask size:          ", masked_ncells_size)
    
    cells_per_patch = 256
    n_patches = masked_ncells_size // cells_per_patch

    print("cells_per_patch:    ", cells_per_patch)
    print("n_patches:          ", n_patches)

    print("=== SET INITIAL VARIABLES === ")
    
    # Different patching techniques
    # labels_full, patches, patch_latlons, algorithm = cluster_patches_kmeans(latCell, lonCell, latitude_threshold) # Patching in clusters, like k-means
    # labels_full, patches, patch_latlons, algorithm = get_rows_of_patches(latCell, lonCell, latitude_threshold) # Patching in rows or stripes
    # labels_full, patches, patch_latlons, algorithm = get_clusters_dbscan(latCell, lonCell, latitude_threshold, cells_per_patch)
    # labels_full, patches, patch_latlons, algorithm = compute_knn_patches(latCell, lonCell, latitude_threshold, cells_per_patch, n_patches, seed=42)
    # labels_full, patches, patch_latlons, algorithm = compute_disjoint_knn_patches(latCell, lonCell, latitude_threshold, cells_per_patch, n_patches, seed=42)
    # labels_full, patches, patch_latlons, algorithm = compute_agglomerative_patches(latCell, lonCell, latitude_threshold, n_patches)
    # labels_full, patches, patch_latlons, algorithm = patchify_by_latitude(latCell, lonCell, patch_size=cells_per_patch, min_lat=latitude_threshold, max_lat=90, step_deg=3, seed=42)
    # labels_full, patches, patch_latlons, algorithm = patchify_by_latitude_simple(latCell, patch_size=cells_per_patch, min_lat=latitude_threshold, max_lat=90, step_deg=3, seed=42)    
    #labels_full, patches, patch_latlons, algorithm = patchify_with_spillover(latCell, patch_size=cells_per_patch, min_lat=latitude_threshold, max_lat=90, step_deg=3, max_patches=n_patches, seed=42)
    # labels_full, patches, patch_latlons, algorithm = patchify_by_latlon_spillover(latCell, lonCell, k=cells_per_patch, max_patches=n_patches, latitude_threshold=latitude_threshold)
    # labels_full, patches, patch_latlons, algorithm = patchify_by_lon_spilldown(latCell, lonCell, k=cells_per_patch, max_patches=n_patches, latitude_threshold=latitude_threshold)
    #labels_full, patches, patch_latlons, algorithm = patchify_staggered_polar_descent(latCell, lonCell, k=cells_per_patch, max_patches=n_patches, latitude_threshold=latitude_threshold)

    # Breadth-first or spectral clustering algorithms based on cellsOnCell adjacency lists
    cellsOnCell = np.load(f'cellsOnCell.npy')
    #labels_full, patches, patch_latlons, algorithm = build_patches_from_seeds(cellsOnCell, latCell, lonCell, n_patches=n_patches, patch_size=cells_per_patch, seed=42, mask=mask)
    #labels_full, patches, patch_latlons, algorithm = build_patches_from_seeds_improved(cellsOnCell, n_patches=n_patches, patch_size=cells_per_patch, seed=42, mask=mask, pad_to_exact_size=False)
    labels_full, patches, patch_latlons, algorithm = build_patches_from_seeds_priority(cellsOnCell, latCell, lonCell, mask=mask)
    
    # --- options for visual
    # Gradient Patches
    # fig, northMap = generate_axes_north_pole()
    # map_patches_by_index(fig, latCell, lonCell, labels_full, northMap)

    # Multicolored Patches
    fig, northMap = generate_axes_north_pole()
    map_patches_by_index_binned(fig,
                                latCell, lonCell, labels_full,   # 1-D numpy arrays
                                northMap,
                                n_patches, cells_per_patch,
                                algorithm = algorithm, 
                                color_map="flag", 
                                )

    animate_patch_reveal(latCell, lonCell, labels_full, gif_path=f"patch_animation_{algorithm}.gif")
    
if __name__ == "__main__":
    main()