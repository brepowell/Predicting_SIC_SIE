from NC_FILE_PROCESSING.nc_utility_functions import *
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.neighbors import kneighbors_graph

import numpy as np

def latlon_to_xyz(lat, lon):
    """Convert lat/lon (in degrees) to 3D unit sphere coordinates"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=1)  # shape (nCells, 3)

def bar_graph_cluster_distribution(labels_full, mask, algorithm = "patches"):
    counts = collections.Counter(labels_full[mask])
    print("Cluster sizes:")
    
    patch_ids = list(counts.keys())
    cells_per_patch = list(counts.values())

    print("min size", min(cells_per_patch))
    print("max size", max(cells_per_patch))

    # What patch has the largest number of cells or least number of cells?
    print("smallest count", counts.most_common()[-1]) #gives the tuple with the smallest count.
    print("max count", counts.most_common(1)[0]) #gives the tuple with the largest count.

    distinct_count = len(set(labels_full[mask]))
    print("number of patches:" , distinct_count) # Output: 727

    plt.bar(patch_ids, cells_per_patch)
    plt.ylim(0, 100)  # Sets the y-axis from 0 to 500
    plt.xlabel("Patch ID's")
    plt.ylabel("Cell Count")
    plt.title("Cells per Patch")
    plt.savefig(f"distribution_{algorithm}_patches_{distinct_count}.png")
    plt.close()

import numpy as np

def patchify_by_latlon_spillover(latCell, lonCell, k=49, max_patches=727, lat_threshold=40.0):
    """
    Create spatially coherent patches based on sorted lat/lon without clustering.
    Walks from high latitude to low, grouping cells in longitude stripes and spilling over as needed.

    Parameters
    ----------
    latCell : np.ndarray
        1D array of cell latitudes.
    lonCell : np.ndarray
        1D array of cell longitudes.
    k : int
        Approximate number of cells per patch.
    max_patches : int
        Maximum number of patches to form.
    lat_threshold : float
        Don't form patches below this latitude.

    Returns
    -------
    patches : List[np.ndarray]
        List of patches, each a list of cell indices (np.ndarray of ints).
    labels_full : np.ndarray
        Array of shape (nCells,) giving patch ID or -1 if unassigned.
    """

    nCells = len(latCell)
    used = np.zeros(nCells, dtype=bool)

    # Filter valid candidates
    mask = latCell >= lat_threshold
    indices = np.where(mask)[0]

    # Sort by latitude (descending), then by longitude (ascending)
    # lexsort returns indices from the original arrays (ex. 0 through 465044)
    sort_order = np.lexsort((lonCell[indices], -latCell[indices]))
    sorted_indices = indices[sort_order]

    patches = []
    labels_full = np.full(nCells, -1, dtype=int)
    patch_id = 0
    current_patch = []

    for idx in sorted_indices:
        if used[idx]:
            continue

        current_patch.append(idx)
        used[idx] = True

        if len(current_patch) == k:
            patches.append(np.array(current_patch, dtype=int))
            labels_full[current_patch] = patch_id
            patch_id += 1
            current_patch = []

            if patch_id >= max_patches:
                break

    # Optionally add a final partial patch
    if current_patch and patch_id < max_patches:
        patches.append(np.array(current_patch, dtype=int))
        labels_full[current_patch] = patch_id

    print(f"Built {len(patches)} patches of size ~{k}")
    
    # Flatten and relabel for visualization
    labels_full = np.full(len(latCell), -1)
    for i, inds in enumerate(patches):
        labels_full[inds] = i
    
    bar_graph_cluster_distribution(labels_full, mask=np.ones_like(latCell, dtype=bool), algorithm="latlon_spillover")

    #return patches
    return labels_full



def patchify_with_spillover(latCell, patch_size=49, 
                            min_lat=40, max_lat=90, 
                            step_deg=3, max_patches=None, seed=42):
    """
    Sweep latitudinal bands and form spatially cohesive patches, 
    letting underfilled patches spill into the next band.
    """
    rng = np.random.default_rng(seed)
    latCell = np.array(latCell)
    
    patch_indices = []
    used = np.zeros(len(latCell), dtype=bool)
    carryover = []

    for band_max in np.arange(max_lat, min_lat, -step_deg):
        band_min = band_max - step_deg
        band_mask = (latCell >= band_min) & (latCell < band_max) & (~used)
        band_ids = np.where(band_mask)[0]
        
        # Combine with carryover from the previous band
        candidates = np.concatenate([carryover, band_ids]).astype(int)
        rng.shuffle(candidates)
        
        num_full = len(candidates) // patch_size
        for i in range(num_full):
            patch = candidates[i*patch_size : (i+1)*patch_size]
            patch_indices.append(patch)
            used[patch] = True
        
        # Save leftovers for the next band
        leftover = candidates[num_full * patch_size:]
        carryover = leftover

        if max_patches is not None and len(patch_indices) >= max_patches:
            break

    # Flatten and relabel for visualization
    labels_full = np.full(len(latCell), -1)
    for i, inds in enumerate(patch_indices):
        labels_full[inds] = i
    
    bar_graph_cluster_distribution(labels_full, mask=np.ones_like(latCell, dtype=bool), algorithm="latitude_spillover_redo")

    # avg latitude of each patch
    avg_lats = np.array([latCell[patch].mean() for patch in patch_indices])
    
    # Scatter plot of patch_id vs avg_latitude
    plt.scatter(range(len(avg_lats)), avg_lats)
    plt.xlabel("Patch ID")
    plt.ylabel("Average Latitude")
    plt.title("Are Patch IDs ordered by latitude?")
    plt.savefig("patch_ids_by_avg_lat.png")
    plt.close()
    print("saved_fig")
    
    #return patch_indices
    return labels_full


import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np

def patchify_by_latitude_simple(latCell, patch_size=49, 
                                min_lat=40, max_lat=90, 
                                step_deg=3, seed=42):
    """
    Divide cells into patches by sweeping latitude bands north → south.
    No clustering used—just chunking within each band.
    
    Parameters:
    - latCell: np.ndarray of cell latitudes
    - patch_size: desired number of cells per patch
    - min_lat, max_lat: bounds for latitude (degrees)
    - step_deg: width of latitude bands
    """
    rng = np.random.default_rng(seed)
    latCell = np.array(latCell)
    
    patch_indices = []
    used = np.zeros(len(latCell), dtype=bool)

    for band_max in np.arange(max_lat, min_lat, -step_deg):
        band_min = band_max - step_deg
        band_mask = (latCell >= band_min) & (latCell < band_max) & (~used)
        band_ids = np.where(band_mask)[0]

        if len(band_ids) < patch_size:
            continue  # skip too-small bands

        rng.shuffle(band_ids)  # randomize order

        for i in range(0, len(band_ids) - patch_size + 1, patch_size):
            patch = band_ids[i:i+patch_size]
            patch_indices.append(patch)
            used[patch] = True

    # Flatten and relabel for visualization
    labels_full = np.full(len(latCell), -1)
    for i, inds in enumerate(patch_indices):
        labels_full[inds] = i
    
    bar_graph_cluster_distribution(labels_full, mask=np.ones_like(latCell, dtype=bool), algorithm="latitude_simple")
    
    #return patch_indices
    return labels_full


def patchify_by_latitude(latCell, lonCell, patch_size=49, 
                         min_lat=40, max_lat=90, 
                         step_deg=3, seed=42):
    """
    Group mesh cells into patches from north pole southward using latitude bands.
    """
    latCell = np.array(latCell)
    lonCell = np.array(lonCell)
    coords = latlon_to_xyz(latCell, lonCell)
    
    rng = np.random.default_rng(seed)
    used = np.zeros(len(latCell), dtype=bool)
    patch_indices = []
    
    for band_max in np.arange(max_lat, min_lat, -step_deg):
        band_min = band_max - step_deg
        band_mask = (latCell >= band_min) & (latCell < band_max) & (~used)
        band_indices = np.where(band_mask)[0]

        if len(band_indices) < patch_size:
            continue

        band_coords = coords[band_indices]

        # Fit kNN model
        knn = NearestNeighbors(n_neighbors=patch_size)
        knn.fit(band_coords)

        selected = set()
        attempts = 0

        while len(selected) < len(band_indices) and attempts < len(band_indices):
            seed_idx = rng.choice(band_indices)
            if used[seed_idx]:
                attempts += 1
                continue

            _, nbrs = knn.kneighbors([coords[seed_idx]])
            nbrs_global = band_indices[nbrs[0]]
            
            # Ensure none of them are already used
            if np.any(used[nbrs_global]):
                attempts += 1
                continue

            patch_indices.append(nbrs_global)
            used[nbrs_global] = True
            attempts += 1

    # Flatten and relabel for visualization
    labels_full = np.full(len(latCell), -1)
    for i, inds in enumerate(patch_indices):
        labels_full[inds] = i
    
    bar_graph_cluster_distribution(labels_full, mask=np.ones_like(latCell, dtype=bool), algorithm="latitude_neighbors")

    #return patch_indices  # list of arrays
    return labels_full

import random

def grow_patch(seed_idx, cellsOnCell, visited, patch_size, valid_mask=None):
    """
    Grow a patch starting from seed_idx using mesh adjacency.
    This method is used in "build_patches_from_seeds"
    """
    patch = set()
    frontier = [seed_idx]
    visited.add(seed_idx)

    while frontier and len(patch) < patch_size:
        current = frontier.pop(0)
        patch.add(current)

        neighbors = cellsOnCell[current]
        for neighbor in neighbors:
            if neighbor == -1:
                continue  # Skip invalid neighbors
            if neighbor in visited:
                continue
            if valid_mask is not None and not valid_mask[neighbor]:
                continue  # Skip masked-out cells

            visited.add(neighbor)
            frontier.append(neighbor)

    return list(patch)  # Return as list for consistency

def build_patches_from_seeds(cellsOnCell, n_patches=727, patch_size=49, seed=42, valid_mask=None):
    rng = np.random.default_rng(seed)
    nCells = cellsOnCell.shape[0]

    if valid_mask is None:
        valid_mask = np.ones(nCells, dtype=bool)

    # Precompute valid indices to sample seeds from
    candidate_seeds = np.where(valid_mask)[0]
    rng.shuffle(candidate_seeds)

    visited = set()
    patches = []

    for seed_idx in candidate_seeds:
        if seed_idx in visited:
            continue

        patch = grow_patch(seed_idx, cellsOnCell, visited, patch_size, valid_mask)

        if len(patch) < patch_size:
            continue  # Skip undersized patches

        patches.append(patch)
        if len(patches) >= n_patches:
            break

    print(f"Built {len(patches)} patches of size ~{patch_size}")

    labels_full = np.full(cellsOnCell.shape[0], -1, dtype=int)
    for i, patch in enumerate(patches):
        labels_full[patch] = i

    bar_graph_cluster_distribution(labels_full, valid_mask, algorithm="breadth_first")

    #return patches
    return labels_full

def compute_agglomerative_patches(latCell, lonCell, lat_threshold=50, n_patches=727, n_neighbors=6):

    mask = latCell > lat_threshold
    coords = latlon_to_xyz(latCell[mask], lonCell[mask])
    
    # Create spatial connectivity graph
    connectivity = kneighbors_graph(coords, n_neighbors=n_neighbors, include_self=False)
    
    # Cluster
    agg = AgglomerativeClustering(
        n_clusters=n_patches,
        connectivity=connectivity,
        linkage='ward'
    )
    labels = agg.fit_predict(coords)
    
    # Insert back into full array
    labels_full = np.full(latCell.shape, -1, dtype=int)
    labels_full[mask] = labels

    bar_graph_cluster_distribution(labels_full, mask, algorithm="agglomerative")

    return labels_full
    
def compute_disjoint_knn_patches(latCell, lonCell, lat_threshold=50, cells_per_patch=49, n_patches=727, 
                                  seed=42):
    mask = latCell > lat_threshold
    lat_filtered = latCell[mask]
    lon_filtered = lonCell[mask]
    coords_xyz = latlon_to_xyz(lat_filtered, lon_filtered)
    original_indices = np.nonzero(mask)[0]

    knn = NearestNeighbors(n_neighbors=cells_per_patch * 3, algorithm='auto')  # oversample neighbors
    knn.fit(coords_xyz)

    rng = np.random.default_rng(seed)
    center_ids = rng.choice(len(lat_filtered), size=n_patches, replace=False)

    labels_masked = np.full(len(lat_filtered), -1, dtype=int)
    used = np.zeros(len(lat_filtered), dtype=bool)

    for i, cid in enumerate(center_ids):
        neighbors = knn.kneighbors([coords_xyz[cid]], return_distance=False)[0]

        # Select only neighbors not already used
        new_patch = [nid for nid in neighbors if not used[nid]]
        new_patch = new_patch[:cells_per_patch]  # trim to desired size

        # Mark as used and assign label
        for nid in new_patch:
            labels_masked[nid] = i
            used[nid] = True

    labels_full = np.full(latCell.shape, -1, dtype=int)
    labels_full[mask] = labels_masked

    bar_graph_cluster_distribution(labels_full, mask, algorithm="knn_disjoint")

    return labels_full


def compute_knn_patches(latCell, lonCell, lat_threshold=50, cells_per_patch=49, n_patches=727, 
                                    seed=42):
    # 1. Apply mask
    mask = latCell > lat_threshold
    lat_filtered = latCell[mask]
    lon_filtered = lonCell[mask]
    coords_xyz = latlon_to_xyz(lat_filtered, lon_filtered)
    original_indices = np.nonzero(mask)[0]

    # 2. Fit kNN
    knn = NearestNeighbors(n_neighbors=cells_per_patch, algorithm='auto')
    knn.fit(coords_xyz)

    # 3. Choose patch centers (randomly among valid ones)
    rng = np.random.default_rng(seed)
    center_ids = rng.choice(len(lat_filtered), size=n_patches, replace=False)

    # 4. Query neighbors and assign patch labels
    patch_indices = []
    labels_masked = np.full(len(lat_filtered), -1, dtype=int)

    for i, cid in enumerate(center_ids):
        neighbors = knn.kneighbors([coords_xyz[cid]], return_distance=False)[0]
        patch_indices.append(original_indices[neighbors])
        labels_masked[neighbors] = i

    # 5. Reinsert into full-length label array
    labels_full = np.full(latCell.shape, -1, dtype=int)
    labels_full[mask] = labels_masked

    patch_indices = np.array(patch_indices)  # (n_patches, cells_per_patch)
    print("patch indices: === ", patch_indices)

    # 6. Visualize the bar graph
    bar_graph_cluster_distribution(labels_full, mask, algorithm = "knn_basic")

    #return patch_indices, labels_full
    return labels_full

def cluster_patches_kmeans(latCell, lonCell, lat_threshold=50, n_patches = 727, seed=42):
    """ Take lat and lon values from only 50 degrees north and cluster them into patches of 49
    cells per patch for 727 patches and 35623 mesh cells. Returns an array of all 465,044 mesh cells
    where -1 indicates that there was no cluster. """
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > lat_threshold           # Boolean array, True for 35623 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask))
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f))  # shape (35623, 2)
    
                                      # 35623 ÷ 49
    kmeans = KMeans(
        n_clusters=n_patches,
        init="k-means++",     # default, good for speed / accuracy
        n_init="auto",        # auto-scales n_init in recent scikit-learn versions
        random_state=seed,      # make runs reproducible
        algorithm="elkan"     # faster for low-dimension dense data
    ).fit(coords)
    
    centroids = kmeans.cluster_centers_       # array shape (727, 2)
    print("centroids", centroids.shape)
    labels_f = kmeans.labels_                 # length 35623
    print("labels", len(labels_f))
    
    # --- 3.  Re-insert labels into full-length array ---------------
    # Make an array filled with -1's in the shape of latCell
    labels_full = np.full(latCell.shape, -1, dtype=int)  # –1 marks “not clustered”
    labels_full[mask] = labels_f # Populate the array with the labels from the clustering

    bar_graph_cluster_distribution(labels_full, mask, algorithm="kmeans")
    
    return labels_full

def get_rows_of_patches(latCell, lonCell, lat_threshold=50, cells_per_cluster=49):
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > lat_threshold
    indices = np.where(latCell > LAT_LIMIT)
    print("Non-zeroes", np.count_nonzero(indices[0]))

    labels_full = np.full(latCell.shape, -1, dtype=int)
    bucket = 0

    for i, index in enumerate(indices[0]):
        labels_full[index] = bucket
        if (i+1) % 49 == 0 and i > 0:
            bucket += 1
    
    bar_graph_cluster_distribution(labels_full, mask, algorithm="row_by_row")
    
    return labels_full

def get_clusters_dbscan(latCell, lonCell, lat_threshold=50, cells_per_cluster=49):

    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > lat_threshold          # Boolean array, True for 35623 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask)) # prints 35623
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f)) # shape (35623, 2)
    print("Shape of coords", coords.shape)

    # make an elbow plot to see the best value for eps
    
    neighbors = NearestNeighbors(n_neighbors=cells_per_cluster, algorithm='ball_tree', metric='haversine')
    neighbors_fit = neighbors.fit(coords)
    distances, indices = neighbors_fit.kneighbors(coords)

    distances = np.sort(distances, axis=0)
    distances = distances[:, cells_per_cluster-1]  # index 48 = 49th nearest
    plt.plot(distances)
    plt.savefig("elbow_plot.png")
    plt.close()

    # The elbow plot has an elbow around 0.022
    
    # DBSCAN parameters
    
    eps = 0.022  # Epsilon (radius)
    min_samples = 20  # MinPts (minimum points within the radius)

    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    dbscan.fit(coords)

    # Get cluster labels
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Estimated number of clusters:", n_clusters)
    print("Estimated number of noise points:", n_noise)
    print("Cluster sizes:", collections.Counter(labels))
    
    # --- 3.  Re-insert labels into full-length array ---------------
    # Make an array filled with -1's in the shape of latCell
    labels_full = np.full(latCell.shape, -1, dtype=int)  # –1 marks “not clustered”
    labels_full[mask] = labels # Populate the array with the labels from the clustering

    bar_graph_cluster_distribution(labels_full, mask, algorithm="dbscan")

    return labels_full
    