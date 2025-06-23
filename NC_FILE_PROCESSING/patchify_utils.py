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
    