from NC_FILE_PROCESSING.nc_utility_functions import *
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
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
    plt.savefig(f"{algorithm}.png")
    plt.close()

def compute_disjoint_knn_patches(latCell, lonCell, k=49, n_patches=727, 
                                  seed=42, lat_threshold=50):
    mask = latCell > lat_threshold
    lat_filtered = latCell[mask]
    lon_filtered = lonCell[mask]
    coords_xyz = latlon_to_xyz(lat_filtered, lon_filtered)
    original_indices = np.nonzero(mask)[0]

    knn = NearestNeighbors(n_neighbors=k * 3, algorithm='auto')  # oversample neighbors
    knn.fit(coords_xyz)

    rng = np.random.default_rng(seed)
    center_ids = rng.choice(len(lat_filtered), size=n_patches, replace=False)

    labels_masked = np.full(len(lat_filtered), -1, dtype=int)
    used = np.zeros(len(lat_filtered), dtype=bool)

    for i, cid in enumerate(center_ids):
        neighbors = knn.kneighbors([coords_xyz[cid]], return_distance=False)[0]

        # Select only neighbors not already used
        new_patch = [nid for nid in neighbors if not used[nid]]
        new_patch = new_patch[:k]  # trim to desired size

        # Mark as used and assign label
        for nid in new_patch:
            labels_masked[nid] = i
            used[nid] = True

    labels_full = np.full(latCell.shape, -1, dtype=int)
    labels_full[mask] = labels_masked

    bar_graph_cluster_distribution(labels_full, mask, algorithm="knn_disjoint")

    return labels_full


def compute_knn_patches_with_labels(latCell, lonCell, k=49, n_patches=727, 
                                    seed=42, lat_threshold=50):
    # 1. Apply mask
    mask = latCell > lat_threshold
    lat_filtered = latCell[mask]
    lon_filtered = lonCell[mask]
    coords_xyz = latlon_to_xyz(lat_filtered, lon_filtered)
    original_indices = np.nonzero(mask)[0]

    # 2. Fit kNN
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
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

    patch_indices = np.array(patch_indices)  # (n_patches, k)
    print("patch indices: === ", patch_indices)

    # 6. Visualize the bar graph
    bar_graph_cluster_distribution(labels_full, mask, algorithm = "knn_patches")

    #return patch_indices, labels_full
    return labels_full


def cluster_patches_kmeans_padded(latCell, lonCell):
    """  """
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    indices = np.where(latCell > 40)              
    lat_f = np.radians(latCell[indices])
    lon_f = np.radians(lonCell[indices])
    
    print("indices size", indices[0].shape)
    print("smallest index", indices[0][0])
    print("largest index", indices[0][-1])

    ideal_cluster_count_k = 640
    ideal_cells_per_cluster = 64
    ideal_cell_count = ideal_cluster_count_k * ideal_cells_per_cluster

    cropped_indices = (indices[0][:ideal_cell_count],) + indices[1:]
    lat_f = np.radians(latCell[cropped_indices])
    lon_f = np.radians(lonCell[cropped_indices])
    
    print("indices size", cropped_indices[0].shape)
    print("smallest index", cropped_indices[0][0])
    print("largest index", cropped_indices[0][-1])
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f))
    
    kmeans = KMeans(
        n_clusters=ideal_cluster_count_k,
        init="k-means++",     # default, good for speed / accuracy
        n_init="auto",        # auto-scales n_init in recent scikit-learn versions
        random_state=42,      # make runs reproducible
        algorithm="elkan"     # faster for low-dimension dense data
    ).fit(coords)
    
    centroids = kmeans.cluster_centers_ 
    print("centroids", centroids.shape)
    labels_f = kmeans.labels_ 
    print("labels", len(labels_f))
    
    # --- 3.  Re-insert labels into full-length array ---------------
    # Make an array filled with -1's in the shape of latCell
    labels_full = np.full(latCell.shape, -1, dtype=int)  # –1 marks “not clustered”
    labels_full[cropped_indices] = labels_f # Populate the array with the labels from the clustering

    counts = collections.Counter(labels_full[cropped_indices])
    print("Cluster sizes:", )
    
    patch_ids = list(counts.keys())
    cells_per_patch = list(counts.values())

    print("min size", min(cells_per_patch))
    print("max size", max(cells_per_patch))

    plt.bar(patch_ids, cells_per_patch)
    plt.xlabel("Patch ID's")
    plt.ylabel("Cell Count")
    plt.title("Cells per Patch")
    plt.savefig("histogram_kmeans_padded.png")
    plt.close()
    
    return labels_full, centroids

def cluster_patches_loop(latCell, lonCell):
    """ This experiment did not work -- after 49 iterations, it no longer found good matches. """

    # Initial Conditions
    mask = latCell > 45  # Boolean array, True for 35623 rows
    labels_full = np.full(latCell.shape, -1, dtype=int)  # –1 = unclustered
    
    n_clusters = 640
    cells_per_cluster = 64
    
    # Repeat
    for iteration in range(200):
        print(f"--- CLUSTERING PART {iteration} ---")
    
        # Filter the coords to re-cluster only unassigned cells
        lat_radians = np.radians(latCell[mask])
        lon_radians = np.radians(lonCell[mask])
        coords = np.column_stack((lat_radians, lon_radians))
    
        if len(coords) < cells_per_cluster:
            print("Too few cells left to cluster. Stopping.")
            break
    
        n_clusters = len(coords) // cells_per_cluster
    
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init="auto",
            random_state=42,
            algorithm="elkan"
        ).fit(coords)
    
        labels_f = kmeans.labels_
    
        # Count how many points in each cluster
        counts = np.bincount(labels_f, minlength=n_clusters)
        good_patch_ids = np.where(counts == cells_per_cluster)[0]
        print(f"Good patches found: {len(good_patch_ids)}")
    
        # Find which points belong to good patches
        is_good_point = np.isin(labels_f, good_patch_ids)
        good_indices_in_masked = np.where(mask)[0][is_good_point]  # these are indices in latCell, lonCell
    
        # Assign patch IDs to full label array
        for i, patch_id in enumerate(good_patch_ids):
            point_indices = np.where(labels_f == patch_id)[0]
            real_indices = np.where(mask)[0][point_indices]
            labels_full[real_indices] = patch_id
    
        # Update mask to exclude good patches
        mask[good_indices_in_masked] = False
        print(f"Remaining points to cluster: {np.count_nonzero(mask)}")
    
    return labels_full

def cluster_patches_kmeans(latCell, lonCell):
    """ Take lat and lon values from only 50 degrees north and cluster them into patches of 49
    cells per patch for 727 patches and 35623 mesh cells. Returns an array of all 465,044 mesh cells
    where -1 indicates that there was no cluster. """
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > 50                     # Boolean array, True for 35623 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask))
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f))  # shape (35623, 2)
    
    k = 727                                   # 35623 ÷ 49
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",     # default, good for speed / accuracy
        n_init="auto",        # auto-scales n_init in recent scikit-learn versions
        random_state=42,      # make runs reproducible
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

    print("Cluster sizes:", collections.Counter(labels_full[mask]))
    
    #plt.hist(labels_full[mask])
    counts, bins = np.histogram(labels_f)
    #plt.stairs(counts, bins)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.savefig("histogram_kmeans_patches.png")
    plt.close()
    
    return labels_full




def get_rows_of_patches(latCell, lonCell):
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    indices = np.where(latCell > LAT_LIMIT)
    print("Non-zeroes", np.count_nonzero(indices[0]))

    labels_full = np.full(latCell.shape, -1, dtype=int)

    bucket = 0

    for i, index in enumerate(indices[0]):
        labels_full[index] = bucket
        if (i+1) % 49 == 0 and i > 0:
            bucket += 1

    print("Cluster sizes:", collections.Counter(labels_full[indices]))
    
    plt.hist(labels_full[indices])
    plt.savefig("histogram_patch_rows.png")
    plt.close()
    
    return labels_full

def get_clusters_dbscan(latCell, lonCell):

    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > 50                     # Boolean array, True for 35623 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask)) # prints 35623
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f)) # shape (35623, 2)
    print("Shape of coords", coords.shape)

    # make an elbow plot to see the best value for eps
    
    neighbors = NearestNeighbors(n_neighbors=49, algorithm='ball_tree', metric='haversine')
    neighbors_fit = neighbors.fit(coords)
    distances, indices = neighbors_fit.kneighbors(coords)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 48]  # index 48 = 49th nearest
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

    print(len(labels_full[mask]))

    plt.hist(labels_full[mask])
    plt.savefig("histogram_dbscan_patches.png")
    plt.close()

    return labels_full
    