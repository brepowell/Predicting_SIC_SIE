from NC_FILE_PROCESSING.nc_utility_functions import *
# from sklearn.cluster import KMeans
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.neighbors import kneighbors_graph
import numpy as np
import random
import collections
import matplotlib.pyplot as plt

def latlon_to_xyz(lat, lon):
    """Convert lat/lon (in degrees) to 3D unit sphere coordinates"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=1)  # shape (nCells, 3)

def bar_graph_cluster_distribution(labels_full, mask, algorithm):
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
    print("number of patches:" , distinct_count) # Output: 210

    total_cells_after_mask = np.count_nonzero(mask)
    ylim_patch_size = (total_cells_after_mask / distinct_count) + 50

    plt.bar(patch_ids, cells_per_patch)
    plt.ylim(0, ylim_patch_size)  # Sets the y-axis from 0 to 20 more than the expected patch size
    plt.xlabel("Patch ID's")
    plt.ylabel("Cell Count")

    plt.title("Cells per Patch")
    plt.savefig(f"distribution_{algorithm}_patches_{distinct_count}.png")
    plt.close()

def patchify_staggered_polar_descent(latCell, lonCell, k=256, max_patches=210, latitude_threshold=40.0):
    """
    Creates spatially coherent patches starting from the North Pole and staggering
    downward by latitude and longitude. Each patch will contain exactly 'k' cells.

    Parameters
    ----------
    latCell : np.ndarray
        1D array of cell latitudes.
    lonCell : np.ndarray
        1D array of cell longitudes.
    k : int
        Exact number of cells per patch. Default is 256.
    max_patches : int
        Maximum number of patches to form.
    latitude_threshold : float
        Don't form patches below this latitude.

    Returns
    -------
    labels_full : np.ndarray
        Array of shape (nCells,) giving patch ID or -1 if unassigned.
    patch_indices : List[np.ndarray]
        List of patches, each a list of cell indices (np.ndarray of ints).
    patch_latlons : np.ndarray
        Array of shape (n_patches, 2) containing (latitude, longitude) for one
        representative cell per patch (the first cell added to the patch).
    """

    algorithm="staggered_polar_descent"
    
    nCells = len(latCell)
    # 'used' array tracks which cells have already been assigned to a patch
    used = np.zeros(nCells, dtype=bool) 

    # Filter valid candidates based on the latitude threshold
    mask = latCell >= latitude_threshold
    indices = np.where(mask)[0] # Get original indices of cells that meet the criteria

    # Sort the eligible cell indices:
    # Primary sort key: latitude (descending) - ensures we start from the North Pole
    # Secondary sort key: longitude (ascending) - ensures a consistent scan across latitude bands
    sort_order = np.lexsort((lonCell[indices], -latCell[indices]))
    sorted_indices = indices[sort_order] # Original indices, sorted as desired

    patch_indices = [] # List to store arrays of cell indices for each patch
    patch_latlons = []  # List to store (lat, lon) of the first cell of each patch
    
    # Initialize labels_full with -1 for all cells, indicating they are unassigned
    labels_full = np.full(nCells, -1, dtype=int)
    
    patch_id = 0 # Counter for the current patch ID
    
    # We will iterate through the sorted_indices and pick cells for patch_indices.
    # We need a pointer to the current position in the sorted_indices array.
    current_sorted_idx_pointer = 0

    while current_sorted_idx_pointer < len(sorted_indices) and patch_id < max_patches:
        # Find the next unused cell to start a new patch
        seed_cell_original_idx = -1
        while current_sorted_idx_pointer < len(sorted_indices):
            potential_seed_idx = sorted_indices[current_sorted_idx_pointer]
            if not used[potential_seed_idx]:
                seed_cell_original_idx = potential_seed_idx
                break
            current_sorted_idx_pointer += 1 # Move pointer past used cells

        if seed_cell_original_idx == -1:
            # No more unused cells found among the eligible ones
            break

        current_patch_indices = []
        current_patch_first_latlon = (latCell[seed_cell_original_idx], lonCell[seed_cell_original_idx])

        # Greedily collect 'k' cells for the current patch
        # Start from the seed cell and move forward in the sorted list
        temp_pointer = current_sorted_idx_pointer
        while len(current_patch_indices) < k and temp_pointer < len(sorted_indices):
            cell_to_add_original_idx = sorted_indices[temp_pointer]
            if not used[cell_to_add_original_idx]:
                current_patch_indices.append(cell_to_add_original_idx)
                used[cell_to_add_original_idx] = True # Mark as used
            temp_pointer += 1
        
        # If we successfully collected 'k' cells, finalize the patch
        if len(current_patch_indices) == k:
            patch_indices.append(np.array(current_patch_indices, dtype=int))
            labels_full[current_patch_indices] = patch_id
            patch_latlons.append(current_patch_first_latlon)
            
            patch_id += 1 # Increment patch ID for the next patch
        else:
            # If we couldn't form a full patch of 'k' cells (e.g., not enough remaining cells),
            # revert 'used' status for these cells and break the loop for this patch.
            # These cells will remain unassigned.
            for idx in current_patch_indices:
                used[idx] = False # Unmark them as used
            # No full patch formed, so we stop trying to form patches from this region
            break 
        
        # Advance the main pointer to the next unassigned cell for the next iteration
        # This is important to avoid re-processing cells that were part of the current patch
        # or were skipped because they were already used.
        # The while loop at the beginning of the main loop handles skipping used cells.
        current_sorted_idx_pointer = temp_pointer

    print(f"Built {len(patch_indices)} patches, each with exactly {k} cells.")
    
    # Call the bar graph function to visualize the distribution of the created patch_indices.
    # The 'mask' here ensures the bar graph considers only the cells that were eligible.
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    # Return the full labels array, the list of patch index arrays, and the representative lat/lons
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def patchify_by_lon_spilldown(latCell, lonCell, k=256, max_patches=210, latitude_threshold=40.0):
    """
    Creates spatially coherent patches by grouping cells primarily by longitude
    and then spilling downward in latitude. Each patch will contain exactly 'k' cells.

    Parameters
    ----------
    latCell : np.ndarray
        1D array of cell latitudes.
    lonCell : np.ndarray
        1D array of cell longitudes.
    k : int
        Exact number of cells per patch.
    max_patches : int
        Maximum number of patches to form.
    latitude_threshold : float
        Don't form patches below this latitude.

    Returns
    -------
    labels_full : np.ndarray
        Array of shape (nCells,) giving patch ID or -1 if unassigned.
    patch_indices : List[np.ndarray]
        List of patches, each a list of cell indices (np.ndarray of ints).
    patch_latlons : np.ndarray
        Array of shape (n_patches, 2) containing (latitude, longitude) for one
        representative cell per patch (the first cell added to the patch).
    """

    algorithm="lon_spilldown"
    
    nCells = len(latCell)
    used = np.zeros(nCells, dtype=bool) # Tracks which cells have been assigned to a patch

    # Filter valid candidates based on the latitude threshold
    mask = latCell >= latitude_threshold
    indices = np.where(mask)[0] # Get original indices of cells that meet the criteria

    # Sort the eligible cell indices:
    # Primary sort key: longitude (ascending) - groups cells into "columns"
    # Secondary sort key: latitude (descending) - within each column, processes from north to south
    sort_order = np.lexsort((-latCell[indices], lonCell[indices]))
    sorted_indices = indices[sort_order] # Original indices, sorted as desired

    patch_indices = [] # List to store arrays of cell indices for each patch
    patch_latlons = []  # List to store (lat, lon) of the first cell of each patch
    
    # Initialize labels_full with -1 for all cells, indicating they are unassigned
    labels_full = np.full(nCells, -1, dtype=int)
    
    patch_id = 0 # Counter for the current patch ID
    current_patch_indices = [] # Temporarily holds indices for the patch being built
    current_patch_first_latlon = None # Stores the lat/lon of the first cell in the current patch

    # Iterate through the sorted cells to form patches
    for idx in sorted_indices:
        if used[idx]:
            # Skip cells that have already been assigned to a patch
            continue

        # If we are starting a new patch, record the lat/lon of its first cell
        if not current_patch_indices:
            current_patch_first_latlon = (latCell[idx], lonCell[idx])

        # Add the current cell to the patch being built
        current_patch_indices.append(idx)
        used[idx] = True # Mark this cell as used

        # If the current patch has reached the desired size 'k'
        if len(current_patch_indices) == k:
            # Finalize the patch
            patch_indices.append(np.array(current_patch_indices, dtype=int))
            labels_full[current_patch_indices] = patch_id # Assign patch ID to these cells
            patch_latlons.append(current_patch_first_latlon) # Store representative lat/lon
            
            patch_id += 1 # Increment patch ID for the next patch
            current_patch_indices = [] # Reset for the next patch
            current_patch_first_latlon = None # Reset the representative lat/lon

            # Stop if the maximum number of patches has been reached
            if patch_id >= max_patches:
                print(f"Reached max_patches ({max_patches}). Stopping patch formation.")
                break

    print(f"Built {len(patch_indices)} patches, each with exactly {k} cells.")
    
    # Call the bar graph function to visualize the distribution of the created patch_indices.
    # The 'mask' here ensures the bar graph considers only the cells that were eligible.
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    # Return the full labels array, the list of patch index arrays, and the representative lat/lons
    return labels_full, patch_indices, np.array(patch_latlons), algorithm 

def patchify_by_latlon_spillover(latCell, lonCell, k=256, max_patches=210, latitude_threshold=40.0):
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
    latitude_threshold : float
        Don't form patches below this latitude.

    Returns
    -------
    labels_full : np.ndarray
        Array of shape (nCells,) giving patch ID or -1 if unassigned.
    patches : List[np.ndarray]
        List of patches, each a list of cell indices (np.ndarray of ints).
    patch_latlons : np.ndarray
        Array of shape (n_patches, 2) containing (latitude, longitude) for one
        representative cell per patch.
    """

    algorithm="latlon_spillover"

    nCells = len(latCell)
    used = np.zeros(nCells, dtype=bool)

    # Filter valid candidates
    mask = latCell >= latitude_threshold
    indices = np.where(mask)[0]

    # Sort by latitude (descending), then by longitude (ascending)
    # lexsort returns indices from the original arrays (ex. 0 through 465044)
    sort_order = np.lexsort((lonCell[indices], -latCell[indices]))
    sorted_indices = indices[sort_order]

    patch_indices = []
    patch_latlons = []  # List to store (lat, lon) for each patch
    labels_full = np.full(nCells, -1, dtype=int)
    patch_id = 0
    current_patch = []
    
    # Variable to store the lat/lon of the first cell in the current patch
    current_patch_first_latlon = None 

    for idx in sorted_indices:
        if used[idx]:
            continue

        if not current_patch:  # If this is the first cell in a new patch
            current_patch_first_latlon = (latCell[idx], lonCell[idx])

        current_patch.append(idx)
        used[idx] = True

        if len(current_patch) == k:
            patch_indices.append(np.array(current_patch, dtype=int))
            labels_full[current_patch] = patch_id
            patch_latlons.append(current_patch_first_latlon) # Add the lat/lon of the first cell
            patch_id += 1
            current_patch = []
            current_patch_first_latlon = None # Reset for the next patch

            if patch_id >= max_patches:
                break

    # Add a final partial patch
    if current_patch and patch_id < max_patches:
        patch_indices.append(np.array(current_patch, dtype=int))
        labels_full[current_patch] = patch_id
        # If there's a leftover patch, its first cell's lat/lon was already captured
        if current_patch_first_latlon is not None:
             patch_latlons.append(current_patch_first_latlon)
        else:
            # Fallback if somehow current_patch_first_latlon was not set for the last patch
            patch_latlons.append((latCell[current_patch[0]], lonCell[current_patch[0]]))

    print(f"Built {len(patch_indices)} patches of size ~{k}")
    
    # Flatten and relabel for visualization
    labels_full_final = np.full(len(latCell), -1)
    for i, inds in enumerate(patch_indices):
        labels_full_final[inds] = i
    
    bar_graph_cluster_distribution(labels_full_final, mask=np.ones_like(latCell, dtype=bool), algorithm=algorithm)

    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def patchify_with_spillover(latCell, patch_size=256, 
                            min_lat=40, max_lat=90, 
                            step_deg=3, max_patches=None, seed=42):
    """
    Sweep latitudinal bands and form spatially cohesive patches, 
    letting underfilled patches spill into the next band.
    """

    algorithm="latitude_spillover_redo"
    
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
    
    bar_graph_cluster_distribution(labels_full, mask=np.ones_like(latCell, dtype=bool), algorithm=algorithm)

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
    
    patch_latlons = [(latCell[patch[0]], 0.0) for patch in patch_indices]
    
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def patchify_by_latitude_simple(latCell, patch_size=256, 
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
    
    algorithm="latitude_simple"
    
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
    
    bar_graph_cluster_distribution(labels_full, mask=np.ones_like(latCell, dtype=bool), algorithm=algorithm)
    
    patch_latlons = [(latCell[patch[0]], 0.0) for patch in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def patchify_by_latitude(latCell, lonCell, patch_size=256, 
                         min_lat=40, max_lat=90, 
                         step_deg=3, seed=42):
    """
    Group mesh cells into patches from north pole southward using latitude bands.
    """
    algorithm="latitude_neighbors"
    
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
    
    bar_graph_cluster_distribution(labels_full, mask=np.ones_like(latCell, dtype=bool), algorithm=algorithm)

    patch_latlons = [(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def grow_patch_improved(seed_idx, cellsOnCell, visited, patch_size, mask=None):
    """
    Grow a patch starting from seed_idx using mesh adjacency.
    This method is used in "build_patches_from_seeds"
    """
    patch = set()
    frontier = collections.deque([seed_idx]) # Use deque for efficient pop(0)
    visited.add(seed_idx)

    # Add the seed to the patch immediately if it's valid
    if mask is None or mask[seed_idx]:
        patch.add(seed_idx)
    else:
        # If the seed itself is invalid, we can't start a patch from it
        # This case should ideally be prevented by `candidate_seeds`
        return []

    while frontier and len(patch) < patch_size:
        current = frontier.popleft() # Use popleft for BFS behavior

        neighbors = cellsOnCell[current]
        for neighbor in neighbors:
            if neighbor == -1:
                continue  # Skip invalid neighbors
            if neighbor in visited:
                continue
            if mask is not None and not mask[neighbor]:
                continue  # Skip masked-out cells

            # If adding this neighbor would exceed patch_size, don't add it to frontier
            # (but still mark as visited to prevent future patches from claiming it in this round)
            if len(patch) < patch_size: # Only add to patch if there's space
                patch.add(neighbor)
                visited.add(neighbor)
                frontier.append(neighbor) # Add to frontier only if it was added to patch
            else:
                # If patch is already full, just mark this neighbor as visited
                # if we want to prevent other patches from immediately taking it
                # For strict patch_size, we just stop growing this patch.
                visited.add(neighbor) # Still mark as visited to prevent re-seeding / re-visiting immediately

    return list(patch)  # Return as list for consistency

def build_patches_from_seeds_improved(cellsOnCell, n_patches=210, patch_size=256, seed=42, mask=None, pad_to_exact_size=True):
    algorithm = "breadth_first_improved_padded" if pad_to_exact_size else "breadth_first_improved"

    rng = np.random.default_rng(seed)
    nCells = cellsOnCell.shape[0]

    if mask is None:
        mask = np.ones(nCells, dtype=bool)

    # Filter candidate seeds to only include valid, unvisited cells
    candidate_seeds_pool = np.where(mask)[0]
    rng.shuffle(candidate_seeds_pool) # Randomize starting order of potential seeds

    visited = set() # Global set of visited cells
    patch_indices = [] # Stores actual cell indices for each patch

    # Full array to store patch labels
    labels_full = np.full(nCells, -1, dtype=int)

    # Iterate through randomized candidate seeds
    for seed_idx in candidate_seeds_pool:
        if seed_idx in visited:
            continue # Skip if this cell has already been assigned to a patch or visited during a growth process

        # Try to grow a patch from this seed
        current_patch_cells = grow_patch_improved(seed_idx, cellsOnCell, visited, patch_size, mask) # visited is modified by grow_patch

        if len(current_patch_cells) == patch_size:
            # Successfully formed a full patch
            patch_indices.append(np.array(current_patch_cells, dtype=int))
            patch_id = len(patch_indices) - 1
            labels_full[current_patch_cells] = patch_id
        elif len(current_patch_cells) > 0 and pad_to_exact_size:
            # Undersized patch that we need to pad
            padded_patch = np.full(patch_size, -2, dtype=int) # -2 as dummy for padding
            padded_patch[:len(current_patch_cells)] = current_patch_cells
            patch_indices.append(padded_patch)
            patch_id = len(patch_indices) - 1
            labels_full[current_patch_cells] = patch_id # Only label the actual cells
        elif len(current_patch_cells) > 0 and not pad_to_exact_size:
            # Undersized patch, but we don't pad, so we add it as is
            patch_indices.append(np.array(current_patch_cells, dtype=int))
            patch_id = len(patch_indices) - 1
            labels_full[current_patch_cells] = patch_id
        else:
            # current_patch_cells is empty (e.g., seed was invalid or isolated)
            continue # Don't increment patch count or try to save an empty patch

        if len(patch_indices) >= n_patches:
            break # Stop if max number of patches is reached

    print(f"Built {len(patch_indices)} patches. Actual cell counts: {[len(p) if not pad_to_exact_size else np.count_nonzero(p != -2) for p in patch_indices]}")

    # Note: For bar_graph_cluster_distribution, use labels_full and the original mask
    # as it counts actual assigned cells, not padded ones.
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    # The patch_latlons here is a placeholder. You'll need latCell/lonCell
    # to compute actual (lat, lon) for each patch.
    # For now, it will return an empty array or needs modification to take latCell/lonCell
    patch_latlons = np.array([(0.0, 0.0)] * len(patch_indices)) # Placeholder

    return labels_full, patch_indices, patch_latlons, algorithm

    
def grow_patch(seed_idx, cellsOnCell, visited, patch_size, mask=None):
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
            if mask is not None and not mask[neighbor]:
                continue  # Skip masked-out cells

            visited.add(neighbor)
            frontier.append(neighbor)

    return list(patch)  # Return as list for consistency

def build_patches_from_seeds(cellsOnCell, latCell, lonCell, n_patches=210, patch_size=256, seed=42, mask=None):
    algorithm="breadth_first"
    
    rng = np.random.default_rng(seed)
    nCells = cellsOnCell.shape[0]

    if mask is None:
        mask = np.ones(nCells, dtype=bool)

    # Precompute valid indices to sample seeds from
    candidate_seeds = np.where(mask)[0]
    rng.shuffle(candidate_seeds)

    visited = set()
    patch_indices = []

    for seed_idx in candidate_seeds:
        if seed_idx in visited:
            continue

        patch = grow_patch(seed_idx, cellsOnCell, visited, patch_size, mask)

        if len(patch) < patch_size:
            continue  # Skip undersized patches

        patch_indices.append(patch)
        if len(patch_indices) >= n_patches:
            break

    print(f"Built {len(patch_indices)} patches of size ~{patch_size}")

    labels_full = np.full(cellsOnCell.shape[0], -1, dtype=int)
    for i, patch in enumerate(patch_indices):
        labels_full[patch] = i

    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_latlons = [(np.mean(latCell[patch]), 0.0) for patch in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm 

def compute_agglomerative_patches(latCell, lonCell, latitude_threshold=40, n_patches=210, n_neighbors=6):

    algorithm="agglomerative"
    mask = latCell > latitude_threshold
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

    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_indices = [np.where(labels_full == i)[0] for i in range(n_patches)]
    patch_latlons = [(latCell[inds[0]], lonCell[inds[0]]) for inds in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def compute_disjoint_knn_patches(latCell, lonCell, latitude_threshold=40, cells_per_patch=256, n_patches=210, 
                                  seed=42):
    algorithm="knn_disjoint"
    
    mask = latCell > latitude_threshold
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

    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_indices = [original_indices[labels_masked == i] for i in range(n_patches) if np.any(labels_masked == i)]
    patch_latlons = [(latCell[inds[0]], lonCell[inds[0]]) for inds in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def compute_knn_patches(latCell, lonCell, latitude_threshold=40, cells_per_patch=256, n_patches=210, 
                                    seed=42):

    algorithm = "knn_basic"
    # 1. Apply mask
    mask = latCell > latitude_threshold
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
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_latlons = [(latCell[inds[0]], lonCell[inds[0]]) for inds in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def cluster_patches_kmeans(latCell, lonCell, latitude_threshold=40, n_patches = 210, seed=42):
    """ Take lat and lon values from only 50 degrees north and cluster them into patches of 256
    cells per patch for 210 patches and 53973 mesh cells. Returns an array of all 465,044 mesh cells
    where -1 indicates that there was no cluster. """

    algorithm="kmeans"
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > latitude_threshold           # Boolean array, True for 53973 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask))
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f))  # shape (53973, 2)
    
                                      # 53973 ÷ 256
    kmeans = KMeans(
        n_clusters=n_patches,
        init="k-means++",     # default, good for speed / accuracy
        n_init="auto",        # auto-scales n_init in recent scikit-learn versions
        random_state=seed,      # make runs reproducible
        algorithm="elkan"     # faster for low-dimension dense data
    ).fit(coords)
    
    centroids = kmeans.cluster_centers_       # array shape (210, 2)
    print("centroids", centroids.shape)
    labels_f = kmeans.labels_                 # length 53973
    print("labels", len(labels_f))
    
    # --- 3.  Re-insert labels into full-length array ---------------
    # Make an array filled with -1's in the shape of latCell
    labels_full = np.full(latCell.shape, -1, dtype=int)  # –1 marks “not clustered”
    labels_full[mask] = labels_f # Populate the array with the labels from the clustering

    bar_graph_cluster_distribution(labels_full, mask, algorithm)
    
    patch_indices = [np.where(labels_full == i)[0] for i in range(n_patches)]
    patch_latlons = [(latCell[inds].mean(), lonCell[inds].mean()) for inds in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def get_rows_of_patches(latCell, lonCell, latitude_threshold=40, cells_per_cluster=256):

    algorithm="row_by_row"
    
    mask = latCell > latitude_threshold
    indices = np.where(latCell > LAT_LIMIT)
    print("Non-zeroes", np.count_nonzero(indices[0]))

    labels_full = np.full(latCell.shape, -1, dtype=int)
    bucket = 0

    for i, index in enumerate(indices[0]):
        labels_full[index] = bucket
        if (i+1) % 256 == 0 and i > 0:
            bucket += 1
    
    bar_graph_cluster_distribution(labels_full, mask, algorithm)
    
    patch_indices = [np.where(labels_full == i)[0] for i in range(np.max(labels_full)+1)]
    patch_latlons = [(latCell[inds].mean(), lonCell[inds].mean()) for inds in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def get_clusters_dbscan(latCell, lonCell, latitude_threshold=40, cells_per_cluster=256):

    algorithm="dbscan"
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell > latitude_threshold          # Boolean array, True for 53973 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask)) # prints 53973
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f)) # shape (53973, 2)
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

    bar_graph_cluster_distribution(labels_full, mask, algorithm=algorithm)

    patch_ids = np.unique(labels[labels >= 0])
    patch_indices = [np.where(labels_full == pid)[0] for pid in patch_ids]
    patch_latlons = [(latCell[inds[0]], lonCell[inds[0]]) for inds in patch_indices]
    return labels_full, patch_indices, np.array(patch_latlons), algorithm
