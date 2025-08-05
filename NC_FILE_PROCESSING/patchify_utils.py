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
import warnings

def latlon_to_xyz(lat, lon):
    """Convert lat/lon (in degrees) to 3D unit sphere coordinates"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=1)  # shape (nCells, 3)

def bar_graph_cluster_distribution(labels_full, mask, algorithm):

    # 1. Filter labels to only include those under the mask
    # 2. Create counts for *all* labels within the masked region, including -1 if present
    all_masked_labels = labels_full[mask]
    counts = collections.Counter(all_masked_labels)

    # Store the count of unassigned cells (ID -1)
    unassigned_count = counts.get(-1, 0) # defaults to 0 if -1 not present

    # 1. Give a warning if -1 is a patch index
    if -1 in counts:
        warnings.warn(
            f"Warning for '{algorithm}': {unassigned_count} cells were unassigned (patch ID -1) "
            f"and will be excluded from the patch distribution plot."
        )
        # 2. Exclude -1 from the plot data by removing its entry from counts
        del counts[-1]     
    
    print("Cluster sizes:")  

    # Check if there are any valid patches left after removing -1
    if not counts: 
        print(f"No valid patches formed for algorithm '{algorithm}', or all cells were unassigned after masking.")
        # No plot can be generated if there are no valid patches
        return
        
    patch_ids = list(counts.keys())
    cells_per_patch = list(counts.values())

    print("min size", min(cells_per_patch))
    print("max size", max(cells_per_patch))

    # What patch has the largest number of cells or least number of cells?
    print("smallest count", counts.most_common()[-1]) # gives the tuple with the smallest count.        
    print("max count", counts.most_common(1)[0]) # gives the tuple with the largest count.

    distinct_count = len(counts)
    print("number of patches:" , distinct_count) # Output: 210

    # Calculate ylim based on the sum of cells in *assigned* patches
    sum_cells_in_assigned_patches = sum(cells_per_patch)
    ylim_patch_size = (sum_cells_in_assigned_patches / distinct_count) + 50 if distinct_count > 0 else 100

    plt.figure(figsize=(12, 6))
    plt.bar(patch_ids, cells_per_patch)
    plt.ylim(0, ylim_patch_size)
    plt.xlabel("Patch ID's")
    plt.ylabel("Cell Count")

    plt.title("Cells per Patch")
    plt.savefig(f"distribution_{algorithm}_patches_{distinct_count}.png")
    plt.close()

def patchify_staggered_polar_descent(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40.0, seed = 42):
    """
    Creates spatially coherent patches starting from the North Pole and staggering
    downward by latitude and longitude. Each patch will contain exactly 'cells_per_patch' cells.

    Parameters
    ----------
    latCell : np.ndarray
        1D array of cell latitudes.
    lonCell : np.ndarray
        1D array of cell longitudes.
    cells_per_patch : int
        Exact number of cells per patch. Default is 256.
    num_patches : int
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
        Array of shape (num_patches, 2) containing (latitude, longitude) for one
        representative cell per patch (the first cell added to the patch).
    """

    algorithm="staggered_polar_descent"
    
    nCells = len(latCell)
    # 'used' array tracks which cells have already been assigned to a patch
    used = np.zeros(nCells, dtype=bool) 

    # Filter valid candidates based on the latitude threshold
    mask = latCell >= latitude_threshold
    indices = np.where(mask)[0] # Get original indices of cells that meet the criteria

    print("Non-zeroes", np.count_nonzero(indices[0])) # This will print the exact number of cells being processed.


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

    while current_sorted_idx_pointer < len(sorted_indices) and patch_id < num_patches:
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

        # Greedily collect 'cells_per_patch' cells for the current patch
        # Start from the seed cell and move forward in the sorted list
        temp_pointer = current_sorted_idx_pointer
        while len(current_patch_indices) < cells_per_patch and temp_pointer < len(sorted_indices):
            cell_to_add_original_idx = sorted_indices[temp_pointer]
            if not used[cell_to_add_original_idx]:
                current_patch_indices.append(cell_to_add_original_idx)
                used[cell_to_add_original_idx] = True # Mark as used
            temp_pointer += 1
        
        # If we successfully collected 'cells_per_patch' cells, finalize the patch
        if len(current_patch_indices) == cells_per_patch:
            patch_indices.append(np.array(current_patch_indices, dtype=int))
            labels_full[current_patch_indices] = patch_id
            patch_latlons.append(current_patch_first_latlon)
            
            patch_id += 1 # Increment patch ID for the next patch
        else:
            # If we couldn't form a full patch of 'cells_per_patch' cells (e.g., not enough remaining cells),
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
    
    # Call the bar graph function to visualize the distribution of the created patch_indices.
    # The 'mask' here ensures the bar graph considers only the cells that were eligible.
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    # Return the full labels array, the list of patch index arrays, and the representative lat/lons
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def patchify_by_lon_spilldown(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40.0, seed = 42):
    """
    Creates spatially coherent patches by grouping cells primarily by longitude
    and then spilling downward in latitude. Each patch will contain exactly 'cells_per_patch' cells.

    Parameters
    ----------
    latCell : np.ndarray
        1D array of cell latitudes.
    lonCell : np.ndarray
        1D array of cell longitudes.
    cells_per_patch : int
        Exact number of cells per patch.
    num_patches : int
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
        Array of shape (num_patches, 2) containing (latitude, longitude) for one
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

        # If the current patch has reached the desired size 'cells_per_patch'
        if len(current_patch_indices) == cells_per_patch:
            # Finalize the patch
            patch_indices.append(np.array(current_patch_indices, dtype=int))
            labels_full[current_patch_indices] = patch_id # Assign patch ID to these cells
            patch_latlons.append(current_patch_first_latlon) # Store representative lat/lon
            
            patch_id += 1 # Increment patch ID for the next patch
            current_patch_indices = [] # Reset for the next patch
            current_patch_first_latlon = None # Reset the representative lat/lon

            # Stop if the maximum number of patches has been reached
            if patch_id >= num_patches:
                break
    
    # Call the bar graph function to visualize the distribution of the created patch_indices.
    # The 'mask' here ensures the bar graph considers only the cells that were eligible.
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    print("LAST PATCH SIZE: ", len(patch_indices[-1]))
    print("Contains a -1 index ", -1 in patch_indices[-1])

    # Return the full labels array, the list of patch index arrays, and the representative lat/lons
    return labels_full, patch_indices, np.array(patch_latlons), algorithm 
    
def patchify_by_latlon_spillover(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40.0, seed = 42):
    """
    Create spatially coherent patches based on sorted lat/lon without clustering.
    Walks from high latitude to low, grouping cells in longitude stripes and spilling over as needed.

    Parameters
    ----------
    latCell : np.ndarray
        1D array of cell latitudes.
    lonCell : np.ndarray
        1D array of cell longitudes.
    cells_per_patch : int
        Approximate number of cells per patch.
    num_patches : int
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
        Array of shape (num_patches, 2) containing (latitude, longitude) for one
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

        if len(current_patch) == cells_per_patch:
            patch_indices.append(np.array(current_patch, dtype=int))
            labels_full[current_patch] = patch_id
            patch_latlons.append(current_patch_first_latlon) # Add the lat/lon of the first cell
            patch_id += 1
            current_patch = []
            current_patch_first_latlon = None # Reset for the next patch

            if patch_id >= num_patches:
                break

    # Add a final partial patch
    if current_patch and patch_id < num_patches:
        patch_indices.append(np.array(current_patch, dtype=int))
        labels_full[current_patch] = patch_id
        # If there's a leftover patch, its first cell's lat/lon was already captured
        if current_patch_first_latlon is not None:
             patch_latlons.append(current_patch_first_latlon)
        else:
            # Fallback if somehow current_patch_first_latlon was not set for the last patch
            patch_latlons.append((latCell[current_patch[0]], lonCell[current_patch[0]]))
    
    # Flatten and relabel for visualization
    labels_full_final = np.full(len(latCell), -1)
    for i, inds in enumerate(patch_indices):
        labels_full_final[inds] = i

    print("LAST PATCH SIZE: ", len(patch_indices[-1]))
    print("Contains a -1 index ", -1 in patch_indices[-1])
    
    bar_graph_cluster_distribution(labels_full_final, mask, algorithm)
    
    return labels_full, patch_indices, np.array(patch_latlons), algorithm

def patchify_with_spillover(latCell, lonCell, cells_per_patch=256, num_patches=None, latitude_threshold=40, seed = 42, max_lat=90, step_deg=3):
    """
    Sweep latitudinal bands and form spatially cohesive patches, 
    letting underfilled patches spill into the next band.
    """

    algorithm="latitude_spillover_redo"
    
    rng = np.random.default_rng(seed)
    latCell = np.array(latCell)
    mask = latCell >= latitude_threshold
    
    patch_indices = []
    used = np.zeros(len(latCell), dtype=bool)
    carryover = []

    for band_max in np.arange(max_lat, latitude_threshold, -step_deg):
        band_min = band_max - step_deg
        band_mask = (latCell >= band_min) & (latCell < band_max) & (~used)
        band_ids = np.where(band_mask)[0]
        
        # Combine with carryover from the previous band
        candidates = np.concatenate([carryover, band_ids]).astype(int)
        rng.shuffle(candidates)

        num_full = len(candidates) // cells_per_patch
        for i in range(num_full):
            patch = candidates[i*cells_per_patch : (i+1)*cells_per_patch]
            patch_indices.append(patch)
            used[patch] = True
        
        # Save leftovers for the next band
        leftover = candidates[num_full * cells_per_patch:]
        carryover = leftover

        if num_patches is not None and len(patch_indices) >= num_patches:
            break

    # Flatten and relabel for visualization
    labels_full = np.full(len(latCell), -1)
    for i, inds in enumerate(patch_indices):
        labels_full[inds] = i
    
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    # avg latitude of each patch
    avg_lats = np.array([latCell[patch].mean() for patch in patch_indices])
    
    print("LAST PATCH SIZE: ", len(patch_indices[-1]))
    print("Contains a -1 index ", -1 in patch_indices[-1])

    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    
    return labels_full, patch_indices, patch_latlons, algorithm
  
def patchify_by_latitude_simple(latCell, lonCell, cells_per_patch=256, num_patches=None, latitude_threshold=40, seed = 42, max_lat=90, step_deg=3):
    """
    Divide cells into patches by sweeping latitude bands north → south.
    No clustering used—just chunking within each band.
    
    Parameters:
    - latCell: np.ndarray of cell latitudes
    - cells_per_patch: desired number of cells per patch
    - latitude_threshold, max_lat: bounds for latitude (degrees)
    - step_deg: width of latitude bands
    """
    
    algorithm="latitude_simple"
    
    rng = np.random.default_rng(seed)
    latCell = np.array(latCell)
    mask = latCell >= latitude_threshold
    
    patch_indices = []
    used = np.zeros(len(latCell), dtype=bool)

    for band_max in np.arange(max_lat, latitude_threshold, -step_deg):
        band_min = band_max - step_deg
        band_mask = (latCell >= band_min) & (latCell < band_max) & (~used)
        band_ids = np.where(band_mask)[0]

        if len(band_ids) < cells_per_patch:
            continue  # skip too-small bands

        rng.shuffle(band_ids)  # randomize order

        for i in range(0, len(band_ids) - cells_per_patch + 1, cells_per_patch):
            patch = band_ids[i:i+cells_per_patch]
            patch_indices.append(patch)
            used[patch] = True

    # Flatten and relabel for visualization
    labels_full = np.full(len(latCell), -1)
    for i, inds in enumerate(patch_indices):
        labels_full[inds] = i

    print("LAST PATCH SIZE: ", len(patch_indices[-1]))
    
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])

    return labels_full, patch_indices, patch_latlons, algorithm

def patchify_by_latitude(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed = 42, max_lat=90, step_deg=3):
    """
    Group mesh cells into patches from north pole southward using latitude bands.
    """
    algorithm="latitude_neighbors"
    
    latCell = np.array(latCell)
    lonCell = np.array(lonCell)
    mask = latCell >= latitude_threshold
    
    coords = latlon_to_xyz(latCell, lonCell)
    
    rng = np.random.default_rng(seed)
    used = np.zeros(len(latCell), dtype=bool)
    patch_indices = []
    
    for band_max in np.arange(max_lat, latitude_threshold, -step_deg):
        band_min = band_max - step_deg
        band_mask = (latCell >= band_min) & (latCell < band_max) & (~used)
        band_indices = np.where(band_mask)[0]

        if len(band_indices) < cells_per_patch:
            continue

        band_coords = coords[band_indices]

        # Fit kNN model
        knn = NearestNeighbors(n_neighbors=cells_per_patch)
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

    print("LAST PATCH SIZE: ", len(patch_indices[-1]))

    bar_graph_cluster_distribution(labels_full, mask, algorithm)
    
    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    
    return labels_full, patch_indices, patch_latlons, algorithm

def grow_patch_improved(seed_idx, cellsOnCell, visited, cells_per_patch, mask=None):
    """
    Grow a patch starting from seed_idx using mesh adjacency.
    This method is used in "build_patches_from_seeds_bfs_basic"
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

    while frontier and len(patch) < cells_per_patch:
        current = frontier.popleft() # Use popleft for BFS behavior

        neighbors = cellsOnCell[current]
        for neighbor in neighbors:
            if neighbor == -1:
                continue  # Skip invalid neighbors
            if neighbor in visited:
                continue
            if mask is not None and not mask[neighbor]:
                continue  # Skip masked-out cells

            # If adding this neighbor would exceed cells_per_patch, don't add it to frontier
            # (but still mark as visited to prevent future patches from claiming it in this round)
            if len(patch) < cells_per_patch: # Only add to patch if there's space
                patch.add(neighbor)
                visited.add(neighbor)
                frontier.append(neighbor) # Add to frontier only if it was added to patch
            else:
                # If patch is already full, just mark this neighbor as visited
                # if we want to prevent other patches from immediately taking it
                # For strict cells_per_patch, we just stop growing this patch.
                visited.add(neighbor) # Still mark as visited to prevent re-seeding / re-visiting immediately

    return list(patch)  # Return as list for consistency

def build_patches_from_seeds_improved_padded(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed=42, cellsOnCell=None, pad_to_exact_size=True):
    algorithm = "breadth_first_improved_padded" if pad_to_exact_size else "breadth_first_improved"

    rng = np.random.default_rng(seed)
    nCells = cellsOnCell.shape[0]

    # Filter valid candidates
    mask = latCell >= latitude_threshold

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
        current_patch_cells = grow_patch_improved(seed_idx, cellsOnCell, visited, cells_per_patch, mask) # visited is modified by grow_patch

        if len(current_patch_cells) == cells_per_patch:
            # Successfully formed a full patch
            patch_indices.append(np.array(current_patch_cells, dtype=int))
            patch_id = len(patch_indices) - 1
            labels_full[current_patch_cells] = patch_id
        elif len(current_patch_cells) > 0 and pad_to_exact_size:
            # Undersized patch that we need to pad
            padded_patch = np.full(cells_per_patch, -2, dtype=int) # -2 as dummy for padding
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

        if len(patch_indices) >= num_patches:
            break # Stop if max number of patches is reached

    # Note: For bar_graph_cluster_distribution, use labels_full and the original mask
    # as it counts actual assigned cells, not padded ones.
    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])

    return labels_full, patch_indices, patch_latlons, algorithm

    
def grow_patch(seed_idx, cellsOnCell, visited, cells_per_patch, mask=None):
    """
    Grow a patch starting from seed_idx using mesh adjacency.
    This method is used in "build_patches_from_seeds_bfs_basic"
    """
    patch = set()
    frontier = [seed_idx]
    visited.add(seed_idx)

    while frontier and len(patch) < cells_per_patch:
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

    
def build_patches_from_seeds_bfs_basic(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed=42, cellsOnCell=None):

    algorithm="breadth_first"
    
    rng = np.random.default_rng(seed)
    nCells = cellsOnCell.shape[0]

    # Filter valid candidates
    mask = latCell >= latitude_threshold

    # Precompute valid indices to sample seeds from
    candidate_seeds = np.where(mask)[0]
    rng.shuffle(candidate_seeds)

    visited = set()
    patch_indices = []

    for seed_idx in candidate_seeds:
        if seed_idx in visited:
            continue

        patch = grow_patch(seed_idx, cellsOnCell, visited, cells_per_patch, mask)

        if len(patch) < cells_per_patch:
            continue  # Skip undersized patches

        patch_indices.append(patch)
        if len(patch_indices) >= num_patches:
            break

    labels_full = np.full(cellsOnCell.shape[0], -1, dtype=int)
    for i, patch in enumerate(patch_indices):
        labels_full[patch] = i

    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    
    return labels_full, patch_indices, np.array(patch_latlons), algorithm 

def compute_agglomerative_patches(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed = 42, n_neighbors=6):

    algorithm="agglomerative"
    
    mask = latCell >= latitude_threshold
    coords = latlon_to_xyz(latCell[mask], lonCell[mask])
    
    # Create spatial connectivity graph
    connectivity = kneighbors_graph(coords, n_neighbors=n_neighbors, include_self=False)
    
    # Cluster
    agg = AgglomerativeClustering(
        n_clusters=num_patches,
        connectivity=connectivity,
        linkage='ward'
    )
    labels = agg.fit_predict(coords)
    
    # Insert back into full array
    labels_full = np.full(latCell.shape, -1, dtype=int)
    labels_full[mask] = labels

    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    patch_indices = [np.where(labels_full == i)[0] for i in range(num_patches)]
    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    
    return labels_full, patch_indices, patch_latlons, algorithm

def compute_disjoint_knn_patches(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed=42):
    algorithm="knn_disjoint"
    
    mask = latCell >= latitude_threshold
    
    lat_filtered = latCell[mask]
    lon_filtered = lonCell[mask]
    coords_xyz = latlon_to_xyz(lat_filtered, lon_filtered)
    original_indices = np.nonzero(mask)[0]

    knn = NearestNeighbors(n_neighbors=cells_per_patch * 3, algorithm='auto')  # oversample neighbors
    knn.fit(coords_xyz)

    rng = np.random.default_rng(seed)
    center_ids = rng.choice(len(lat_filtered), size=num_patches, replace=False)

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

    patch_indices = [original_indices[labels_masked == i] for i in range(num_patches) if np.any(labels_masked == i)]
    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    
    return labels_full, patch_indices, patch_latlons, algorithm


def compute_knn_patches(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed=42):
    """
    Computes patches based on k-Nearest Neighbors, selecting random centroids
    and forming patches from their nearest neighbors. Patches can overlap.
    """
    algorithm = "knn_basic"
    
    mask = latCell >= latitude_threshold # Changed to >= for consistency with other functions
    lat_filtered = latCell[mask]
    lon_filtered = lonCell[mask]
    coords_xyz = latlon_to_xyz(lat_filtered, lon_filtered)
    original_indices = np.nonzero(mask)[0]

    knn = NearestNeighbors(n_neighbors=cells_per_patch, algorithm='auto')
    knn.fit(coords_xyz)

    rng = np.random.default_rng(seed)
    
    # Select random centroids from the filtered cells
    center_ids_in_filtered = rng.choice(len(lat_filtered), size=num_patches, replace=False)

    labels_masked = np.full(len(lat_filtered), -1, dtype=int)
    patch_indices = []
    patch_latlons = []

    for i, cid_in_filtered in enumerate(center_ids_in_filtered):
        # Find the k-nearest neighbors to this centroid
        distances, neighbors_in_filtered = knn.kneighbors([coords_xyz[cid_in_filtered]])
        
        # Get global indices of these neighbors
        patch_global_indices = original_indices[neighbors_in_filtered[0]]
        
        # Append the NumPy array of indices to the list
        patch_indices.append(patch_global_indices)
        labels_masked[neighbors_in_filtered[0]] = i # Assign label to these cells in the masked array
        
        # Store lat/lon of the first cell in the patch (or the centroid itself)
        patch_latlons.append((latCell[patch_global_indices[0]], lonCell[patch_global_indices[0]]))

    labels_full = np.full(len(latCell), -1, dtype=int)
    labels_full[mask] = labels_masked

    bar_graph_cluster_distribution(labels_full, mask, algorithm)

    return labels_full, patch_indices, np.array(patch_latlons), algorithm
    
def cluster_patches_kmeans(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed=42):
    """ Take lat and lon values from only 50 degrees north and cluster them into patches of 256
    cells per patch for 210 patches and 53973 mesh cells. Returns an array of all 465,044 mesh cells
    where -1 indicates that there was no cluster. """

    algorithm="kmeans"
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell >= latitude_threshold           # Boolean array, True for 53973 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask))
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f))  # shape (53973, 2)
    
                                      # 53973 ÷ 256
    kmeans = KMeans(
        n_clusters=num_patches,
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
    
    patch_indices = [np.where(labels_full == i)[0] for i in range(num_patches)]
    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    
    return labels_full, patch_indices, patch_latlons, algorithm

def get_rows_of_patches(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed = 42):

    algorithm="row_by_row"
    
    mask = latCell >= latitude_threshold
    indices = np.where(mask)[0] # Get original indices of cells that meet the criteria

    print("Number of cells considered for patching:", len(indices))

    labels_full = np.full(latCell.shape, -1, dtype=int)
    bucket = 0

    for i, index in enumerate(indices):
        labels_full[index] = bucket # Assign current cell to the current bucket

        # Increment bucket (patch ID) every 'cells_per_patch' cells
        if (i + 1) % cells_per_patch == 0:
            bucket += 1
            if bucket >= num_patches:
                break
    
    bar_graph_cluster_distribution(labels_full, mask, algorithm)
    
    # The number of patches is determined by the max assigned bucket ID + 1.
    # np.max(labels_full) might be -1 if no patches were formed, so handle this.
    max_bucket_id = np.max(labels_full)
    if max_bucket_id == -1: # No patches were formed
        patch_indices = []
    else:
        patch_indices = [np.where(labels_full == i)[0] for i in range(max_bucket_id + 1)]

    print("LAST PATCH SIZE: ", len(patch_indices[-1]))
    print("Contains a -1 index ", -1 in patch_indices[-1])
    
    # Check if patch_indices is empty before attempting to access patch[0]
    if patch_indices: # If the list of patches is not empty
        patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    else:
        patch_latlons = np.empty((0, 2)) # Return an empty array of shape (0,2) if no patches


    
    return labels_full, patch_indices, patch_latlons, algorithm
    
def get_clusters_dbscan(latCell, lonCell, cells_per_patch=256, num_patches=210, latitude_threshold=40, seed=42):

    algorithm="dbscan"
    
    # --- 1.  Mask for ≥50 ° N ------------------------------------
    mask = latCell >= latitude_threshold          # Boolean array, True for 53973 rows
    lat_f = np.radians(latCell[mask])
    lon_f = np.radians(lonCell[mask])
    print("Non-zeroes", np.count_nonzero(mask)) # prints 53973
    
    # --- 2.  Stack into (n_samples, 2) & run K-means --------------
    coords = np.column_stack((lat_f, lon_f)) # shape (53973, 2)
    print("Shape of coords", coords.shape)

    # make an elbow plot to see the best value for eps
    
    neighbors = NearestNeighbors(n_neighbors=cells_per_patch, algorithm='ball_tree', metric='haversine')
    neighbors_fit = neighbors.fit(coords)
    distances, indices = neighbors_fit.kneighbors(coords)

    distances = np.sort(distances, axis=0)
    distances = distances[:, cells_per_patch-1]  # index 48 = 49th nearest
    plt.plot(distances)
    plt.savefig("elbow_plot.png")
    plt.close()

    # The elbow plot has an elbow around 0.06
    # DBSCAN parameters
    
    eps = 0.06  # Epsilon (radius)
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
    patch_latlons = np.array([(latCell[patch[0]], lonCell[patch[0]]) for patch in patch_indices])
    
    return labels_full, patch_indices, patch_latlons, algorithm
