# Author:   Breanna Powell
# Date:     07/02/2024

##########
# TO RUN #
##########

from NC_FILE_PROCESSING.nc_utility_functions import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from config import *

import imageio.v2 as imageio  # for GIF

def generate_daily_pngs_from_one_nc_file_with_multiple_days():

    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)
    output = load_data(runDir, outputFileName)
    days = get_number_of_days(output, keyVariableToPlot=VARIABLETOPLOT)
    
    # Get list of all days / time values to plot that exist in one .nc file
    timeList = print_date_time(output, timeStringVariable=START_TIME_VARIABLE, days=days)

    fig, northMap, southMap = generate_axes_north_and_south_pole()

    # List to store the paths of the generated PNG frames
    frames_for_gif = []
    
    # Create a directory to store the frames
    output_dir = "daily_frames"
    os.makedirs(output_dir, exist_ok=True)

    # Loop through each day to generate a separate image
    for i in range(days):
        
        # Get the time for this day
        textBoxString = "Time: " + str(timeList[i])
        
        variableForOneDay = reduce_to_one_dimension(output, keyVariableToPlot=VARIABLETOPLOT, dayNumber=i)
        
        # Generate a unique file name for each frame
        mapImageFileName = os.path.join(output_dir, generate_static_map_png_file_name(outputFileName, day=i + 1))
        
        fig, northMap, southMap = generate_axes_north_and_south_pole()
        
        generate_maps_north_and_south(fig, northMap, southMap,
                                      latCell, lonCell, variableForOneDay,
                                      mapImageFileName,
                                      textBoxString=textBoxString)
        
        plt.close(fig) # Close the figure to free up memory
        
        print("Saved file: ", mapImageFileName)
        frames_for_gif.append(mapImageFileName)
    
    # After the loop, create the GIF from the saved frames
    if frames_for_gif:
        gif_path = f"daily_animation_{outputFileName}.gif" # Name of the final GIF
        print(f"Creating GIF at {gif_path}...")
        
        # Read the images and create the GIF
        images = [imageio.imread(frame) for frame in frames_for_gif]
        imageio.mimsave(gif_path, images, fps=5) # 'fps' is frames per second
        
        print(f"GIF saved to {gif_path}")
    else:
        print("No frames were generated to create the GIF.")

def animate_patch_reveal(latCell, lonCell, patch_indices, dot_size=DOT_SIZE,
                         output_dir="frames", gif_path="patch_animation.gif",
                         step=25, save_gif=True):
    """Create an animation where patch indices appear gradually."""

    os.makedirs(output_dir, exist_ok=True)

    # Filter valid cells
    mask = (latCell > LAT_LIMIT) & (patch_indices != -1)
    lat_filtered = latCell[mask]
    lon_filtered = lonCell[mask]
    patch_id = patch_indices[mask]

    if len(patch_id) == 0:
        print("No valid patches to animate after filtering. Check LAT_LIMIT and patch_indices.")
        return

    max_patch_id = patch_id.max()
    print(f"Calculated max_patch_id: {max_patch_id}")
    print(f"Unique patch_ids in filtered data: {np.unique(patch_id)}")

    base_cmap = get_cmap("flag")
    all_colors = [base_cmap(i / (max_patch_id + 1)) for i in range(max_patch_id + 1)]
    cmap = ListedColormap(all_colors)

    frames = []

    # Ensure the last frame always includes max_patch_id
    for max_index_limit in range(0, max_patch_id + step, step):
        current_max_index = min(max_index_limit, max_patch_id)
        
        fig = plt.figure(figsize=(8, 6))
        fig.subplots_adjust(bottom=0.07, top=0.85,
                            left=0.04, right=0.95,
                            wspace=0.02, hspace=0.12)
        ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=270, globe=None))
        ax.set_extent([MINLONGITUDE, MAXLONGITUDE, LAT_LIMIT, NORTHPOLE], ccrs.PlateCarree())
        add_map_features(ax)
        ax.set_boundary(make_circle(), transform=ax.transAxes)
        add_polar_labels(ax, hemisphere='north')

        # Mask to only show patches up to the current frame's limit
        frame_mask = patch_id <= current_max_index
        
        # Only plot if there are points to plot in this frame
        if np.any(frame_mask):
            sc = ax.scatter(lon_filtered[frame_mask], lat_filtered[frame_mask],
                            s=dot_size,
                            c=patch_id[frame_mask],
                            cmap=cmap,
                            vmin=0,
                            vmax=max_patch_id,
                            transform=ccrs.PlateCarree())

        ax.set_title(f"Patches 0 to {current_max_index}")
        ax.axis('off')
        plt.suptitle(gif_path, fontsize="x-large", fontweight="bold")

        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{current_max_index:03d}.png")
        plt.savefig(frame_path)
        plt.close(fig)

        if save_gif:
            frames.append(imageio.imread(frame_path))
            
        # Optional: Print progress for debugging
        print(f"Generated frame for max_index: {current_max_index}")


    # Assemble GIF if requested
    if save_gif and frames: # Ensure there are frames to save
        imageio.mimsave(gif_path, frames, duration=1.5)  # duration is seconds per frame
        print(f"GIF saved to {gif_path}")
    elif save_gif and not frames:
        print("No frames were generated for the GIF.")
