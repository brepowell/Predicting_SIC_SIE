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

    # TODO - MAKE THIS RUN IN PARALLEL
    #for i in range(days):
    i = 0
    # Get the time for this day
    textBoxString = "Time: " + str(timeList[i])
    
    variableForOneDay = reduce_to_one_dimension(output, keyVariableToPlot=VARIABLETOPLOT, dayNumber=i)
    
    mapImageFileName = generate_static_map_png_file_name(outputFileName, day=i+1)
    
    generate_maps_north_and_south(fig, northMap, southMap, 
                                                                       latCell, lonCell, variableForOneDay, 
                                                                       mapImageFileName,
                                                                       textBoxString=textBoxString)
    print("Saved file: ", mapImageFileName)


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

    max_patch_id = patch_id.max()
    base_cmap = get_cmap("flag")
    all_colors = [base_cmap(i / (max_patch_id + 1)) for i in range(max_patch_id + 1)]
    cmap = ListedColormap(all_colors)

    frames = []

    for max_index in range(0, max_patch_id + 1, step):
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
        frame_mask = patch_id <= max_index
        sc = ax.scatter(lon_filtered[frame_mask], lat_filtered[frame_mask],
                        s=dot_size,
                        c=patch_id[frame_mask],
                        cmap=cmap,
                        vmin=0,
                        vmax=max_patch_id,
                        transform=ccrs.PlateCarree())

        ax.set_title(f"Patches 0 to {max_index}")
        ax.axis('off')
        plt.suptitle("Mesh Patches - Ascending by Patch Index", fontsize="x-large", fontweight="bold")

        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{max_index:03d}.png")
        plt.savefig(frame_path)
        plt.close(fig)

        if save_gif:
            frames.append(imageio.imread(frame_path))

    # Assemble GIF if requested
    if save_gif:
        imageio.mimsave(gif_path, frames, duration=0.2)  # duration is seconds per frame
        print(f"GIF saved to {gif_path}")
