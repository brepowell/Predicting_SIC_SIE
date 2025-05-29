# Author:   Breanna Powell
# Date:     07/02/2024

##########
# TO RUN #
##########

# Use the config.py file to specify max latitude, max longitude, file paths, etc.
# Ensure that you are looking for a variable that exists in the output file

import numpy as np

import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt     # For plotting
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import get_cmap

# Cartopy for map features, like land and ocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from config import *

def make_circle():
    """ Use this with Cartopy to make a circular map of the globe, 
    rather than a rectangular map. """
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    return mpath.Path(verts * radius + center)

def add_map_features(my_map, oceanFeature=OCEANFEATURE, landFeature=LANDFEATURE, grid=GRIDON, coastlines=COASTLINES):
    """ Set optional features on the map """
    if (oceanFeature == 1):
        my_map.add_feature(cfeature.OCEAN)

    if (landFeature == 1):
        my_map.add_feature(cfeature.LAND)

    if (grid == 1):
        my_map.gridlines()

    if (coastlines == 1):
        my_map.coastlines()

def add_polar_labels(ax, hemisphere='north'):
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.ylabels_left = False
    gl.ylabels_right = False
    gl.xlabel_style = {'size': 8}

    # Enable longitude labels (around the circle)
    if hemisphere == 'north':
        gl.xlabels_top = True  # show on top
    elif hemisphere == 'south':
        gl.xlabels_bottom = True  # show on bottom

def generate_axes_north_pole():
    """ Return a figure and axes (map) to use for plotting data for the North Pole. """

    fig = plt.figure(figsize=[5, 5]) #north pole only
    
    # Define projections for each map.
    map_projection_north = ccrs.NorthPolarStereo(central_longitude=270, globe=None)
    
    # Create an axes for a map of the Arctic on the figure
    northMap = fig.add_subplot(1, 1, 1, projection=map_projection_north)
    return fig, northMap

def generate_axes_north_and_south_pole():
    """ Return a figure and axes (maps) to use for plotting data for the North and South Poles. """
    fig = plt.figure(figsize=[10, 5]) #both north and south pole
    
    # Define projections for each map.
    map_projection_north = ccrs.NorthPolarStereo(central_longitude=270, globe=None)
    map_projection_south = ccrs.SouthPolarStereo(central_longitude=0, globe=None)

    # Create the two maps as subplots of the figure.
    northMap = fig.add_subplot(1, 2, 1, projection=map_projection_north)
    southMap = fig.add_subplot(1, 2, 2, projection=map_projection_south)

    return fig, northMap, southMap
    
def generate_axes_all_projections():
    """Return a figure and six axes with different projections in a 3x2 layout."""

    fig = plt.figure(figsize=(12, 15))  # Taller to accommodate more rows

    # Define projections
    north_proj = ccrs.NorthPolarStereo(central_longitude=270)
    south_proj = ccrs.SouthPolarStereo(central_longitude=90)
    #plate_proj = ccrs.PlateCarree()
    robinson_proj = ccrs.Robinson()
    ortho1_proj = ccrs.Orthographic(central_longitude=0, central_latitude=0) # changed latitude from 20
    ortho2_proj = ccrs.Orthographic(central_longitude=-90, central_latitude=0)
    ortho3_proj = ccrs.Orthographic(central_longitude=180, central_latitude=0)
    

    # Subplot layout:
    # 1 | 2
    # 3 | 4
    # 5 | 6
    ax1 = fig.add_subplot(3, 2, 1, projection=north_proj)
    ax2 = fig.add_subplot(3, 2, 2, projection=south_proj)
    ax3 = fig.add_subplot(3, 2, 3, projection=robinson_proj)
    ax4 = fig.add_subplot(3, 2, 4, projection=ortho1_proj)
    ax5 = fig.add_subplot(3, 2, 5, projection=ortho2_proj)
    ax6 = fig.add_subplot(3, 2, 6, projection=ortho3_proj)

    # Set titles
    ax1.set_title("North Polar Stereo")
    ax2.set_title("South Polar Stereo")
    ax3.set_title("Robinson Projection of Earth")
    ax4.set_title("Orthographic at 0 longitude")
    ax5.set_title("Orthographic at -90 longitude")
    ax6.set_title("Orthographic at 180 longitude")

    plt.tight_layout()
    return fig, [ax1, ax2, ax3, ax4, ax5, ax6]


def map_hemisphere_north(latCell, lonCell, variableToPlot1D, title, hemisphereMap, dot_size=DOT_SIZE):
    """ Map the northern hemisphere onto a matplotlib figure. 
    This requires latCell and lonCell to be filled by a mesh file.
    It also requires variableToPlot1D to be filled by an output .nc file. 
    Returns a scatter plot. """

    indices = np.where(latCell > LAT_LIMIT)     # Only capture points between the lat limit and the pole.
    
    norm=mpl.colors.Normalize(VMIN, VMAX)
    sc = hemisphereMap.scatter(lonCell[indices], latCell[indices], 
                               c=variableToPlot1D[indices], cmap='bwr', 
                               s=dot_size, transform=ccrs.PlateCarree(),
                               norm=norm)
    hemisphereMap.set_title(title)
    hemisphereMap.axis('off')

    return sc

def map_hemisphere_southern(latCell, lonCell, variableToPlot1D, title, hemisphereMap, dot_size=DOT_SIZE):
    """ Map one hemisphere onto a matplotlib figure. 
    You do not need to include the minus sign for lower latitudes. 
    This requires latCell and lonCell to be filled by a mesh file.
    It also requires variableToPlot1D to be filled by an output .nc file. 
    Returns a scatter plot. """

    indices = np.where(latCell < -LAT_LIMIT)    # Only capture points between the lat limit and the pole.
    
    norm=mpl.colors.Normalize(VMIN, VMAX)
    sc = hemisphereMap.scatter(lonCell[indices], latCell[indices], 
                               c=variableToPlot1D[indices], cmap='bwr', 
                               s=dot_size, transform=ccrs.PlateCarree(),
                               norm=norm)
    hemisphereMap.set_title(title)
    hemisphereMap.axis('off')

    return sc

def map_lats_lons_one_color(fig, latCell, lonCell, hemisphereMap, dot_size=DOT_SIZE):
    """ Map the northern hemisphere onto a matplotlib figure. 
    This requires latCell and lonCell to be filled by a mesh file.
    It also requires variableToPlot1D to be filled by an output .nc file. 
    Returns a scatter plot. """

    indices = np.where(latCell > LAT_LIMIT)     # Only capture points between the lat limit and the pole.

    norm=mpl.colors.Normalize(VMIN, VMAX)

    # Adjust layout
    fig.subplots_adjust(bottom=0.05, top=0.85, left=0.04, right=0.95, wspace=0.02)
    hemisphereMap.set_extent([MINLONGITUDE, MAXLONGITUDE, LAT_LIMIT, NORTHPOLE], ccrs.PlateCarree())
    add_map_features(hemisphereMap)
    hemisphereMap.set_boundary(make_circle(), transform=hemisphereMap.transAxes)
    
    sc = hemisphereMap.scatter(lonCell[indices], latCell[indices],
                               s=dot_size, color = 'hotpink', transform=ccrs.PlateCarree(),
                               norm=norm)
    hemisphereMap.set_title("Mesh in one color")
    hemisphereMap.axis('off')

    plt.suptitle("Mesh", size="x-large", fontweight="bold")

    # Save the maps as an image
    plt.savefig("mesh_pink.png")

    plt.close(fig)

    return sc

def map_lats_lons_gradient_by_index(fig, latCell, lonCell, hemisphereMap, dot_size=DOT_SIZE):
    """ Map points with a gradient from white to dark, scaled across all cells. """

    # Create global index array based on all cells
    all_indices = np.arange(len(latCell))

    # Normalize over the full range
    norm = mpl.colors.Normalize(vmin=0, vmax=len(latCell) - 1)

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("white_to_dark", ["white", "navy"])

    # Filter the points to plot
    mask = latCell > LAT_LIMIT
    filtered_lat = latCell[mask]
    filtered_lon = lonCell[mask]
    filtered_indices = all_indices[mask]  # These are the indices to be color-mapped

    # Adjust layout
    fig.subplots_adjust(bottom=0.25, top=0.85, left=0.04, right=0.95, wspace=0.02)
    hemisphereMap.set_extent([MINLONGITUDE, MAXLONGITUDE, LAT_LIMIT, NORTHPOLE], ccrs.PlateCarree())
    add_map_features(hemisphereMap)
    hemisphereMap.set_boundary(make_circle(), transform=hemisphereMap.transAxes)

    # Plot only the filtered points, but use color values from the full gradient
    sc = hemisphereMap.scatter(filtered_lon, filtered_lat,
                               s=dot_size,
                               c=filtered_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    # Colorbar setup (based on the full index range)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(all_indices)
    
    # Create colorbar for the whole figure 
    cbar = fig.colorbar(sm, ax=[hemisphereMap],
                        orientation='horizontal',  # or 'vertical'
                        fraction=0.05,  # size of the colorbar
                        pad=0.08,       # space between colorbar and subplots
                        shrink=0.8,     # shrink the bar to fit nicely
                        aspect=30,      # width of colorbar
                        location='bottom',  # try 'bottom' or 'right'
                        label='Index (all cells)')
        
    hemisphereMap.set_title("Mesh in a gradient")
    hemisphereMap.axis('off')
    plt.suptitle("Mesh", size="x-large", fontweight="bold")
    plt.savefig("mesh_color.png")
    plt.close(fig)

    return sc

def map_all_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap, dot_size=DOT_SIZE):
    """ Map points with a gradient from white to dark, scaled across all cells. """

    # Create global index array based on all cells
    all_indices = np.arange(len(latCell))

    # Normalize over the full range
    norm = mpl.colors.Normalize(vmin=0, vmax=len(latCell) - 1)

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("white_to_dark", ["white", "navy"])

    # Adjust the margins around the plots (as a fraction of the width or height).
    fig.subplots_adjust(bottom=0.05, top=0.85, left=0.04, right=0.95, wspace=0.02)
    
    northMap.set_extent([MINLONGITUDE, MAXLONGITUDE,  LAT_LIMIT, NORTHPOLE], ccrs.PlateCarree())
    southMap.set_extent([MINLONGITUDE, MAXLONGITUDE, -LAT_LIMIT, SOUTHPOLE], ccrs.PlateCarree())
    
    # Add map features, like landFeature and oceanFeature.
    add_map_features(northMap)
    add_map_features(southMap)
    add_map_features(plateCar)
    add_map_features(orthoMap)
    add_map_features(robinMap)
    add_map_features(rotPolMap)

    # Crop the map to be round instead of rectangular.
    northMap.set_boundary(make_circle(), transform=northMap.transAxes)
    southMap.set_boundary(make_circle(), transform=southMap.transAxes)
    #orthoPole.set_boundary(make_circle(), transform=orthoPole.transAxes)

    add_polar_labels(northMap, hemisphere='north')
    add_polar_labels(southMap, hemisphere='south')

    # Plot the points
    northPoleScatter = northMap.scatter(lonCell, latCell,
                               s=DOT_SIZE,
                               c=all_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    southPoleScatter = southMap.scatter(lonCell, latCell,
                               s=DOT_SIZE,
                               c=all_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    plateCarScatter = plateCar.scatter(lonCell, latCell,
                               s=DOT_SIZE,
                               c=all_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    orthoMapScatter = orthoMap.scatter(lonCell, latCell,
                               s=DOT_SIZE,
                               c=all_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    robinMapScatter = robinMap.scatter(lonCell, latCell,
                               s=DOT_SIZE,
                               c=all_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())
    rotPolMapScatter = rotPolMap.scatter(lonCell, latCell,
                               s=DOT_SIZE,
                               c=all_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    # Colorbar setup (based on the full index range)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(all_indices)
    
    # Create colorbar for the whole figure 
    cbar = fig.colorbar(sm, ax=[northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap],
                        orientation='horizontal',  # or 'vertical'
                        fraction=0.05,  # size of the colorbar
                        pad=0.08,       # space between colorbar and subplots
                        shrink=0.8,     # shrink the bar to fit nicely
                        aspect=30,      # width of colorbar
                        location='bottom',  # try 'bottom' or 'right'
                        label='Index (all cells)')
    
    plt.suptitle("Mesh", size="x-large", fontweight="bold")
    plt.savefig("mesh_color_all.png")
    plt.close(fig)
    
    return northPoleScatter, southPoleScatter, plateCarScatter, orthoMapScatter, robinMapScatter, rotPolMapScatter

def map_all_lats_lons_binned_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap, dot_size=DOT_SIZE):
    """ Map points with discrete color bins based on index range (~50k per bin). """

    # Create global index array
    all_indices = np.arange(len(latCell))

    # Define bin size and calculate boundaries
    bin_size = 50000
    max_index = len(latCell)
    boundaries = list(range(0, max_index, bin_size)) + [max_index]  # Ensure last bin includes all
    num_bins = len(boundaries) - 1

    # Generate distinct colors using a perceptually uniform colormap
    base_cmap = get_cmap("tab20")  # or try "Set3", "nipy_spectral", etc.
    colors = [base_cmap(i / num_bins) for i in range(num_bins)]
    
    # Assign each index to a bin
    bin_indices = np.digitize(all_indices, boundaries) - 1  # subtract 1 to index into `colors`
    scatter_colors = [colors[i] for i in bin_indices]

    # Set up discrete colormap and norm
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    # Adjust margins
    fig.subplots_adjust(bottom=0.05, top=0.85, left=0.04, right=0.95, wspace=0.02, hspace=0.15)

    # Set map extents and features
    for ax, extent in zip(
        [northMap, southMap],
        [[LAT_LIMIT, NORTHPOLE], [-LAT_LIMIT, SOUTHPOLE]]
    ):
        ax.set_extent([MINLONGITUDE, MAXLONGITUDE] + extent, ccrs.PlateCarree())
        ax.set_boundary(make_circle(), transform=ax.transAxes)

    for ax in [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]:
        add_map_features(ax)

    add_polar_labels(northMap, hemisphere='north')
    add_polar_labels(southMap, hemisphere='south')

    # Plot scatter on all maps
    scatters = []
    for ax in [northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap]:
        sc = ax.scatter(lonCell, latCell,
                        s=dot_size,
                        c=scatter_colors,
                        cmap=cmap,
                        norm=norm,
                        transform=ccrs.PlateCarree())
        scatters.append(sc)

    # Colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                        ax=[northMap, southMap, plateCar, orthoMap, robinMap, rotPolMap],
                        orientation='horizontal',
                        fraction=0.05,
                        pad=0.08,
                        shrink=0.8,
                        aspect=30,
                        location='bottom')

    # Center ticks and dynamic labels
    bin_centers = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(num_bins)]
    cbar.set_ticks(bin_centers)

    def format_tick_label(start, end):
        return f"{start//1000}k–{(end-1)//1000}k" if end < max_index else f"{start//1000}k+"

    tick_labels = [format_tick_label(boundaries[i], boundaries[i+1]) for i in range(num_bins)]
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Index Ranges')

    # Final touches
    plt.suptitle("Mesh by Index Ranges (~50k)", size="x-large", fontweight="bold")
    plt.savefig("mesh_color_all_binned.png")
    plt.close(fig)

    return tuple(scatters)


def map_patches_by_index(fig, latCell, lonCell, patch_indices, hemisphereMap, dot_size=DOT_SIZE):
    """ Map points with a gradient from white to dark, scaled across all cells. """

    # Create global index array based on all cells
    all_indices = np.arange(len(latCell))

    # Normalize over the full range
    norm = mpl.colors.Normalize(vmin=0, vmax=patch_indices.max())

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("white_to_dark", ["white", "navy"])

    # Filter the points to plot
    mask = latCell > 50

    # Adjust layout
    fig.subplots_adjust(bottom=0.25, top=0.85, left=0.04, right=0.95, wspace=0.02)
    hemisphereMap.set_extent([MINLONGITUDE, MAXLONGITUDE, LAT_LIMIT, NORTHPOLE], ccrs.PlateCarree())
    add_map_features(hemisphereMap)
    hemisphereMap.set_boundary(make_circle(), transform=hemisphereMap.transAxes)

    # Plot only the filtered points, but use color values from the full gradient
    sc = hemisphereMap.scatter(lonCell[mask], latCell[mask],
                               s=dot_size,
                               c=patch_indices[mask],
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    # Colorbar setup (based on the full index range)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(all_indices)
    
    # Create colorbar for the whole figure 
    cbar = fig.colorbar(sm, ax=[hemisphereMap],
                        orientation='horizontal',  # or 'vertical'
                        fraction=0.05,  # size of the colorbar
                        pad=0.08,       # space between colorbar and subplots
                        shrink=0.8,     # shrink the bar to fit nicely
                        aspect=30,      # width of colorbar
                        location='bottom',  # try 'bottom' or 'right'
                        label='Index (all cells)')
        
    hemisphereMap.set_title("Mesh patches in a gradient")
    hemisphereMap.axis('off')
    plt.suptitle("Mesh", size="x-large", fontweight="bold")
    plt.savefig("mesh_patches.png")
    plt.close(fig)

    return sc

PATCH_SIZE = 49          # 49 cells per patch

def map_patches_by_index_binned(fig,
                                latCell, lonCell,            # 1-D numpy arrays
                                patch_indices,               # 0 … len(latCell)-1
                                hemisphereMap,
                                dot_size=DOT_SIZE):
    """Plot every 49-cell patch in its own colour (727 patches total)."""

    # --------- choose the cells you really want to see ----------
    mask = latCell > LAT_LIMIT          # keep everything else the same

    # Recompute patch_id for valid cells only
    patch_id = patch_indices[mask]
    
    # Determine number of unique patches
    num_bins = patch_id.max() + 1
    
    # Generate colors
    base_cmap = get_cmap("flag")  # Easier to see stripes
    colors = [base_cmap(i / num_bins) for i in range(num_bins)]
    
    # Assign color per bin
    bin_indices = np.digitize(patch_id, bins=np.arange(num_bins+1)) - 1
    scatter_colors = [colors[i] for i in bin_indices]
    cmap = ListedColormap(colors)

    # --------- cartopy housekeeping ----------------------------
    fig.subplots_adjust(bottom=0.07, top=0.85,
                        left=0.04, right=0.95,
                        wspace=0.02, hspace=0.12)

    hemisphereMap.set_extent([MINLONGITUDE, MAXLONGITUDE,
                              LAT_LIMIT, NORTHPOLE],
                              ccrs.PlateCarree())
    add_map_features(hemisphereMap)
    hemisphereMap.set_boundary(make_circle(), transform=hemisphereMap.transAxes)
    add_polar_labels(hemisphereMap, hemisphere='north')

    # --------- scatter plot ------------------------------------
    sc = hemisphereMap.scatter(lonCell[mask], latCell[mask],
                               s=dot_size,
                               c=scatter_colors,           # <-- patch IDs drive colour
                               cmap=cmap,
                               transform=ccrs.PlateCarree())

    # --------- (optional) colour-bar ----------------------------
    sm = mpl.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[hemisphereMap],
                        orientation='horizontal',
                        fraction=0.05, pad=0.08,
                        shrink=0.8, aspect=30,
                        location='bottom')
    cbar.set_label('Patch ID (49 cells each)')
    # 727 ticks are illegible – remove them:
    cbar.set_ticks([])

    # --------- titles & save -----------------------------------
    hemisphereMap.set_title("Mesh – 727 patches (49 cells each)")
    hemisphereMap.axis('off')
    plt.suptitle("Mesh by Patch", size="x-large", fontweight="bold")
    plt.savefig("mesh_patches_binned.png")
    plt.close(fig)

    return sc

    
def generate_maps_north_and_south(fig, northMap, southMap, latCell, lonCell, variableToPlot1D, mapImageFileName, 
                                  colorBarOn=COLORBARON, grid=GRIDON,
                                  oceanFeature=OCEANFEATURE, landFeature=LANDFEATURE, 
                                  coastlines=COASTLINES, dot_size=DOT_SIZE, textBoxString=""):
    """ Generate 2 maps; one of the north pole and one of the south pole. """

    # Adjust the margins around the plots (as a fraction of the width or height).
    fig.subplots_adjust(bottom=0.05, top=0.85, left=0.04, right=0.95, wspace=0.02)

    # Set your viewpoint (the bounding box for what you will see).
    # You want to see the full range of longitude values, since this is a polar plot.
    # The range for the latitudes should be from your latitude limit (i.e. 50 degrees or -50 to the pole at 90 or -90).
    
    northMap.set_extent([MINLONGITUDE, MAXLONGITUDE,  LAT_LIMIT, NORTHPOLE], ccrs.PlateCarree())
    southMap.set_extent([MINLONGITUDE, MAXLONGITUDE, -LAT_LIMIT, SOUTHPOLE], ccrs.PlateCarree())

    # Add map features, like landFeature and oceanFeature.
    add_map_features(northMap, oceanFeature, landFeature, grid, coastlines)
    add_map_features(southMap, oceanFeature, landFeature, grid, coastlines)

    # Crop the map to be round instead of rectangular.
    northMap.set_boundary(make_circle(), transform=northMap.transAxes)
    southMap.set_boundary(make_circle(), transform=southMap.transAxes)

    # Map the 2 hemispheres.
    northPoleScatter = map_hemisphere_north(latCell, lonCell, variableToPlot1D, "Arctic Sea Ice", northMap, dot_size=dot_size)
    southPoleScatter = map_hemisphere_southern(latCell, lonCell, variableToPlot1D, "Antarctic Sea Ice", southMap, dot_size=dot_size)
    
    # Add the timestamp to the North Map on the left side
    if textBoxString != "":    
        textBox = northMap.text(0.05, 0.95, textBoxString, transform=northMap.transAxes, fontsize=14,
            verticalalignment='top', bbox=boxStyling, zorder=5)

    # Set Color Bar - make sure this is the last thing you add to the map!
    if colorBarOn:
        plt.colorbar(northPoleScatter, ax=northMap)
        plt.colorbar(southPoleScatter, ax=southMap)

    # Add the bold title at the top
    plt.suptitle(suptitle_variable_year(), size="x-large", fontweight="bold")

    # Save the maps as an image
    plt.savefig(mapImageFileName)

    plt.close(fig)

    return northPoleScatter, southPoleScatter

def generate_map_north_pole(fig, northMap, latCell, lonCell, variableToPlot1D, mapImageFileName, 
                         colorBarOn=COLORBARON, grid=GRIDON,
                         oceanFeature=OCEANFEATURE, landFeature=LANDFEATURE, 
                         coastlines=COASTLINES, dot_size=DOT_SIZE):
    """ Generate one map of the north pole. No timestamp. """

    print("--- starting to make a map of the Arctic ---")
    
    # Adjust the margins around the plots (as a fraction of the width or height).
    fig.subplots_adjust(bottom=0.05, top=0.85, left=0.04, right=0.95, wspace=0.02)

    # Set your viewpoint (the bounding box for what you will see).
    # You want to see the full range of longitude values, since this is a polar plot.
    # The range for the latitudes should be from your latitude limit (i.e. 50 degrees or -50 to the pole at 90 or -90).
    northMap.set_extent([MINLONGITUDE, MAXLONGITUDE, LAT_LIMIT, NORTHPOLE], ccrs.PlateCarree())

    # Add map features, like landFeature and oceanFeature.
    add_map_features(northMap, oceanFeature, landFeature, grid, coastlines)

    # Crop the map to be round instead of rectangular.
    northMap.set_boundary(make_circle(), transform=northMap.transAxes)

    # Map the hemisphere
    scatter = map_hemisphere_north(latCell, lonCell, variableToPlot1D, f"Arctic Sea Ice", northMap, dot_size)     # Map northern hemisphere
    
    # Set Color Bar - make sure this is the last thing you add to the map!
    if colorBarOn:
        plt.colorbar(scatter, ax=northMap)

    plt.suptitle(suptitle_variable_year(), size="x-large", fontweight="bold")

    # Save the maps as an image.
    plt.savefig(mapImageFileName)

    plt.close(fig)

    return scatter

def generate_static_map_png_file_name(file_path, day=1):    
    """Generate a PNG file name based on the .nc file name and a given day.
    
    Assumes the .nc file name contains a date, and the last two characters before 
    the extension are replaced with the day number.
    """
    name_without_extension = os.path.splitext(os.path.basename(file_path))[0][:-2]  
    return f"{name_without_extension}{day:02d}.png"