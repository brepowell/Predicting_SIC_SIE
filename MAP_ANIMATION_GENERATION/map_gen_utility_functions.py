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

# Cartopy for map features, like land and ocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from config import *

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
    fig.subplots_adjust(bottom=0.05, top=0.85, left=0.04, right=0.95, wspace=0.02)
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
    plt.colorbar(sm, ax=hemisphereMap, orientation='vertical', label='Index (all cells)')

    hemisphereMap.set_title("Mesh in a gradient")
    hemisphereMap.axis('off')
    plt.suptitle("Mesh", size="x-large", fontweight="bold")
    plt.savefig("mesh_color.png")
    plt.close(fig)

    return sc

def map_all_lats_lons_gradient_by_index(fig, latCell, lonCell, northMap, southMap, plateCar, rotPole, dot_size=DOT_SIZE):
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
    add_map_features(rotPole)
    
    # Crop the map to be round instead of rectangular.
    northMap.set_boundary(make_circle(), transform=northMap.transAxes)
    southMap.set_boundary(make_circle(), transform=southMap.transAxes)

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

    rotPolScatter = rotPole.scatter(lonCell, latCell,
                               s=DOT_SIZE,
                               c=all_indices,
                               cmap=cmap,
                               norm=norm,
                               transform=ccrs.PlateCarree())

    # Colorbar setup (based on the full index range)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(all_indices)
    
    # Create colorbar for the whole figure 
    cbar = fig.colorbar(sm, ax=[northMap, southMap, plateCar, rotPole],
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
    
    return northPoleScatter, southPoleScatter, plateCarScatter, rotPolScatter

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
    """Return a figure and four axes with different projections in a 2x2 layout."""

    fig = plt.figure(figsize=(10, 10))  # Adjust size as needed

    # Projections
    north_proj = ccrs.NorthPolarStereo(central_longitude=270)
    south_proj = ccrs.SouthPolarStereo(central_longitude=90)
    plate_proj = ccrs.PlateCarree()
    rotated_proj = ccrs.RotatedPole(pole_longitude=180, pole_latitude=35)

    # Subplot layout:
    # 1 | 2
    # -----
    # 3 | 4

    ax1 = fig.add_subplot(2, 2, 1, projection=north_proj)
    ax2 = fig.add_subplot(2, 2, 2, projection=south_proj)
    ax3 = fig.add_subplot(2, 2, 3, projection=plate_proj)
    ax4 = fig.add_subplot(2, 2, 4, projection=rotated_proj)

    # Set titles
    ax1.set_title("North Polar Stereo")
    ax2.set_title("South Polar Stereo")
    ax3.set_title("Plate Carree")
    ax4.set_title("Rotated Pole")

    plt.tight_layout()
    return fig, [ax1, ax2, ax3, ax4]

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