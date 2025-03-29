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

def map_lats_lons_one_color(latCell, lonCell, title, hemisphereMap, dot_size=DOT_SIZE):
    """ Map the northern hemisphere onto a matplotlib figure. 
    This requires latCell and lonCell to be filled by a mesh file.
    It also requires variableToPlot1D to be filled by an output .nc file. 
    Returns a scatter plot. """

    indices = np.where(latCell > LAT_LIMIT)     # Only capture points between the lat limit and the pole.
    
    norm=mpl.colors.Normalize(VMIN, VMAX)
    sc = hemisphereMap.scatter(lonCell[indices], latCell[indices],
                               s=dot_size, color = 'hotpink', transform=ccrs.PlateCarree(),
                               norm=norm)
    hemisphereMap.set_title(title)
    hemisphereMap.axis('off')

    return sc

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

def generate_axes_north_pole():
    """ Return a figure and axes (map) to use for plotting data for the North Pole. """

    fig = plt.figure(figsize=[5, 5]) #north pole only
    
    # Define projections for each map.
    map_projection_north = ccrs.NorthPolarStereo(central_longitude=270, globe=None)
    
    # Create an axes for a map of the Arctic on the figure
    northMap = fig.add_subplot(1, 1, 1, projection=map_projection_north)
    return fig, northMap

def generate_maps_north_and_south(fig, northMap, southMap, latCell, lonCell, variableToPlot1D, mapImageFileName, 
                                  colorBarOn=COLORBARON, grid=GRIDON,
                                  oceanFeature=OCEANFEATURE, landFeature=LANDFEATURE, 
                                  coastlines=COASTLINES, dot_size=DOT_SIZE, textBoxString=""):
    """ Generate 2 maps; one of the north pole and one of the south pole. """

    print(f"MINLONGITUDE: {MINLONGITUDE}, MAXLONGITUDE: {MAXLONGITUDE}, LAT_LIMIT: {LAT_LIMIT}, NORTHPOLE: {NORTHPOLE}")

    
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

    print(f"northPoleScatter: {northPoleScatter}")  
    print(f"southPoleScatter: {southPoleScatter}")  
    
    if textBoxString != "":
        # Add the timestamp to the North Map on the left side
        textBox = northMap.text(0.05, 0.95, textBoxString, transform=northMap.transAxes, fontsize=14,
            verticalalignment='top', bbox=boxStyling, zorder=5)

    # Set Color Bar
    if colorBarOn:
        plt.colorbar(northPoleScatter, ax=northMap)
        plt.colorbar(southPoleScatter, ax=southMap)

    # Add time textbox
    plt.suptitle(suptitle_variable_year(), size="x-large", fontweight="bold")

    # Save the maps as an image
    plt.savefig(mapImageFileName)

    #return northPoleScatter, southPoleScatter
    #return southPoleScatter

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
    
    # Set Color Bar
    if colorBarOn:
        plt.colorbar(scatter, ax=northMap)

    plt.suptitle(suptitle_variable_year(), size="x-large", fontweight="bold")

    # Save the maps as an image.
    plt.savefig(mapImageFileName)

    return scatter

def generate_static_map_png_file_name(file_path, day=1):    
    """Generate a PNG file name based on the .nc file name and a given day.
    
    Assumes the .nc file name contains a date, and the last two characters before 
    the extension are replaced with the day number.
    """
    name_without_extension = os.path.splitext(os.path.basename(file_path))[0][:-2]  
    return f"{name_without_extension}{day:02d}.png"