# Author:   Breanna Powell
# Date:     07/02/2024

##########
# TO RUN #
##########

from NC_FILE_PROCESSING.nc_utility_functions import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from config import *

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
