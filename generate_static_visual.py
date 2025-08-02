from NC_FILE_PROCESSING.nc_utility_functions import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from config import *

def static_image_no_time_label(latCell, lonCell, variableToPlot1Day):
    ###################
    # ARTIC-ONLY PLOT #
    ###################
    fig, northMap = generate_axes_north_pole()

    mapImageFileName = generate_static_map_png_file_name(outputFileName, day=13)

    # Plotting with a variable
    generate_map_north_pole(fig, northMap, latCell, lonCell, variableToPlot1Day, mapImageFileName)

    print("Saving file: ", mapImageFileName)

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)
    #latCell, lonCell = load_mesh("", meshFileName)
    output = load_data(runDir, outputFileName)

    ####################################################
    # PLOTTING REGULAR E3SM OUTPUT DATA, LIKE ICE AREA #
    ####################################################
    print("Days total: ", get_number_of_days(output, keyVariableToPlot=VARIABLETOPLOT))

    variableToPlot1Day = reduce_to_one_dimension(output, keyVariableToPlot=VARIABLETOPLOT, dayNumber=13)
    
    ##############################
    # PLOTTING MY NEW.NC RESULTS #
    ##############################
    # print("nCells total: ", get_number_of_days(output, keyVariableToPlot=VARIABLETOPLOT))
    # variableToPlot1Day = output.variables[VARIABLETOPLOT][:]
    # variableToPlot1Day.ravel()
    # print(variableToPlot1Day.shape)

    static_image_no_time_label(latCell, lonCell, variableToPlot1Day)
    

if __name__ == "__main__":
    main()