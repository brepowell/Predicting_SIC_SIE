from NC_FILE_PROCESSING.nc_utility_functions import *
from MAP_ANIMATION_GENERATION.map_gen_utility_functions import *
from MAP_ANIMATION_GENERATION.map_label_utility_functions import *
from config import *

def main():
    
    # Load the mesh and data to plot.
    latCell, lonCell = load_mesh(perlmutterpathMesh)
    #latCell, lonCell = load_mesh("", meshFileName)
    output = load_data(runDir, outputFileName)

    ####################################################
    # PLOTTING REGULAR E3SM OUTPUT DATA, LIKE ICE AREA #
    ####################################################
    print("Days total: ", get_number_of_days(output, keyVariableToPlot=VARIABLETOPLOT))

    variableToPlot1Day = reduce_to_one_dimension(output, keyVariableToPlot=VARIABLETOPLOT, dayNumber=1)
    
    ##############################
    # PLOTTING MY NEW.NC RESULTS #
    ##############################
    # print("nCells total: ", get_number_of_days(output, keyVariableToPlot=VARIABLETOPLOT))
    # variableToPlot1Day = output.variables[VARIABLETOPLOT][:]
    # variableToPlot1Day.ravel()
    # print(variableToPlot1Day.shape)

    ###################
    # ARTIC-ONLY PLOT #
    ###################
    fig, northMap = generate_axes_north_pole()

    mapImageFileName = generate_static_map_png_file_name(outputFileName)

    # Plotting with a variable
    generate_map_north_pole(fig, northMap, latCell, lonCell, variableToPlot1Day, mapImageFileName)

    print("Saving file: ", mapImageFileName)

if __name__ == "__main__":
    main()