from NC_FILE_PROCESSING.nc_utility_functions import *

#runDir         = os.path.dirname(os.path.abspath(__file__))       # Get current directory path
runDir = perlmutterpath1 # For perlmutter only

# fileName   = r"\mesh_files\mpassi.IcoswISC30E3r5.20231120.nc"
#fileName = r"\mesh_files\E3SM_IcoswISC30E3r5_ICESat_Orbital_Synchronizer.nc"

# Checking the sizes for the number of grid cells per .nc file
# fileName = r"\mesh_files\DECK_Coast.nc"                     # Size in grid cells: 18157
# fileName = r"\mesh_files\E3SM_DECK_Emulator_File.nc"        # Size in grid cells: 235160
# fileName = r"\mesh_files\E3SM_V1_C_grid_Coast.nc"           # Size in grid cells: 74686
# fileName = r"\mesh_files\ICESat_Masked_E3SM.nc"             # Size in grid cells: 6220

# FILES FOR 5 OR 10 DAY SIMULATION:
# fileName = r"\mesh_files\seaice.EC30to60E2r2.210210.nc"     # Size in grid cells: 236853
# fileName = r"\output_files\Breanna_D_test_1x05_days.mpassi.hist.am.timeSeriesStatsDaily.0001-01-01.nc" # 236853
fileName = r"v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.2003-02-01.nc"


# FILES FOR 1 YEAR SIMULATION:
# fileName = r"\mesh_files\mpassi.IcoswISC30E3r5.20231120.nc" # Size in grid cells:     465044

# FILES FOR SATELLITE TRACK ANIMATION:
#fileName = r"\satellite_data_preprocessed\one_day\icesat_E3SM_spring_2008_02_22_16.nc"  # 6533

output = loadData(runDir, fileName)
printAllAvailableVariables(output)