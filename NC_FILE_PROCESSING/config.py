# Author:   Breanna Powell
# Date:     07/03/2024
# Use this in conjunction with e3sm_data_over_time_visualization.py and e3sm_data_practice_visualization.py
# and other files in the repo.
# Make sure to set these variables to the proper file locations.

import os
from perlmutterpath import *

####################
# NetCDF Variables #  
####################                                                                                                                            ####################

# Change these for different runs if you want to narrow down your focus
VARIABLETOPLOT     = "timeDaily_avg_iceAreaCell"   # The variable to plot from the E3SM data
#VARIABLETOPLOT      = "freeboard"                   # The variable to plot from the satellite data
#VARIABLETOPLOT      = "samplemf" # make sure to set VMAX
#VARIABLETOPLOT      = "sampleof"
#VARIABLETOPLOT       = "meanof"
#VARIABLETOPLOT       = "stdof"
#VARIABLETOPLOT      = "meanmf"

TIMESTRINGVARIABLE  = "time_string"      # Used from E3SM data
START_TIME_VARIABLE = "xtime_startDaily" # Used in making a netCDF file

#START_TIME_VARIABLE = "time_string"
END_TIME_VARIABLE   = ""
TIMEVARIABLE        = "time"
NC_TIME_COUNT_VARIABLE = "timeDaily_counter"

LATITUDEVARIABLE    = "latCell"    #latitude
LONGITUDEVARIABLE   = "lonCell"   #longitude
CELLVARIABLE        = "cell"

SEASON              = "spring"        # spring or fall
YEAR                = "2008"    # 2003 to 2008
NEW_NETCDF_FILE_NAME = f"{SEASON}_{YEAR}.nc"
#NEW_NETCDF_FILE_NAME = "ALL_SATELLITE_DATA.nc"

# Change if you want a wider or narrower view
#LAT_LIMIT       =  50  # Good wide view for the north and south poles for E3SM data
LAT_LIMIT       =  65  # More of a closeup, better for the satellite data
#LAT_LIMIT        =  80  # Extreme closeup for the freeboard of one satellite track

# Change if you want larger or smaller dots for the scatterplot
#DOT_SIZE        = 0.4  # Good for the ice area variable
#DOT_SIZE        = 7.0  # Good for satellite tracks
DOT_SIZE        = 0.1

# Change if you want to downsample the amount of data by a certain factor
DEFAULT_DOWNSAMPLE_FACTOR = 100

# Color Bar Range
VMIN = 0
#VMAX = 1      # Good for Ice Area
#VMAX = 0.7    # Good for Freeboard
#VMAX = 150    # samplemf for ALL FILES - the max is 295, but there are not many cells that go above 150 samples; 100 is too low
#VMAX = 25     # samplemf for spring 2003 - there are not many cells that go above 45 samples;
#VMAX = 15000  # sampleof for ALL FILES - the max is 46893, but there are not that many tracks that go about 15000 samples; 20000 looks ok 
#VMAX = 4000   # sampleof for spring 2003
#VMAX = 1.0   # meanof - spring 2003 to 2008
VMAX = 0.5    # stdof - spring 2003 to 2008
#VMAX = 0.9   # meanmf fall 2003 
#VMAX = 0.25  # stdof fall 2003

# Animation speed
INTERVALS = 500 # good for smaller animations, like 5 to 10 days
#INTERVALS = 250
#INTERVALS = 50 # used for year-long animation

################
#  File Paths  #
################

#runDir = ""
#runDir         = os.path.dirname(os.path.abspath(__file__))       # Get current directory path
runDir = perlmutterpath2 # For perlmutter (PM) only

# Change these for different runs if you want to grab other .nc files

#meshFileName   = r"\mesh_files\seaice.EC30to60E2r2.210210.nc"    # for 5 day and 10 day simulations
meshFileName   = r"\mesh_files\mpassi.IcoswISC30E3r5.20231120.nc"  # for satellite emulator
#meshFileName   = r"/mesh_files/mpassi.IcoswISC30E3r5.20231120.nc" # for PM Perlmutter for the 1 year mesh
#meshFileName = perlmutterpathMesh # for PM

#SYNCH_FILE_NAME = r"\mesh_files\E3SM_IcoswISC30E3r5_ICESat_Orbital_Synchronizer.nc"
SYNCH_FILE_NAME = r"/mesh_files/E3SM_IcoswISC30E3r5_ICESat_Orbital_Synchronizer.nc" #PM

outputFileName = r"/10yrFebruaryExample.nc"

#outputFileName = r"\output_files\Breanna_D_test_1x05_days.mpassi.hist.am.timeSeriesStatsDaily.0001-01-01.nc"  # 5-day Ice Area
#outputFileName = r"\output_files\Breanna_D_test_1x10_days.mpassi.hist.am.timeSeriesStatsDaily.0001-01-01.nc"  # 10-day Ice Area
#outputFileName = r"\satellite_data_preprocessed\one_day\icesat_E3SM_spring_2008_02_22_14.nc" # One Satellite Track
#outputFileName = r"/output_files/Breanna_D_test_5_nodes_1_nyears_with_fewer_nodes.mpassi.hist.am.timeSeriesStatsDaily.0001-01-01.nc" # 1-year, month 1
#outputFileName = r"\new.nc" # Satellite emulator
#outputFileName = f"/{NEW_NETCDF_FILE_NAME}" # Satellite emulator on PM

#subdirectory = "" # Use when running make_a_netCDF_file.py
#subdirectory = r"/satellite_data_preprocessed/one_month" # Satellite Track folder for one month
#subdirectory = r"/satellite_data_preprocessed/one_week" # Satellite Track folder for one week
#subdirectory = r"/satellite_data_preprocessed/one_day" # Satellite Track folder for one day
#subdirectory = r"/output_files/" # for plotting more than one output file (Use on PM Perlmutter for year simulation)
subdirectory = f"/2003_to_2008_{SEASON}/"
#subdirectory = r"/2003_to_2008_spring/"
#subdirectory = perlmutterpathDailyData

FULL_PATH = runDir + subdirectory

# Change these to save without overwriting your files
#animationFileName = f"E3SM_2003_5_months_simulation.gif"
#mapImageFileName = f"static_image.png"

animationFileName   = f"{VARIABLETOPLOT}_{SEASON}_2003_to_{YEAR}.gif"                # Should be a .gif extension
mapImageFileName    = f"{VARIABLETOPLOT}_{SEASON}_{YEAR}.png"

#mapImageFileName    = "samplemf_all_time.png"             # Should be a .png file extension

################
# Map settings #
################

boxStyling = dict(boxstyle='round', facecolor='wheat') #other options are alpha (sets transparency)

#MAP_SUPTITLE_TOP = f"{VARIABLETOPLOT.upper()} {SEASON.upper()} {YEAR}"
#MAP_SUPTITLE_TOP = f"{VARIABLETOPLOT.upper()} ALL TIME"
MAP_SUPTITLE_TOP = f"{VARIABLETOPLOT.upper()} {SEASON.upper()} 2003 - {YEAR}"
#MAP_SUPTITLE_TOP = f"{VARIABLETOPLOT.upper()} from E3SM"

# These features are on
OCEANFEATURE    = 1   
LANDFEATURE     = 1
COASTLINES      = 1
COLORBARON      = 1
GRIDON          = 1

# These features are off
# N/A

# Constants
MAXLONGITUDE    =  180
MINLONGITUDE    = -180
NORTHPOLE       =  90
SOUTHPOLE       = -90
