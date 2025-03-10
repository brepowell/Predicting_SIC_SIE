from utility import *

#runDir         = os.path.dirname(os.path.abspath(__file__))       # Get current directory path
#runDir = perlmutterpath1 # For perlmutter only
runDir = perlmutterpathDailyData


#fileName = r"/output_files/Breanna_D_test_5_nodes_1_nyears_with_fewer_nodes.mpassi.hist.am.timeSeriesStatsDaily.0001-03-01.nc"
#fileName = r"v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.2008-02-01.nc"
fileName = r"v3.LR.historical_0051.mpassi.hist.am.timeSeriesStatsDaily.2003-02-01.nc"

output = loadData(runDir, fileName)
days = getNumberOfDays(output)

# Get list of all days / time values to plot that exist in one .nc file
timeList = printDateTime(output, timeStringVariable=START_TIME_VARIABLE, days=days)
