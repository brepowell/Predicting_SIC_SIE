from config import *

def validate_suptitle_length(variable, shortName):
    if len(variable) > MAX_SUPTITLE_LENGTH:
        return shortName
    else:
        return variable

def suptitle_variable_season_year(variable=VARIABLETOPLOT, shortName=VARIABLETOPLOTSHORTNAME, season=SEASON, year=YEAR):
    """ Use this if you want the suptitle (the bold one at the top of the plot) to be for one season and year. """
    return f"{validate_suptitle_length(variable, shortName).upper()} {season.upper()} {year}"

def suptitle_variable_year(variable=VARIABLETOPLOT, shortName=VARIABLETOPLOTSHORTNAME, year=YEAR):
    """ Use this if you want the suptitle (the bold one at the top of the plot) to be for one year. """
    return f"{validate_suptitle_length(variable, shortName).upper()} {year}"

def suptitle_variable_all_time(variable=VARIABLETOPLOT, shortName=VARIABLETOPLOTSHORTNAME, season=SEASON, year=YEAR):
    """ Use this if you want the suptitle (the bold one at the top of the plot) to be for all time. """
    return f"{validate_suptitle_length(variable, shortName).upper()} ALL TIME"

def suptitle_variable_start_end(variable=VARIABLETOPLOT, shortName=VARIABLETOPLOTSHORTNAME, season=SEASON, startYear=STARTYEAR, endYear=ENDYEAR):
    return f"{validate_suptitle_length(variable, shortName).upper()} {season.upper()} {startYear} - {endYear}"

def suptitle_variable_from_E3SM(variable=VARIABLETOPLOT, shortName=VARIABLETOPLOTSHORTNAME):
    return f"{validate_suptitle_length(variable, shortName).upper()} from E3SM"
