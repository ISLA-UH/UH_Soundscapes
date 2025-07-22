import os

"""
Configuration file for discovery stage
Guam Data, PEcoC CSON
# 2023-07-07 00:00 GMT  to 2024-09-30 23:59 GMT

"""

# Input directory, depends on OS.  Copy the line and edit as needed.
# Windows:
# INPUT_DIR = "C:\\path\\to\\files\\"
# Linux/Mac:
# INPUT_DIR = "/Path/to/Files"
# MAG
INPUT_DIR = "/Users/mgarces/Documents/DATA_2025/GUAM_202307/api1000"

# Absolute path for output files
RPD_DIR = "Qi/discovery/"
OUTPUT_DIR = os.path.join(INPUT_DIR, RPD_DIR)

# List all station IDs here
# Leave blank if you want to process all stations
STATIONS = []

# Microseconds to seconds conversion factor
MICROS_TO_S = 1E-6

# NOTE: BEGIN WITH FIRST HOUR OF FIRST DAY FOR TESTING
# PEcoC CSON
# 2023-07-07 00:01 GMT  to 2024-09-30 23:59 GMT
REF_HUMAN_GMT_0 = "2023-07-07 00:01 GMT"
REF_HUMAN_GMT_1 = "7 July 2023 00:01 GMT"
# REF_EPOCH_S = 1688688600
REF_EPOCH_S = 1688688000
# FOR FUTURE END OF LOOP
# END OF DATA SET: 2024-10-01 00:00 GMT
# REF_END_S = 1727697600

# Window duration and hop duration in seconds
EPISODE_START_EPOCH_S = REF_EPOCH_S
WINDOW_DURATION_S = 60
HOP_S = 60 # [no overlap]
# Batch process duration
# for 60min/h, 24d, 1 day
BATCH_DURATION_S = int(24*60)  # 1 day, not used (for future ref)

# Event name
EVENT_NAME = "GuamNet"


# NOMINAL Station locations
# 1637653001
# lat: 13.477441292783102
# lng: 144.79427615160373
#
# 1637653002
# lat: 13.472148240673388
# lng: 144.76390970046936
#
# 1637653003
# lat: 13.401056611220948
# lng: 144.6995204150371
#
# 1637653004
# lat: 13.50856407545507
# lng: 144.80886653065681
#
# 1637653005
# lat: 13.477465349715203
# lng: 144.7942762915045
#
# 1637653006
# lat: 13.473639938863599
# lng: 144.78171961615016
#
# 1637653007
# lat: 13.531094861739364
# lng: 144.86088062673053
#
# 1637653008
# lat: 13.470666477176973
# lng: 144.80433055202366
#
# 1637653009
# lat: 13.562106960453093
# lng: 144.84369149431586
