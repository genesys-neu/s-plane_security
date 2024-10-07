# (c) 2024 Northeastern University
# Institute for the Wireless Internet of Things
# Created by Davide Villa (villa.d@northeastern.edu)

# Script that keeps updating the influxdb database as soon as new data are written in the ptp logs.
# To call the script use: "python3 updatedb_ptp.py"

import sys
import os
import time
import re
import argparse
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


# VARIOUS PARAMETERS
SYNC_FILES = ['/var/log/ptp4l.log', '/var/log/phc2sys.log']       # Path of the files with the data
SLEEP_TIME = 0.250                      # Sleep time between each operation in seconds
DELTA_TIME = 1000                       # Delta time for writing data in ms
DATA_FIELD = ['timestamp', 'rms', 'max_offset', 'freq', 'freq_uncertainty', 'delay', 'delay_uncertainty']

# OpenShift influx db parameters
INFLUX_URL = 'http://10.112.100.101'    # Influxdb url
INFLUX_PORT = '8086'                    # Influxdb port
INFLUX_ORG = 'wineslab'                 # Influxdb organization
INFLUX_BUCKET = 'wineslab-xapp-demo'    # Influxdb bucket
INFLUX_TOKEN = 'HaiL1SY6DIY87zKW_J7Ho9I2B_PkVjNgU9Rzcco10rUuSZcDG4OtG2JAtc7w6FU2oTwMiO2Mpd5A_nUPUyvCzA=='   # Influxdb token Colosseum container

# RE Patterns
pattern_phc2sys = r"phc2sys\[(\d+\.\d+)\]: CLOCK_REALTIME rms\s+(\d+)\s+max\s+(\d+)\s+freq\s+([+-]?\d+)\s+\+/\-\s+(\d+)\s+delay\s+(\d+)\s+\+/\-\s+(\d+)"
pattern_ptp4l = r"ptp4l\[(\d+\.\d+)\]: rms\s+(\d+)\s+max\s+(\d+)\s+freq\s+([+-]?\d+)\s+\+/\-\s+(\d+)\s+delay\s+(\d+)\s+\+/\-\s+(\d+)"

def read_last_line(file):
    """
    Read last line of a file
    """

    with open(file, "rb") as rfile:

        # Go to the end of the file before the last break-line
        rfile.seek(-2, os.SEEK_END)

        # Keep reading backward until you find the next break-line
        while rfile.read(1) != b'\n':
            rfile.seek(-2, os.SEEK_CUR)

        # print(rfile.readline().decode())

        line = rfile.readline().decode()

        # Remove new line characters
        line = line.strip()

        # Use Regular Expression to split line
        if 'ptp4l' in file:
            match = re.search(pattern_ptp4l, line)
        elif 'phc2sys' in file:
            match = re.search(pattern_phc2sys, line)

        return match


def is_valid_timestamp(file, match, timestamp_data):
    """
    Check if the timestamp is valid for the new data
    """

    # Not found RE
    if not match:
        return False, timestamp_data

    # Get last line timestamp
    timestamp = float(match.group(1))

    # Update timestamp and return true if not present or newer
    if (file not in timestamp_data) or (timestamp > timestamp_data[file]):
        timestamp_data[file] = timestamp
        return True, timestamp_data

    return False, timestamp_data


def create_dict(file, match):
    """
    Create dict data that is going to be written to the db
    """

    # Create Dictionary
    dict = {
        "measurement": "sync_stats",
        "tags": {
            "type": "",
        },
        "fields": {},
    }

    if 'ptp4l' in file:
        dict["tags"]["type"] = 'ptp4l'
    elif 'phc2sys' in file:
        dict["tags"]["type"] = 'phc2sys'

    # Fill all the data
    for i in range(0, len(DATA_FIELD)):
        dict["fields"][DATA_FIELD[i]] = float(match.group(i+1))

    # print(dict)
    return dict


def update_influxdb(data):
    """
    Update the influxdb by sending the newly loaded file
    """

    # Open an influxdb Client
    client = InfluxDBClient(url=INFLUX_URL+':'+INFLUX_PORT, token=INFLUX_TOKEN, org=INFLUX_ORG)

    # Open a write api
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Write data as a dictionary point
    write_api.write(INFLUX_BUCKET, INFLUX_ORG, data, 'ms')

    # Close influxdb Client session
    client.close()


def main():

    """
    Main operations
    """

    print("Starting to push ptp data into the influxdb.\n")
    print("Press CTRL-C to exit...")

    timestamp_data = {}

    while True:        # Keep reading and printing

        try:
            for file in SYNC_FILES:

                # Read last line
                match = read_last_line(file)

                # Check if need to be sent
                valid, timestamp_data = is_valid_timestamp(file, match, timestamp_data)
                if not valid:
                    continue

                # Create file to be written
                data = create_dict(file, match)

                # Send new data to influxdb
                update_influxdb(data)

            # Sleep 250ms
            time.sleep(SLEEP_TIME)

        except Exception as e:
            print("Exception in Main(): " + str(e))


if __name__ == '__main__':
    main()