import os
import argparse
import numpy as np
import plotly.graph_objects as px
import re


def parse_file(file_name):
    # Open the file
    with open(file_name, 'r') as file:
        # Initialize an empty list to store the rms values
        rms_values = []
        time_values = []

        start_time = None
        # Read the file line by line
        for line in file:
            # Split the line into fields based on the colon (:) separator
            fields = line.split(':')
            # Check if the line contains the 'rms' value
            if 'rms' in line:
                # Extract the 'rms' value from the line
                # print(line)
                rms_value = fields[1].strip().split()[1]
                # Append the 'rms' value to the list
                rms_values.append(float(rms_value))
                # find the time stamp
                # Find the time stamp using regex
                time_stamp_match = re.search(r'\[(\d+\.\d+)\]', line)
                if time_stamp_match:
                    time_stamp = float(time_stamp_match.group(1))
                    # If start_time has not been set yet, set it to the current time_stamp
                    if start_time is None:
                        start_time = time_stamp
                    # Calculate the time relative to the start_time
                    time_sec = time_stamp - start_time
                    # Append the relative time to the list
                    time_values.append(time_sec)

    return rms_values, time_values


def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Extract rms values from a text file')

    # Add an argument for the file name
    parser.add_argument('-f', '--file', type=str, help='Name of the input file')
    
    # Add an argument for the directory path
    parser.add_argument('-d', '--directory', type=str, help='Path to the directory containing input files')

    # Add a flag enable attacks filter lines based on a keyword
    parser.add_argument('-a', '--attacks', type=str, help='Enable Attacks')

    # Add a flag enable attacks filter lines based on a keyword
    parser.add_argument('-t', '--trace', type=str, help='To distinguish trace')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if neither file nor directory argument is provided
    if not args.file and not args.directory:
        parser.error('Please provide either a file or a directory')

    # Create an empty dictionary to store the rms arrays
    rms_arrays = {}
    time_arrays = {}

    # If file argument is provided, process the single file
    if args.file and not args.attacks:
        print(args.file)
        rms_values, time_values = parse_file(args.file)
        rms_array = np.array(rms_values)
        time_array = np.array(time_values)
        rms_arrays[args.file] = rms_array
        time_arrays[args.file] = time_array
    
        # If file argument is provided, process the single file
    if args.file and args.attacks:
        print(args.attacks)

        rms_values, time_values = parse_file(args.file)
        rms_array = np.array(rms_values)
        time_array = np.array(time_values)
        rms_arrays[args.file] = rms_array
        time_arrays[args.file] = time_array

        # Define a dictionary to map file names to keys
        file_key_map = {'run1-8sep-aerial-increasingDL-withUL.txt': '0', 
                    'run1-12sep-aerial-udpDL.txt': '1', 
                    'run2-8sep-aerial-increasingDL-noUL.txt': '2', 
                    'run3-8sep-aerial-maxDLUL.txt': '3'}
        
        # Define the folder path based on the attack type
        attack_folder_path = {
            'Announce': './DataCollectionPTP/RU/MaliciousTraffic/Announce/',
            'Sync': './DataCollectionPTP/RU/MaliciousTraffic/Sync_FollowUp/'
        }

        if args.attacks in ['Announce', 'Sync']:
            attack_type = args.attacks  # Get the attack type
            folder_path = attack_folder_path[attack_type]  # Get the folder path based on the attack type

            for attack_name in os.listdir(folder_path):
                attack_freq_folder_path = os.path.join(folder_path, attack_name)
                
                for file_name in os.listdir(attack_freq_folder_path):
                    file_path = os.path.join(attack_freq_folder_path, file_name)
                    #print(file_path)
                    if os.path.isfile(file_path):
                        key = file_key_map.get(file_name)
                        print(key)
                        if key == args.trace:
                            print(file_path)
                            rms_values, time_values = parse_file(file_path)
                            rms_array = np.array(rms_values)
                            time_array = np.array(time_values)
                            rms_arrays[file_path] = rms_array
                            time_arrays[file_path] = time_array


    # If directory argument is provided, process all files in the directory
    if args.directory:
        for file_name in os.listdir(args.directory):
            file_path = os.path.join(args.directory, file_name)
            if os.path.isfile(file_path):
                rms_values, time_values = parse_file(args.file)
                rms_array = np.array(rms_values)
                time_array = np.array(time_values)
                rms_arrays[args.file] = rms_array
                time_arrays[args.file] = time_array


    # Create a Plotly figure outside the loop
    plot = px.Figure()
    # Print the rms arrays for each file
    for file_name, rms_array in rms_arrays.items():
            print(file_name)
            #print(f'RMS array for {file_name}: {rms_array}')
            cur_file_name = file_name.replace('../securing_ORAN-master/DataCollectionPTP/RU/', "")
            new_file_name = cur_file_name.replace('/', " ")
            # print(time_array)
            plot.add_trace(px.Scatter(x=time_array, y=rms_array, mode='lines', name=f'{new_file_name}'))

            plot.update_layout(
                title="PTP Clock Offset for {}".format(args.file.replace('../securing_ORAN-master/DataCollectionPTP/RU/', "")),
                yaxis_title="Offset (ns)",
                xaxis_title="Elapsed time (s)",
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                step="second",
                                stepmode="backward"),
                        ])
                    ),
                    rangeslider=dict(
                        visible=True
                    ),
                ),
                yaxis=dict(
                    autorange=True,
                    fixedrange=False
                )
            )

    plot.show()


if __name__ == '__main__':
    main()


