import csv
import subprocess
import time

# Append raw to the file
def log_lable_timestamp(ptp_info, filename):
    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(ptp_info)

# Run tcpdump to capture file and convert the file to csv to save memory usage
def acquire_tcpdump(filename):
    '''
    timeout to determine each capture timeframe (one timeframe = one log file) specify number of second followed by 's' Ex. 300s
    -w to specify the output filename
    -i to specify the interface
    ether proto 0x88f7 to only filter PTP packets
    '''
    subprocess.run(["sudo", "timeout", "DURATION", 'tcpdump', '-w', filename, '-i', 'INTERFACE', "ether", "proto", "0x88f7"])
    subprocess.run(["sudo", "python3","./PATH TO/pcap_csv_converter.py"])


if __name__ == "__main__":
    initial_time = None

    # Define experiment duration
    duration_experiment = 1200
    
    # Take initial timestamp
    start_experiment = time.time()

    # Take initial timestamp
    start_experiment = time.time()

    # Define filename 
    filename = './PATH TO PREFERRED FOLDER/FILENAME.csv'

    # Start of experiment 
    while time.time()-start_experiment < duration_experiment:
       
        # Start capturing logs
        print('Capturing started')
        acquire_tcpdump(filename)


