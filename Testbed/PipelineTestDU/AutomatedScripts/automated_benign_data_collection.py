import csv
import subprocess
import time
import argparse
import os



# Run tcpdump to capture file and convert the file to csv to save memory usage
def acquire_tcpdump(filename):
    '''
    timeout to determine each capture timeframe (one timeframe = one log file) specify number of second followed by 's' Ex. 300s
    -w to specify the output filename
    -i to specify the interface
    ether proto 0x88f7 to only filter PTP packets
    '''
    subprocess.run(["sudo", "timeout", str(args.duration_test), 'tcpdump', '-w', filename, '-i', args.interface, "ether", "proto", "0x88f7"])
    subprocess.run(["sudo", "python3", '../ProcessData/pcap_csv_converter.py', '-f', args.output])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interface", help="enter the interface", type=str, required=True)
    parser.add_argument("-de", "--duration_experiment", help="enter the duration of the whole experiment", type=int, required=True)
    parser.add_argument("-dt", "--duration_test", help="enter the duration of each test", type=int, required=True)
    parser.add_argument("-o", "--output", help="enter the folder where to store outputs", type=str, required=True)

    args = parser.parse_args()

    #Creates the output folder if not exists
    if not os.path.exists(args.output): 
        os.makedirs(args.output)
    initial_time = None

    # Define experiment duration
    duration_experiment = args.duration_experiment
    
    # Take initial timestamp
    start_experiment = time.time()


    test_number = 0

    # Start of experiment 
    while time.time()-start_experiment < duration_experiment:

        # Start capturing logs
        print('Capturing started')
        acquire_tcpdump(f'{args.output}dump_{test_number}.pcap')

        test_number += 1



