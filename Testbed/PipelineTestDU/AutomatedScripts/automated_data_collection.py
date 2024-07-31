import csv
import subprocess
import time
import socket
import argparse
import os


# Establish connections
def establish_connection():
    # Start listening for connection
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9999))
    server.listen(1)  # Listen for 1 connection

    print('Server is listening for a connection...')

    conn, addr = server.accept()
    print(f'Accepted connection from {addr}')

    return server, conn

# Run tcpdump to capture file and convert the file to csv to save memory usage
def acquire_tcpdump(filename):
    '''
    timeout to determine each capture timeframe (one timeframe = one log file) specify number of second followed by 's' Ex. 300s
    -w to specify the output filename
    -i to specify the interface
    ether proto 0x88f7 to only filter PTP packets
    '''
    subprocess.run(["sudo", "timeout", str(args.duration_test), 'tcpdump', '-w', filename, '-i', args.interface, "ether", "proto", "0x88f7"])
    subprocess.run(["sudo", "python3","../ProcessData/pcap_csv_converter.py", '-f', args.output])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--interface", help="enter the interface", type=str, required=True)
    parser.add_argument("-de", "--duration_experiment", help="enter the duration of the whole experiment", type=int, required=True)
    parser.add_argument("-dt", "--duration_test", help="enter the duration of each test", type=int, required=True)
    parser.add_argument("-o", "--output", help="enter the folder where to store outputs", type=str, required=True)
    args = parser.parse_args()
    
    # Creates output folder if does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    initial_time = None

     # Establish connection
    server, conn = establish_connection()
    
    # Label to synchronize the devices
    DU_ready = False
    Attacker_ready = False
    
    # Define experiment duration
    duration_experiment = args.duration_experiment
    
    # Take initial timestamp
    start_experiment = time.time()

    # Initialize counter for experiments
    i = 0

    # Start of experiment 
    while time.time()-start_experiment < duration_experiment:
    # Send READY message to connected devices
        conn.sendall(f'READY TEST {i}'.encode())
        print(f"CONNECTION ESTABLISHED AND MESSAGE READY NUMBER {str(i)} SENT")
        
        # Set DU flag True
        DU_ready = True

        # Receive message from connected devices
        client_signal = conn.recv(16)

        # If connected device is READY, set flag True
        if client_signal == f'READY TEST {i}'.encode():
            Attacker_ready = True

        # If all devices are ready start test 
        if DU_ready and Attacker_ready:
            print(f'READY {i}')

            # Define filename 
            filename = f'{args.output}test_DU_{i}.pcap'
           
            print('Capturing started')

            # Start capturing logs
            acquire_tcpdump(filename)
            Attacker_ready=False  
            DU_ready = False  

            # Increment counter for experiments
            i+=1
    
    # Close connection when experiment ends
    conn.close()
    server.close()

