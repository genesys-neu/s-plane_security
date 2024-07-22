import csv
import subprocess
import time
import socket

# Append raw to the file
def log_lable_timestamp(ptp_info, filename):
    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(ptp_info)

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
    subprocess.run(["sudo", "timeout","DURATION", 'tcpdump', '-w', filename, '-i', 'INTERFACE', "ether", "proto", "0x88f7"])
    subprocess.run(["sudo", "python3","pcap_csv_converter.py"])


if __name__ == "__main__":
    initial_time = None

     # Establish connection
    server, conn = establish_connection()
    
    # Label to synchronize the devices
    DU_ready = False
    Attacker_ready = False
    
    # Define experiment duration
    duration_experiment = 57600
    
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
            filename = f'./PATH TO PREFERRED FOLDER/FILENAME_number{i}.pcap'
           
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

