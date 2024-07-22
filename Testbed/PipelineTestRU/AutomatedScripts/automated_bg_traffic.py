import socket
import time
import subprocess

def establish_connection():
    # Connect to server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('10.188.57.241', 9998))  # Replace 'server_ip_address' with the actual server IP
    return client

if __name__ == "__main__":
    # Connect to server
    client = establish_connection()
   
    # Label to synchronize the devices
    RU_ready = False
    DU_ready = False

    # Counter of tests
    test_number = 0

    # Define experiment duration in seconds
    duration_experiment = 57600

    # Take initial timestamp
    start_experiment = time.time()
    
    # Send READY message to connected devices
    client.sendall(f'READY TEST {test_number}'.encode())
    print(f"CONNECTION ESTABLISHED AND MESSAGE READY NUMBER {str(test_number)} SENT")

    # Set Attacker flag True
    DU_ready = True

    # Receive message from connected devices
    start_signal = client.recv(1024)

    # If connected device is READY, set flag True 
    if start_signal == f'READY TEST {test_number}'.encode():
        RU_ready = True

    # Start of experiment 
    while time.time()-start_experiment < duration_experiment:

        # If all devices are ready start test 
        if RU_ready and DU_ready:
            # Cyclically run all background traffic
            print(f'READY {test_number}')
            subprocess.run(["sudo", "python3", 'OFH_tgen.py',"-r", "-i","192.168.40.1", "-f", "./Cleaned_CU_plane_traces/run1-12sep-aerial-udpDL.csv"])
            subprocess.run(["sudo", "python3", 'OFH_tgen.py',"-r", "-i","192.168.40.1", "-f", "./Cleaned_CU_plane_traces/run1-8sep-aerial-increasingDL-withUL.csv"])
            subprocess.run(["sudo", "python3", 'OFH_tgen.py',"-r", "-i","192.168.40.1", "-f", "./Cleaned_CU_plane_traces/run2-8sep-aerial-increasingDL-noUL.csv"])
            subprocess.run(["sudo", "python3", 'OFH_tgen.py',"-r", "-i","192.168.40.1", "-f", "./Cleaned_CU_plane_traces/run3-8sep-aerial-maxDLUL.csv"])

    # Close connection when experiment ends
    client.close()
