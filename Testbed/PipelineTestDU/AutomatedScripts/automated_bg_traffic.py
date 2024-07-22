import socket
import time
import subprocess

def establish_connection():
    # Start listening for connection
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9998))
    server.listen(1)  # Listen for 1 connection

    print('Server is listening for a connection...')

    conn, addr = server.accept()
    print(f'Accepted connection from {addr}')

    return server, conn


if __name__ == "__main__":

    # Establish connection
    server, conn = establish_connection()
    
    # Label to synchronize the devices
    DU_ready = False
    Attacker_ready = False
    
    # Counter of tests
    test_number = 0

    # Define experiment duration
    duration_experiment = 57600
    
    # Take initial timestamp
    start_experiment = time.time()

    # Send READY message to connected devices
    conn.sendall(f'READY TEST {test_number}'.encode())
    print(f"CONNECTION ESTABLISHED AND MESSAGE READY NUMBER {str(test_number)} SENT")
    
    # Set DU flag True
    DU_ready = True

    # Receive message from connected devices
    client_signal = conn.recv(16)

    # If connected device is READY, set flag True
    if client_signal == f'READY TEST {test_number}'.encode():
        Attacker_ready = True
            
    # Start of experiment 
    while time.time()-start_experiment < duration_experiment:

        # If all devices are ready start test 
        if DU_ready and Attacker_ready:
            print(f'READY {test_number}')
            subprocess.run(["sudo", "python3", "OFH_tgen.py", "-i", "192.168.40.51", "-f", "./PATH TO/Cleaned_CU_plane_traces/run1-12sep-aerial-udpDL.csv"])
            subprocess.run(["sudo", "python3", "OFH_tgen.py", "-i", "192.168.40.51", "-f", "./PATH TO/Cleaned_CU_plane_traces/run1-8sep-aerial-increasingDL-withUL.csv"])
            subprocess.run(["sudo", "python3", "OFH_tgen.py", "-i", "192.168.40.51", "-f", "./PATH TO/Cleaned_CU_plane_traces/run2-8sep-aerial-increasingDL-noUL.csv"])
            subprocess.run(["sudo", "python3", "OFH_tgen.py", "-i", "192.168.40.51", "-f", "./PATH TO/Cleaned_CU_plane_traces/run3-8sep-aerial-maxDLUL.csv"])

    # Close connection when experiment ends
    conn.close()
    server.close()
