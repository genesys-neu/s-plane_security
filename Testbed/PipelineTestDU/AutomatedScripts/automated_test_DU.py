import csv
import subprocess
import time
import socket
import argparse
import os

# Create log file
def create_log_file(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['predicted_label','timestamp'])
        print(f'Created {filename}')

# Starts attack
def start_capturing(timeout, filename, type_detection):
    # Choose between the Machine Learning model or the Heuristic Rule based solution
    ''' 
    -m to choose the model (only for machine learning model)
    -t time duration for each test performed during the experiment
    -l filename to store the logs
    '''
    if type_detection == 'ml':
        subprocess.run(["sudo", "python3", '../PipelineScripts/pipeline.py', '-m', '../PipelineScripts/Models/'+args.model, '-t', str(timeout), '-l', filename, '-i', args.interface])
    elif type_detection == 'he':
        subprocess.run(["sudo", "python3", '../PipelineScripts/pipeline_heuristic.py', '-t', str(timeout), '-l', filename, '-i', args.interface])

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



if __name__ == "__main__":

    # Parse command-line arguments to choose the type of pipeline (default is machine learning)
    parser = argparse.ArgumentParser(description="Run packet processing with optional timeout")
    parser.add_argument("-t", "--type", type=str, help="Type of detection", default='ml')
    parser.add_argument("-if", "--interface", type=str, help="interface where to sniff")
    parser.add_argument("-m", "--model", type=str, help="model to use in the pipeline")
    parser.add_argument("-o", "--output", type=str, help="output folder")
    parser.add_argument("-de", "--duration_experiment", type=str, help="duration of the whole experiment")
    parser.add_argument("-dt", "--duration_test", type=str, help="duration of the single test")
    args = parser.parse_args()

    # Creates output folder if does not exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)  
    type_detection = args.type

    # Establish connection
    server, conn = establish_connection()
    
    # Label to synchronize the devices
    DU_ready = False
    Attacker_ready = False
    
    # Counter of tests
    test_number = 0

    # Define experiment duration
    duration_experiment = args.duration_experiment
    
    # Take initial timestamp
    start_experiment = time.time()

    # Start of experiment 
    while time.time()-start_experiment < int(duration_experiment):

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

        # If all devices are ready start test 
        if DU_ready and Attacker_ready:
            print(f'READY {test_number}')

            # Create log files
            filename = f'{args.output}/test_DU_{test_number}.csv'
            create_log_file(filename)

            # Define single test duration
            duration_test = args.duration_test

            print(f'START TEST NUMBER {test_number}')

            # Start capturing logs
            print('Capturing started')
            start_capturing(duration_test, filename, type_detection)

            # Increase counter after test ends
            test_number += 1

            # Set all flags False to wait for next test
            Attacker_ready = False
            DU_ready = False

    # Close connection when experiment ends
    conn.close()
    server.close()


