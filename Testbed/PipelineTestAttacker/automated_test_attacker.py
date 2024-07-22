import csv
import subprocess
import random
import time
import socket


# Initialize CSV file with headers
def create_log_file(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['attack_type','attack_start', 'attack_end'])
        print(f'Created {filename}')

# Starts attack
def start_attack(attack, duration, sleep, filename):
    subprocess.run(["sudo", "python3", attack, '-i', 'enp4s0', '-d', str(duration),'-s', str(sleep), '-l', filename])

# Establish connections
def establish_connection():
    # Connect to server
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('10.188.57.241', 9999))  # Replace 'server_ip_address' with the actual server IP
    return client

if __name__ == "__main__":

    # Establish connection
    client = establish_connection()
   
    # Label to synchronize the devices
    DU_ready = False
    Attacker_ready = False

    # Counter of tests
    test_number = 0

    # Define experiment duration
    duration_experiment = 57600

    # Take initial timestamp
    start_experiment = time.time()

    # Start of experiment 
    while time.time()-start_experiment < duration_experiment:

        # Send READY message to connected devices
        client.sendall(f'READY TEST {test_number}'.encode())
        print(f"CONNECTION ESTABLISHED AND MESSAGE READY NUMBER {str(test_number)} SENT")

        # Set Attacker flag True
        Attacker_ready = True

        # Receive message from connected devices
        start_signal = client.recv(1024)

        # If connected device is READY, set flag True 
        if start_signal == f'READY TEST {test_number}'.encode():
            DU_ready = True

        # If all devices are ready start test 
        if DU_ready and Attacker_ready:
            print(f'READY {test_number}')

            # Create log files
            filename = f'./AttackerLogs/test_attacker_{test_number}.csv'
            create_log_file(filename)
            
            # Define single test duration
            duration_test = 300

            # Take initial timestamp per tests
            current_time = time.time()
            start_time = time.time()
            
            # Start test
            print(f'START TEST NUMBER {test_number}')
            while current_time - start_time < duration_test: 
                # Choose Attack files
                attacks = ['./Scripts/Announce_Attack.py', './Scripts/Sync_FollowUp_Attack.py']
                attack = random.choice(attacks)

                # Choose Sleep and Duration times
                duration = random.randint(10,30)
                sleep = random.randint(40,60)
                print(f'{duration} seconds of duration and {sleep} seconds of sleep')

                # Start selected Attack
                print(f'{attack} started')
                start_attack(attack, duration, sleep, filename)

                # Take current timestamp
                current_time = time.time()
                
                # Set all flags False to wait for next test
                DU_ready = False
                Attacker_ready = False

            # Increase counter after test ends   
            test_number += 1

    # Close connection when experiment ends
    client.close()



