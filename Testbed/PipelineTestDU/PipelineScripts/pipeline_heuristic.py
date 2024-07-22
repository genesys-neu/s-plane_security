import threading
import queue
import time
import argparse
from train_test import TransformerNN
from functools import partial
from scapy.all import *
from scapy.all import Ether
import signal 
import csv 

initial_time = None # Time used to calculate the offset for interarrival time
mac_mapping = {} #Dictionary for MAC mapping {MAC:index}
index = 0 # Index to map MAC addresses

def acquisition(packet_queue):
    """
    Function representing the packet acquisition thread.
    Acquires packets and puts them into the packet queue.
    """
    while not exit_flag.is_set():
        acquire_packet(packet_queue)
        # Optionally, add a delay or condition to control the acquisition rate

def pre_processing(packet_queue, preprocessed_queue):
    """
    Function representing the pre-processing thread.
    Retrieves packets from the packet queue, preprocesses them, and puts them into the preprocessed queue.
    """
    while not exit_flag.is_set():
        try:
            packet = packet_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
            preprocessed_packet = preprocess_packet(packet)
            preprocessed_queue.put(preprocessed_packet)
        except queue.Empty:
            continue

# Log the labels in the file
def log_lable_timestamp(label, timestamp, filename):
    # Open the file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([label, timestamp])


def inference(preprocessed_queue, sequence_length):
    """
    Function representing the inference thread.
    Retrieves preprocessed packets from the preprocessed queue, accumulates them to form a sequence,
    and then performs inference on the complete sequence.
    """
    global lable 
    seq_id_sync = []
    seq_id_folup = []
    previous_sync_id = 0
    previous_folup_id = 0
    is_sync_attack = False
    announce_count = 0
    sequence = []  # Initialize an empty sequence
    # start_time = time.time()  # Initialize start time for the timer
    while not exit_flag.is_set():
        try:
            preprocessed_packet = preprocessed_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
            sequence.append(preprocessed_packet) 
            # Spoofing attack check. Counts number of adjacent Announce packets and changes the label if at least 3 adjacent packets are found
            if preprocessed_packet[4]==1: # Ignores delay requests
                pass
            elif preprocessed_packet[4]==11: 
                announce_count+=1
            else:
                announce_count = 0
            if announce_count >= 3:
                lable = 'tensor([[1.]], grad_fn=<RoundBackward0>)' # Same output as the ML solution for a better data processing
                log_lable_timestamp(lable, time.time(), args.logs)
            else:
                lable = 'tensor([[0.]], grad_fn=<RoundBackward0>)' # Same output as the ML solution for a better data processing
                log_lable_timestamp(lable, time.time(), args.logs)
            # Replay attack checks for Sync packets
            if preprocessed_packet[4]==0: # Checks if the packet is Sync
                if previous_sync_id == 0: 
                    previous_sync_id = preprocessed_packet[3] #  Stores the last seen packet
                    seq_id_sync.append(preprocessed_packet[3]) #  Stores the packet in the queue
                else:
                    if preprocessed_packet[3]> previous_sync_id: # checks the value of the seq ID
                        previous_sync_id = preprocessed_packet[3] #  Stores the last seen packet
                    if preprocessed_packet[3] < previous_sync_id and preprocessed_packet[3] in seq_id_sync: # If the sequence ID has already occurred and is lower than the previous one it is a replayed message
                        is_sync_attack = True
                if is_sync_attack:
                    lable = 'tensor([[1.]], grad_fn=<RoundBackward0>)' # Same output as the ML solution for a better data processing
                    log_lable_timestamp(lable, time.time(), args.logs) # Stores output of detection in the log file
                    is_sync_attack = False
                else:
                    lable = 'tensor([[0.]], grad_fn=<RoundBackward0>)' # Same output as the ML solution for a better data processing
                    log_lable_timestamp(lable, time.time(), args.logs) # Stores output of detection in the log file
                    if len(seq_id_sync)>= sequence_length: # If queue is full
                        seq_id_sync.pop(0) # Removes the last recent sequence id from the queue
                        seq_id_sync.append(preprocessed_packet[3]) # Stores the last seen sequence ID in the queue
                    else:
                        seq_id_sync.append(preprocessed_packet[3]) # Stores the last seen sequence ID in the queue
            # Checks Reply attack for Follow Up packet
            if preprocessed_packet[4]==8:
                if previous_folup_id == 0:
                    previous_folup_id = preprocessed_packet[3] #  Stores the last seen packet
                    seq_id_folup.append(preprocessed_packet[3]) #  Stores the packet in the queue
                else:
                    if preprocessed_packet[3]> previous_folup_id: # checks the value of the seq ID
                        previous_folup_id = preprocessed_packet[3] #  Stores the last seen packet
                    if preprocessed_packet[3] < previous_folup_id and preprocessed_packet[3] in seq_id_folup: # If the sequence ID has already occurred and is lower than the previous one it is a replayed message
                        is_sync_attack = True
                if is_sync_attack:
                    lable ='tensor([[1.]], grad_fn=<RoundBackward0>)' # Same output as the ML solution for a better data processing
                    log_lable_timestamp(lable, time.time(), args.logs) # Stores output of detection in the log file
                    is_sync_attack = False
                else:
                    lable = 'tensor([[0.]], grad_fn=<RoundBackward0>)' # Same output as the ML solution for a better data processing
                    log_lable_timestamp(lable, time.time(), args.logs) # Stores output of detection in the log file
                    if len(seq_id_folup)>= sequence_length: # If queue is full
                        seq_id_folup.pop(0) # Removes the last recent sequence id from the queue
                        seq_id_folup.append(preprocessed_packet[3]) # Stores the last seen sequence ID in the queue
                    else:
                        seq_id_folup.append(preprocessed_packet[3]) # Stores the last seen sequence ID in the queue
                        
        except queue.Empty:
            continue



def acquire_packet(packet_queue):
    """
    Placeholder function for packet acquisition logic.
    This function acquires packets from the network.
    """
    # Placeholder for actual packet acquisition logic
    # For simplicity, let's return a dummy packet
    # Local function to process data
    def process_ptp_packet(pkt, packet_queue):
        global initial_time # Variable used to calculate the interarrival time
        ptp_info = [] # List to be added to the queue representing each packet
        if (initial_time == None):
            initial_time = pkt.time
        # Check if the packet is ETH 
        if Ether in pkt:
            # Check if the protocol is PTP 
            if pkt[Ether].type == 35063:
                # Extracting PTP raw data
                ptp_layer = pkt.load
                # Extracting desired parameters
                ptp_info.append(pkt[Ether].src) # source
                ptp_info.append(pkt[Ether].dst) # destination
                ptp_info.append(len(pkt)) # packet length
                ptp_info.append(int.from_bytes(ptp_layer[30:32], byteorder='big')) #sequence ID
                ptp_info.append(int.from_bytes(ptp_layer[:1], byteorder='big')) # message type
                ptp_info.append(pkt.time - initial_time) # timestamp
                # Put ptp info into the queue
                packet_queue.put(ptp_info)
                initial_time = pkt.time # Update the last timestamp
    # Define the interface to sniff on
    interface = 'INTERFACE' # Select interface you want to use
    # Create partial function to include new parameters
    process_ptp_packet_with_param = partial(process_ptp_packet, packet_queue=packet_queue)
    time = 5 # default time for sniffing
    if args.time_out:
        time = int(args.time_out)/2
    # Start sniffing packets
    sniff(iface=interface, prn=process_ptp_packet_with_param, timeout = time)
    return "Dummy Packet"


def preprocess_packet(packet_within_queue):
    global index # Index to map the MAC addresses
    # Source addresses processed
    if packet_within_queue[0] not in mac_mapping.keys():
        mac_mapping[packet_within_queue[0]] = index
        packet_within_queue[0] = index
        index += 1
    else:
        packet_within_queue[0] = mac_mapping[packet_within_queue[0]]
    # Destination addresses processed
    if packet_within_queue[1] not in mac_mapping.keys():
        mac_mapping[packet_within_queue[1]] = index
        packet_within_queue[1] = index
        index += 1
    else:
        packet_within_queue[1] = mac_mapping[packet_within_queue[1]]
    return packet_within_queue


def signal_handler(sig, frame):
    """
    Signal handler function to gracefully handle keyboard interrupts (Ctrl+C).
    """
    print("\nExiting gracefully...")
    exit_flag.set()


if __name__ == "__main__":
    # Initialize queues for communication between threads
    packet_queue = queue.Queue()
    preprocessed_queue = queue.Queue()

    # Initialize exit flag for graceful termination
    exit_flag = threading.Event()

    # Register signal handler for Ctrl+C (KeyboardInterrupt)
    signal.signal(signal.SIGINT, signal_handler)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run packet processing with optional timeout")
    parser.add_argument("-t", "--time_out", type=str, help="Timeout in seconds")
    parser.add_argument("-l", "--logs", type=str, help="Log files where to store information")# ADDED BY SIMONE
    args = parser.parse_args()

    # Set the desired_sequence_length based on the command-line argument
    #desired_sequence_length = args.length SIMONE COMMENTED BECAUSE NOT USED AND GENERATES AN ERROR

    slice_length = 40 # Initially declared model

    # Create and start the acquisition thread
    acquisition_thread = threading.Thread(target=acquisition, args=(packet_queue,))
    acquisition_thread.start()

    # Create and start the pre-processing thread
    pre_processing_thread = threading.Thread(target=pre_processing, args=(packet_queue, preprocessed_queue,))
    pre_processing_thread.start()

    # Create and start the inference thread
    inference_thread = threading.Thread(target=inference,
                                        args=(preprocessed_queue, slice_length,))
    inference_thread.start()

    try:
        # Optionally, run the program continuously for a specified time (time_out)
        if args.time_out:
            time.sleep(int(args.time_out))
            exit_flag.set()
        else:
            while not exit_flag.is_set():
                time.sleep(1)  # Main thread sleeps to keep the program running
    except KeyboardInterrupt:
        exit_flag.set()

    # Join the threads to wait for their completion
    acquisition_thread.join()
    pre_processing_thread.join()
    inference_thread.join()

    print("Program exited successfully.")
