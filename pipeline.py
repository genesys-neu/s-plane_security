import threading
import queue
import time
import argparse
from train_test import TransformerNN
import re
import torch
from functools import partial
from scapy.all import *
from scapy.all import Ether

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


def inference(preprocessed_queue, model, sequence_length, device):
    """
    Function representing the inference thread.
    Retrieves preprocessed packets from the preprocessed queue, accumulates them to form a sequence,
    and then performs inference on the complete sequence.
    """
    sequence = []  # Initialize an empty sequence
    # start_time = time.time()  # Initialize start time for the timer
    while not exit_flag.is_set():
        try:
            preprocessed_packet = preprocessed_queue.get(timeout=1)  # Timeout to prevent blocking indefinitely
            sequence.append(preprocessed_packet)

            # Check if sequence length meets the desired criteria (e.g., a fixed length or a certain number of packets)
            if len(sequence) == sequence_length:
                # Move sequence to the correct device
                sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
                # Forward pass
                outputs = model(sequence_tensor)
                label = torch.round(outputs)  # Round the predictions to 0 or 1
                print("Predicted label:", label)
                # sequence = []  # Reset the sequence after making an inference
                sequence = sequence[2:]  # Remove the 2 oldest element from the sequence (sliding window)

            ''' 
            # Another time based approach we could use.
            if len(sequence) == desired_sequence_length:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timer_interval:
                    label = model_inference(sequence)
                    print("Predicted label:", label)
                    start_time = time.time()  # Reset the start time for the timer
                sequence.pop(0)  # Remove the oldest element from the sequence (sliding window)
            '''

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
    interface = 'enp4s0'
    # Create partial function to include new parameters
    process_ptp_packet_with_param = partial(process_ptp_packet, packet_queue=packet_queue)
    time = 5 # default time for sniffing
    if args.time_out:
        time = args.time_out/2
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
    parser.add_argument("-t", "--time_out", type=int, help="Timeout in seconds")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model weights to load (include path)")
    args = parser.parse_args()

    # Set the desired_sequence_length based on the command-line argument
    desired_sequence_length = args.length

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and use your model with the extracted parameters
    slice_size_match = re.findall(r'\.(\d+)', args.model)
    if len(slice_size_match) == 2:
        slice_length = int(slice_size_match[1])
        n_heads = int(slice_size_match[0])
        print(f'Using Transformer with slice size {slice_length} and {n_heads} heads')
        model = TransformerNN(slice_len=slice_length, nhead=n_heads).to(device)
        try:
            model.load_state_dict(torch.load(args.model))
        except FileNotFoundError:
            print(f"Model weights file '{args.model}' not found.")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
    else:
        print("Invalid model name format. Please specify the slice size and number of heads.")

    # Create and start the acquisition thread
    acquisition_thread = threading.Thread(target=acquisition, args=(packet_queue,))
    acquisition_thread.start()

    # Create and start the pre-processing thread
    pre_processing_thread = threading.Thread(target=pre_processing, args=(packet_queue, preprocessed_queue,))
    pre_processing_thread.start()

    # Create and start the inference thread
    inference_thread = threading.Thread(target=inference,
                                        args=(preprocessed_queue, model, slice_length, device,))
    inference_thread.start()

    try:
        # Optionally, run the program continuously for a specified time (time_out)
        if args.time_out:
            time.sleep(args.time_out)
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
