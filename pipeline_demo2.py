import threading
import queue
import time
import argparse
from train_test import TransformerNN
import re
import torch
from functools import partial
from scapy.all import rdpcap, PcapReader, Ether, Scapy_Exception
from scapy.all import *
import signal  #SIMONE add signal library
import subprocess
import os


mac_mapping = {}  #Dictionary for MAC mapping {MAC:index}
index = 0  # Index to map MAC addresses


def start_tcpdump(file_path, interface):
    """
    Starts tcpdump to capture only PTP packets on the specified interface
    and saves them to a file.
    """
    password = 'op3nran'

    # Check if the file exists and remove it using sudo
    if os.path.exists(file_path):
        try:
            subprocess.run(f"echo {password} | sudo -S rm {file_path}", shell=True, check=True)
            print(f"Removed existing file: {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove the file: {e}")


    tcpdump_command = f"echo {password} | sudo -S tcpdump -i {interface} -w {file_path} ether proto 0x88f7"
    process = subprocess.Popen(tcpdump_command, shell=True)
    return process


def pre_processing(packet_queue, preprocessed_queue):
    """
    Function representing the pre-processing thread.
    Retrieves packets from the packet queue, preprocesses them, and puts them into the preprocessed queue.
    """
    global index  # Index to map the MAC addresses
    while not exit_flag.is_set():
        try:
            # Retrieve packet from packet queue
            # print(f'Packet queue in preprocessing: {packet_queue.qsize()}')
            packet = packet_queue.get(timeout=.04)  # Timeout to prevent blocking indefinitely

            # Preprocessing logic: mapping MAC addresses to indices
            # Source address processing
            if packet[0] not in mac_mapping:
                mac_mapping[packet[0]] = index
                packet[0] = index
                index += 1
            else:
                packet[0] = mac_mapping[packet[0]]

            # Destination address processing
            if packet[1] not in mac_mapping:
                mac_mapping[packet[1]] = index
                packet[1] = index
                index += 1
            else:
                packet[1] = mac_mapping[packet[1]]

            # Place preprocessed packet into the preprocessed queue
            preprocessed_queue.put(packet)
            # print(f'Preprocessed queue length in preprocessing: {preprocessed_queue.qsize()}')

        except queue.Empty:
            continue

        if exit_flag.is_set():  # Break the outer loop if flag is set
            break


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
            preprocessed_packet = preprocessed_queue.get(timeout=.04)  # Timeout to prevent blocking indefinitely
            sequence.append(preprocessed_packet)

            # print(f'Sequence length is {len(sequence)}')
            # Check if sequence length meets the desired criteria (e.g., a fixed length or a certain number of packets)
            if len(sequence) == sequence_length:
                # Move sequence to the correct device
                sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
                # Forward pass
                outputs = model(sequence_tensor)
                label = int(torch.round(outputs).item())  # Round the predictions to 0 or 1
                print("Predicted label:", label)
                # Adjust sliding window size based on queue size
                queue_size = preprocessed_queue.qsize()
                # print(f'Inference queue length is {queue_size}')
                window_size = 2
                # if queue_size <= 2:
                #     window_size = 2
                # elif queue_size >= 30:
                #     window_size = 30
                # else:
                #     window_size = queue_size
                sequence = sequence[window_size:]  # Remove the oldest element from the sequence (sliding window)

        except queue.Empty:
            continue

        if exit_flag.is_set():  # Break the outer loop if flag is set
            break


def acquisition_from_file(packet_queue, file_path, initial_time):
    packet_buffer = b""  # Initialize packet buffer
    offset = 0  # Initialize offset to start reading from the beginning
    first_run = True  # Flag to track the first run to read the global header

    while not exit_flag.is_set():
        while not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            time.sleep(0.1)
            print('Waiting for .pcap file')

        start = time.time()
        try:
            with open(file_path, 'rb') as f:
                # print(f"offset: {offset}")
                f.seek(offset)
                # start = time.time()

                if first_run:
                    # Read the global header only once
                    pcap_global_header = f.read(24)  # Global header size in pcap files
                    first_run = False
                    # print(f"global header: {pcap_global_header}")

                new_data = f.read()
                if new_data:
                    # print(f'Took {1000*(time.time()-start)} ms to read')
                    # Append the new data to the buffer
                    # print("new data")
                    packet_buffer += new_data
                    offset = f.tell()

                    # Combine the global header with the buffer
                    combined_data = pcap_global_header + packet_buffer

                    # Try to parse packets one at a time
                    while len(packet_buffer) >= 100:
                        try:
                            # buffer_length = len(packet_buffer)
                            # print(f'Before reading packet, packet buffer length : {buffer_length}')
                            packets = rdpcap(io.BytesIO(combined_data), count=1)  # Read one packet

                            if not packets:  # Check if no packets were read
                                break

                            # print('Packet read')
                            packet = packets[0]  # Get the first packet
                            ptp_info = []  # Prepare packet info

                            if initial_time is None:
                                initial_time = packet.time  # Use the time from the first packet

                            if Ether in packet and packet[Ether].type == 35063:
                                # Extract relevant PTP info
                                ptp_info.append(packet[Ether].src)
                                ptp_info.append(packet[Ether].dst)
                                ptp_info.append(len(packet))
                                ptp_info.append(int.from_bytes(packet.load[30:32], byteorder='big'))  # Sequence ID
                                ptp_info.append(int.from_bytes(packet.load[:1], byteorder='big'))  # Message type
                                ptp_info.append(float(packet.time - initial_time))
                                # start = time.time()
                                packet_queue.put(ptp_info, timeout=0.1)
                                # print(f'Took {1000*(time.time()-start)} ms to place item in queue')
                                # print(f'Adding {ptp_info} to queue')
                                initial_time = packet.time

                            # Update combined_data to remove the processed packet
                            processed_bytes = len(packet) + 16
                            # Use the actual length of the packet + 16 bytes for the PCAP packet header
                            # print(f'Processed bytes: {processed_bytes}')

                            # Ensure we have enough data to read next packet
                            combined_data = combined_data[processed_bytes:]  # Retain unused data
                            packet_buffer = combined_data[24:]  # Keep the rest of the data in the buffer

                            # Check the buffer length before next read
                            # buffer_length = len(packet_buffer)
                            # print(f'After reading packet, packet buffer length: {buffer_length}')

                            # Reset combined_data to include the global header
                            combined_data = pcap_global_header + packet_buffer
                            # print(f'Packet queue in acquisition: {packet_queue.qsize()}')

                        except Scapy_Exception as e:
                            print(f'Exception {e}, waiting for more data')
                            break  # Break the inner loop if an exception occurs
        print(f'Took {1000*(time.time()-start)} ms to complete try loop')

        except Exception as e:
            print(f"Error reading from file: {str(e)}")
        # time.sleep(0.01)


def signal_handler(sig, frame):
    """
    Signal handler function to gracefully handle keyboard interrupts (Ctrl+C).
    """
    print("\nExiting gracefully...")
    exit_flag.set()
    tcpdump_process.terminate()  # Terminate tcpdump


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
    parser.add_argument("-i", "--interface", type=str, required=True,
                        help="Network interface to listen on")
    args = parser.parse_args()

    # Set the desired_sequence_length based on the command-line argument
    #desired_sequence_length = args.length SIMONE COMMENTED BECAUSE NOT USED AND GENERATES AN ERROR

    # Path to the file where tcpdump will save packets
    pcap_file_path = "/tmp/ptp_packets.pcap"

    # Start tcpdump on the specified interface
    tcpdump_process = start_tcpdump(pcap_file_path, args.interface)

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and use your model with the extracted parameters
    slice_size_match = re.findall(r'\.(\d+)', args.model)
    model = None #SIMONE Initially declared model
    slice_length = None #SIMONE Initially declared model
    if len(slice_size_match) == 2:
        slice_length = int(slice_size_match[1])
        n_heads = int(slice_size_match[0])
        print(f'Using Transformer with slice size {slice_length} and {n_heads} heads')
        model = TransformerNN(slice_len=slice_length, nhead=n_heads).to(device)
        try:
            #SIMONE add the map_location since without this parameter it raises an error if CPU is used 
            model.load_state_dict(torch.load(args.model, map_location=device))
        except FileNotFoundError:
            print(f"Model weights file '{args.model}' not found.")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
    else:
        print("Invalid model name format. Please specify the slice size and number of heads.")

    # Create and start the acquisition thread
    acquisition_thread = threading.Thread(target=acquisition_from_file,
                                          args=(packet_queue, pcap_file_path, None))
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
            start = time.time()
            while time.time()-start <= args.time_out:
                time.sleep(1)
                if exit_flag.is_set():  # Break the outer loop if flag is set
                    break
            exit_flag.set()
        else:
            while not exit_flag.is_set():
                time.sleep(1)  # Main thread sleeps to keep the program running
    except KeyboardInterrupt:
        exit_flag.set()

    # Stop tcpdump
    tcpdump_process.terminate()

    # Join the threads to wait for their completion
    acquisition_thread.join()
    pre_processing_thread.join()
    inference_thread.join()

    print("Program exited successfully.")
