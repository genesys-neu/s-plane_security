import threading
import queue
import time
import argparse
from train_test import TransformerNN
import re
import torch
import signal  #SIMONE add signal library
import subprocess
import struct


mac_mapping = {}  #Dictionary for MAC mapping {MAC:index}
index = 0  # Index to map MAC addresses


# def start_tcpdump(file_path, interface):
#     """
#     Starts tcpdump to capture only PTP packets on the specified interface
#     and saves them to a file.
#     """
#     password = 'op3nran'
#
#     # Check if the file exists and remove it using sudo
#     if os.path.exists(file_path):
#         try:
#             subprocess.run(f"echo {password} | sudo -S rm {file_path}", shell=True, check=True)
#             print(f"Removed existing file: {file_path}")
#         except subprocess.CalledProcessError as e:
#             print(f"Failed to remove the file: {e}")
#
#     tcpdump_command = f"echo {password} | sudo -S tcpdump -B 64 -U -i {interface} -w {file_path} ether proto 0x88f7"
#     process = subprocess.Popen(tcpdump_command, shell=True)
#     return process


def kill_tcpdump_processes():
    """
    Kills all running tcpdump processes.
    """
    try:
        # Get the list of all tcpdump processes
        result = subprocess.run("ps aux | grep tcpdump | grep -v grep", shell=True, text=True, capture_output=True)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                # Extract the PID from the line
                parts = line.split()
                pid = parts[1]  # The second part is the PID
                print(f"Killing tcpdump process with PID: {pid}")
                # Kill the process
                subprocess.run(f"sudo kill {pid}", shell=True)
    except Exception as e:
        print(f"Error while killing tcpdump processes: {e}")


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
            packet = packet_queue.get(timeout=.01)  # Timeout to prevent blocking indefinitely

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
            preprocessed_packet = preprocessed_queue.get(timeout=.01)  # Timeout to prevent blocking indefinitely
            sequence.append(preprocessed_packet)

            # print(f'Sequence length is {len(sequence)}')
            # Check if sequence length meets the desired criteria (e.g., a fixed length or a certain number of packets)
            if len(sequence) == sequence_length:
                # Move sequence to the correct device
                sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(device)
                # Forward pass
                outputs = model(sequence_tensor)
                label = int(torch.round(outputs).item())  # Round the predictions to 0 or 1
                print(f"Predicted label: {label}", flush=True)
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


def acquisition_from_tcp(packet_queue, interface, initial_time):
    """
    Captures PTP packets using tcpdump and reads them directly from subprocess output.
    Extracts relevant information and puts it into a packet queue.
    """
    # Start tcpdump to capture PTP packets without writing to a file
    tcpdump_command = [
        'sudo', 'tcpdump', '-i', interface, '-U', '-B', '64', '-s', '0', 'ether', 'proto', '0x88F7'
    ]

    with subprocess.Popen(tcpdump_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        while not exit_flag.is_set():
            # Read raw data from tcpdump
            raw_data = proc.stdout.read(2048)  # Adjust size as needed
            if not raw_data:
                continue  # No data read

            # Process the raw data to extract packets
            offset = 0
            while offset < len(raw_data):
                # Read the length of the packet
                if offset + 16 > len(raw_data):  # Minimum size for PCAP packet header
                    break  # Not enough data for header

                # Unpack the PCAP header (16 bytes)
                pcap_header = raw_data[offset:offset + 16]
                packet_length, = struct.unpack('I', pcap_header[8:12])  # Get the captured length
                total_length, = struct.unpack('I', pcap_header[12:16])  # Get the total length

                # Check if we have enough data for the entire packet
                if offset + 16 + packet_length > len(raw_data):
                    break  # Not enough data for the entire packet

                # Read the actual packet data
                packet_data = raw_data[offset + 16:offset + 16 + packet_length]

                # Process the packet data
                ptp_info = []
                if len(packet_data) >= 34:  # Minimum PTP packet size (Ethernet header + PTP header)
                    eth_type = struct.unpack('>H', packet_data[12:14])[0]  # Get EtherType
                    if eth_type == 0x88F7:  # PTP EtherType
                        # Extract PTP info
                        ptp_info.append(packet_data[6:12].hex())  # Source MAC
                        ptp_info.append(packet_data[0:6].hex())   # Destination MAC
                        ptp_info.append(len(packet_data))           # Packet length
                        ptp_info.append(int.from_bytes(packet_data[30:32], byteorder='big'))  # Sequence ID
                        ptp_info.append(int.from_bytes(packet_data[0:1], byteorder='big'))  # Message type
                        if initial_time is None:
                            initial_time = time.time()  # Use current time if no initial time
                        ptp_info.append(time.time() - initial_time)  # Time since initial packet

                        # Add PTP info to the queue
                        packet_queue.put(ptp_info)

                # Move to the next packet
                offset += 16 + packet_length  # Move past the packet header and data

    # Make sure to handle process termination properly
    proc.terminate()


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
    # desired_sequence_length = args.length SIMONE COMMENTED BECAUSE NOT USED AND GENERATES AN ERROR

    # Path to the file where tcpdump will save packets
    # pcap_file_path = "/tmp/ptp_packets.pcap"

    # Start tcpdump on the specified interface
    # tcpdump_process = start_tcpdump(pcap_file_path, args.interface)

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize and use your model with the extracted parameters
    slice_size_match = re.findall(r'\.(\d+)', args.model)
    model = None  #SIMONE Initially declared model
    slice_length = None  #SIMONE Initially declared model
    if len(slice_size_match) == 2:
        slice_length = int(slice_size_match[1])
        n_heads = int(slice_size_match[0])
        print(f'Using Transformer with slice size {slice_length} and {n_heads} heads')
        model = TransformerNN(slice_len=slice_length, nhead=n_heads).to(device)
        try:
            # SIMONE add the map_location since without this parameter it raises an error if CPU is used
            model.load_state_dict(torch.load(args.model, map_location=device))
        except FileNotFoundError:
            print(f"Model weights file '{args.model}' not found.")
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
    else:
        print("Invalid model name format. Please specify the slice size and number of heads.")

    # Create and start the acquisition thread
    acquisition_thread = threading.Thread(target=acquisition_from_tcp,
                                          args=(packet_queue, args.interface, None))
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

    # # Stop tcpdump
    # if tcpdump_process:
    #     tcpdump_process.terminate()  # First try to terminate
    #     tcpdump_process.wait()  # Wait for the process to terminate

    # Check and kill any remaining tcpdump processes
    kill_tcpdump_processes()

    # Join the threads to wait for their completion
    acquisition_thread.join()
    pre_processing_thread.join()
    inference_thread.join()

    print("Program exited successfully.", flush=True)
