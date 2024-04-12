import threading
import queue
import time
import argparse


def acquisition(packet_queue):
    """
    Function representing the packet acquisition thread.
    Acquires packets and puts them into the packet queue.
    """
    while not exit_flag.is_set():
        packet = acquire_packet()
        packet_queue.put(packet)
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


def inference(preprocessed_queue, sequence_length):
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
                label = model_inference(sequence)
                print("Predicted label:", label)
                # sequence = []  # Reset the sequence after making an inference
                sequence = sequence[2:] # Remove the 2 oldest element from the sequence (sliding window)

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


def acquire_packet():
    """
    Placeholder function for packet acquisition logic.
    This function acquires packets from the network.
    """
    # Placeholder for actual packet acquisition logic
    # For simplicity, let's return a dummy packet
    return "Dummy Packet"


def preprocess_packet(packet):
    """
    Placeholder function for packet preprocessing logic.
    This function preprocesses the acquired packet data.
    """
    # Placeholder for actual packet preprocessing logic
    # For simplicity, let's return the packet as is
    return packet


def model_inference(sequence):
    """
    Placeholder function for model inference logic.
    This function performs inference using the ML model.
    """
    # Placeholder for actual inference logic using the ML model
    # For simplicity, let's return a dummy label
    return "Dummy Label"


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
    parser.add_argument("-l", "--length", type=int, default=40, help="Length of sequence")
    args = parser.parse_args()

    # Set the desired_sequence_length based on the command-line argument
    desired_sequence_length = args.length

    # Create and start the acquisition thread
    acquisition_thread = threading.Thread(target=acquisition, args=(packet_queue,))
    acquisition_thread.start()

    # Create and start the pre-processing thread
    pre_processing_thread = threading.Thread(target=pre_processing, args=(packet_queue, preprocessed_queue,))
    pre_processing_thread.start()

    # Create and start the inference thread
    inference_thread = threading.Thread(target=inference, args=(preprocessed_queue, desired_sequence_length,))
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
