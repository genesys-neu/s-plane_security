import time
import streamlit as st
import subprocess
import os
import random
import psutil
import csv
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("app.log"),  # Log to a file
    logging.StreamHandler()  # Log to the console
])

# Directory containing the attack scripts
directory = 'Testbed/PipelineTestAttacker/Scripts/'

# Global variables to manage the attack process and status
if 'attack_process' not in st.session_state:
    st.session_state.attack_process = None  # Initialize in session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

def get_network_interfaces():
    """Retrieve available network interfaces."""
    interfaces = psutil.net_if_addrs()
    return list(interfaces.keys())


# Function to create the log file
def create_log_file(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['attack type', 'attack start', 'attack end'])
        st.success(f'Created {filename}')


# Function to handle starting the attack
def start_attack(attack, interface, duration, sleep, filename, password):

    # Only start the attack if no attack is running
    if not st.session_state.is_running:
        st.session_state.is_running = True

        # Prepare the command
        command = ["sudo", "-S", "python3", attack, '-i', interface, '-d', str(duration), '-s', str(sleep), '-l', filename]

        # Run the command with the password
        try:
            st.session_state.attack_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            st.session_state.attack_process.stdin.write(password.encode() + b'\n')  # Write the password to stdin
            st.session_state.attack_process.stdin.flush()

            # Log the PID
            logging.info(f'Started attack process with PID: {st.session_state.attack_process.pid}')

            st.write(
                f'Starting attack: {os.path.basename(attack)} for {duration} seconds with {sleep} seconds of sleep.'
            )
        except Exception as e:
            st.error(f"Error starting attack: {e}")
            st.session_state.is_running = False

        logging.info(f'global attack process after start_attack: {st.session_state.attack_process}')


def stop_attack():
    logging.info("Attempting to stop the attack...")
    # logging.info(f'global attack process before stop_attack: {st.session_state.attack_process}')

    if st.session_state.attack_process is not None and st.session_state.is_running:
        # logging.info(f"attack_process: {st.session_state.attack_process}, is_running: {st.session_state.is_running}")
        try:
            logging.info(f"Terminating process with PID: {st.session_state.attack_process.pid}")
            st.session_state.attack_process.terminate()  # Attempt to terminate the process
            st.session_state.attack_process.wait(timeout=5)  # Wait for the process to exit, with a timeout
            st.session_state.is_running = False
            logging.info("Attack stopped successfully.")
        except subprocess.TimeoutExpired:
            logging.info("Process did not terminate in time, killing it...")
            st.session_state.attack_process.kill()  # Force kill if it doesn't exit in time
            st.session_state.is_running = False
            logging.info("Attack process was forcefully killed.")
        except Exception as e:
            logging.error(f"Error stopping the attack: {e}")
    else:
        logging.info("No attack process to stop.")
        st.session_state.is_running = False
    st.session_state.attack_process = None


def main():
    # Streamlit UI with formatted title using HTML
    st.markdown(
        "<h1 style='text-align: center;'>TIMESAFE:"
        "<div style='text-align: center; font-size: 16px;'><u>T</u>iming <u>I</u>nterruption <u>M</u>onitoring and "
        "<u>S</u>ecurity <u>A</u>ssessment for <u>F</u>ronthaul <u>E</u>nvironments</h1></div>",
        unsafe_allow_html=True
    )

    # Get available network interfaces
    interfaces = get_network_interfaces()

    # Inputs
    interface = st.selectbox("Select Network Interface:", interfaces)
    output_folder = st.text_input("Enter the folder to store outputs:")
    attack_duration = st.selectbox("Select Duration Test:", ["random", "fixed", "continuous"])
    if attack_duration == "random":
        duration = random.randint(10, 30)
        sleep = random.randint(40, 60)
    elif attack_duration == "continuous":
        duration = 600
        sleep = 0
    else:
        duration = st.number_input("Enter the attack duration in seconds:", min_value=1)
        sleep = st.number_input("Enter the sleep duration in seconds:", min_value=1)

    # Dropdown menu for attack selection
    attacks = ["Spoofing Attack", "Replay Attack"]
    selected_attack = st.selectbox("Select Attack Type:", attacks)

    # Initialize attack script mapping
    attack_script = None

    # Additional dropdown for "Replay Attack" options
    if selected_attack == "Replay Attack":
        replay_options = st.selectbox("Select Replay Attack Type:", ["Sync only", "Sync and Follow Up"])
        if replay_options == "Sync only":
            attack_script = os.path.join(directory, "Sync_Attack.py")
        elif replay_options == "Sync and Follow Up":
            attack_script = os.path.join(directory, "Sync_FollowUp_Attack.py")
    elif selected_attack == "Spoofing Attack":
        attack_script = os.path.join(directory, "Announce_Attack.py")

    # Prompt for sudo password
    password = st.text_input("Enter your sudo password:", type="password")
    logging.info(f'global attack process: {st.session_state.attack_process}')

    if st.button("Start Attack") and not st.session_state.is_running:
        if interface and output_folder and password:
            # Create the output folder if it does not exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Extract the script name minus the .py extension
            script_name = os.path.basename(attack_script).replace(".py", "")
            logging.info(f'Script name: {script_name}')

            # Start the attack and generate the log filename
            filename = os.path.join(output_folder, f'{script_name}.{interface}.{duration}.{sleep}.csv')
            create_log_file(filename)

            # Call the attack function
            logging.info(f'Calling start_attack, is_running = {st.session_state.is_running}')
            start_attack(attack_script, interface, duration, sleep, filename, password)
        else:
            st.error("Please fill in all fields.")

    # Add a spinning indicator when the attack is running
    if st.session_state.is_running:
        st.markdown("<h3 style='color:red;'>🔴 Attack is running...</h3>", unsafe_allow_html=True)
        logging.info('The attack is running display is showing')
        if st.button("Stop Attack"):
            st.markdown("<h3 style='color:orange;'>🟠 Terminating Attack...</h3>", unsafe_allow_html=True)
            logging.info(f'Stop button clicked, calling stop_attack()')
            stop_attack()
            logging.info(f'Completed stop_attack, is_running = {st.session_state.is_running}')
            st.rerun()
    else:
        st.markdown("<h3 style='color:green;'>🟢 No attack running</h3>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
