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
if 'terminating' not in st.session_state:
    st.session_state.terminating = False

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
    remote_host = "your.remote.server"  # Replace with the actual remote server address
    remote_user = "your_username"  # Replace with the actual remote user
    remote_password = "your_password"  # Replace with the actual remote user's password

    # Only start the attack if no attack is running
    if not st.session_state.is_running:
        st.session_state.is_running = True

        # Prepare the command
        command = ["sudo", "-S", "python3", attack, '-i', interface, '-d', str(duration), '-s', str(sleep), '-l', filename]

        # Prepare the remote command
        command = [
            "ssh", f"{remote_user}@{remote_host}", "sudo", "-S", "python3", attack,
            '-i', interface, '-d', str(duration), '-s', str(sleep), '-l', filename
        ]

        # Run the command with the password for SSH and sudo
        try:
            st.session_state.attack_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                               stderr=subprocess.PIPE)
            st.session_state.attack_process.stdin.write((remote_password + '\n').encode())  # SSH password
            st.session_state.attack_process.stdin.write((password + '\n').encode())  # sudo password
            st.session_state.attack_process.stdin.flush()

            st.write(
                f'Starting attack: {os.path.basename(attack)} for {duration} seconds with {sleep} seconds of sleep on {remote_host}.')
        except Exception as e:
            st.error(f"Error starting attack on remote server: {e}")
            st.session_state.is_running = False

        logging.info(f'Global attack process after start_attack: {st.session_state.attack_process}')


def stop_attack():
    logging.info("Attempting to stop the attack...")
    # logging.info(f'global attack process before stop_attack: {st.session_state.attack_process}')

    if st.session_state.attack_process is not None and st.session_state.is_running:
        # logging.info(f"attack_process: {st.session_state.attack_process}, is_running: {st.session_state.is_running}")
        try:
            # logging.info(f"Terminating process with PID: {st.session_state.attack_process.pid}")
            st.session_state.attack_process.terminate()  # Attempt to terminate the process
            st.session_state.attack_process.wait(timeout=5)  # Wait for the process to exit, with a timeout
            st.session_state.is_running = False
            # logging.info("Attack stopped successfully.")
        except subprocess.TimeoutExpired:
            # logging.info("Process did not terminate in time, killing it...")
            st.session_state.attack_process.kill()  # Force kill if it doesn't exit in time
            st.session_state.is_running = False
            # logging.info("Attack process was forcefully killed.")
        except Exception as e:
            logging.error(f"Error stopping the attack: {e}")
    else:
        # logging.info("No attack process to stop.")
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
    col1, col2 = st.columns(2)
    with col2:
        st.markdown("### Configure Attack")
        # Get available network interfaces
        # interfaces = get_network_interfaces()

        # Inputs
        # interface = st.selectbox("Select Network Interface:", interfaces)
        interface_a = 'eth0'
        # output_folder = st.text_input("Enter the folder to store outputs:")
        output_folder = './tmp'
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
        # logging.info(f'global attack process: {st.session_state.attack_process}')

        if not st.session_state.is_running:
            st.markdown("<h3 style='color:green;'>🟢 No attack running</h3>", unsafe_allow_html=True)
            if st.button("Start Attack"):
                if password:
                    # Create the output folder if it does not exist
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)

                    # Extract the script name minus the .py extension
                    script_name = os.path.basename(attack_script).replace(".py", "")
                    logging.info(f'Script name: {script_name}')

                    # Start the attack and generate the log filename
                    filename = os.path.join(output_folder, f'{script_name}.{interface_a}.{duration}.{sleep}.csv')
                    create_log_file(filename)

                    # Call the attack function
                    logging.info(f'Calling start_attack, is_running = {st.session_state.is_running}')
                    st.session_state.terminating = False
                    start_attack(attack_script, interface_a, duration, sleep, filename, password)
                    st.rerun()
                else:
                    st.error("Please fill in all fields.")


        # Add a spinning indicator when the attack is running
        if st.session_state.is_running:
            st.markdown("<h3 style='color:red;'>🔴 Attack is running...</h3>", unsafe_allow_html=True)
            # logging.info('The attack is running display is showing')
            nested_col1, nested_col2 = st.columns(2, vertical_alignment="center")
            with nested_col1:
                if not st.session_state.terminating:
                    if st.button("Stop Attack"):
                        # st.markdown("<h3 style='color:orange;'>🟠 Terminating Attack...</h3>", unsafe_allow_html=True)
                        st.session_state.terminating = True
                        # logging.info(f'Stop button clicked, calling stop_attack()')
            with nested_col2:
                if st.session_state.terminating:
                    with st.spinner('Terminating Attack'):
                        stop_attack()
                    # logging.info(f'Completed stop_attack, is_running = {st.session_state.is_running}')
                    st.rerun()

    with col1:
        st.markdown("### Configure Monitor")
        # Input fields for each argument
        timeout = st.text_input("Timeout (in seconds, leave blank for continuous)", help="Leave blank to run continuously")
        model_path = st.text_input("Model weights path", value="path/to/your/model.weights",
                                   help="Enter the full path to the model weights file")
        interface = st.text_input("Network interface to listen on", value="eth0",
                                  help="Enter the network interface to listen on")


        # Button to run the script
        if st.button("Start TIMESAFE Detection"):
            # Construct the command with the provided inputs
            command = [
                "python3", "pipeline_demo2.py",
                "-m", model_path,
                "-i", interface
            ]
            # Add timeout only if greater than 0 (not continuous)
            if timeout:
                command.extend(["-t", str(timeout)])


            # Display the command to be run
            st.write("Running command:", " ".join(command))

            # Placeholder for status updates
            status_placeholder = st.empty()

            try:
                # Start the subprocess with Popen to read output in real-time
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Read the process output continuously
                while True:
                    # Read a line of output
                    output_line = process.stdout.readline()

                    # If the process has terminated and no output is left, break the loop
                    if output_line == '' and process.poll() is not None:
                        break

                    # Check for the output containing the predicted label
                    if "Predicted label:" in output_line:
                        # Extract the predicted label (0 or 1)
                        predicted_label = int(output_line.strip().split()[-1])

                        # Update the UI based on the predicted label
                        if predicted_label == 0:
                            status_placeholder.markdown(
                                "<h3 style='color:green;'>🟢 SAFE - No Attack detected</h3>",
                                unsafe_allow_html=True
                            )
                        elif predicted_label == 1:
                            status_placeholder.markdown(
                                "<h3 style='color:red;'>🔴 Malicious Activity Detected</h3>",
                                unsafe_allow_html=True
                            )

                    # Sleep for a short duration to prevent busy-waiting
                    time.sleep(0.1)

            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                # If the process ends, handle exit status and errors
                return_code = process.poll()
                if return_code:
                    st.error(f"Error: {process.stderr.read()}")
                else:
                    st.success("Process completed successfully.")
                process.stdout.close()
                process.stderr.close()

if __name__ == "__main__":
    main()
