import streamlit as st
import subprocess
import os
import random
import psutil
import csv
import threading

# Directory containing the attack scripts
directory = 'Testbed/PipelineTestAttacker/Scripts/'

# Global variables to manage the attack process and status
attack_process = None
is_running = False
attack_thread = None


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


# Function to start the attack in a background thread
def run_attack(attack, interface, duration, sleep, filename, password):
    global attack_process
    global is_running

    # Prepare the command
    command = ["sudo", "-S", "python3", attack, '-i', interface, '-d', str(duration), '-s', str(sleep), '-l', filename]

    # Run the command with the password
    attack_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    attack_process.stdin.write(password.encode() + b'\n')  # Write the password to stdin
    attack_process.stdin.flush()

    # Wait for the process to complete
    attack_process.wait()
    is_running = False  # Mark the attack as stopped


# Function to handle starting the attack
def start_attack(attack, interface, duration, sleep, filename, password):
    global attack_thread
    global is_running

    # Only start the attack if no attack is running
    if not is_running:
        is_running = True
        attack_thread = threading.Thread(target=run_attack, args=(attack, interface, duration, sleep, filename, password))
        attack_thread.start()


def stop_attack():
    global attack_process
    global is_running
    st.write("Attempting to stop the attack...")

    if attack_process is not None and is_running:
        try:
            st.write(f"Terminating process with PID: {attack_process.pid}")
            attack_process.terminate()  # Attempt to terminate the process
            attack_process.wait(timeout=5)  # Wait for the process to exit, with a timeout
            is_running = False
            st.write("Attack stopped successfully.")
        except subprocess.TimeoutExpired:
            st.write("Process did not terminate in time, killing it...")
            attack_process.kill()  # Force kill if it doesn't exit in time
            is_running = False
            st.write("Attack process was forcefully killed.")
        except Exception as e:
            st.error(f"Error stopping the attack: {e}")
    else:
        st.write("No attack process to stop.")


def main():
    global is_running

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

    if st.button("Start Attack"):
        if interface and output_folder and password:

            # Create the output folder if it does not exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Extract the script name minus the .py extension
            script_name = os.path.basename(attack_script).replace(".py", "")

            # Start the attack and generate the log filename
            filename = os.path.join(output_folder, f'{script_name}.{interface}.{duration}.{sleep}.csv')
            create_log_file(filename)

            # Call the attack function
            st.write(
                f'Starting attack: {selected_attack} for {duration} seconds with {sleep} seconds of sleep.'
            )
            start_attack(attack_script, interface, duration, sleep, filename, password)
            is_running = True

        else:
            st.error("Please fill in all fields.")

    # Add a spinning indicator when the attack is running
    if is_running:
        st.markdown("<h3 style='color:red;'>🔴 Attack is running...</h3>", unsafe_allow_html=True)
        if st.button("Stop Attack"):
            stop_attack()
    else:
        st.markdown("<h3 style='color:green;'>🟢 No attack running</h3>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
