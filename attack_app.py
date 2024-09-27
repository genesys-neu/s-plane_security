import time
import streamlit as st
import os
import random
import logging
import paramiko
import itertools


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("attack_app.log"),
    logging.StreamHandler()
])

# Directory containing the attack scripts
directory = 's-plane_security/Testbed/PipelineTestAttacker/Scripts/'

# Global variables to manage the attack process and status
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'terminating' not in st.session_state:
    st.session_state.terminating = False
if 'attack_pid' not in st.session_state:
    st.session_state.attack_pid = None  # To store the PID of the remote attack process
if 'script_name' not in st.session_state:
    st.session_state.script_name = None

# List of images to display
image_files_announce = ['Images/spoofing1.png', 'Images/spoofing2.png', 'Images/spoofing3.png', 'Images/spoofing4.png']


# Function to display images continuously while an attack is running
def display_cycling_images():
    # Create an empty container to hold the image
    image_placeholder = st.empty()

    # Create an iterator for the images
    if st.session_state.script_name == 'Announce_Attack':
        image_iterator = itertools.cycle(image_files_announce)

    while st.session_state.is_running:  # Keep looping while the attack is running
        # Get the next image from the iterator
        current_image = next(image_iterator)

        # Update the placeholder with the current image
        image_placeholder.image(current_image, width=300)

        # Pause for a short period before showing the next image
        time.sleep(1)  # Adjust the sleep time as needed


def start_attack(attack, interface, duration, sleep, filename):
    remote_host = "10.188.57.241"  # Replace with the actual remote server address
    remote_user = "orantestbed"  # Replace with the actual remote user
    remote_password = "op3nran"  # Replace with the actual remote user's password

    if not st.session_state.is_running:
        st.session_state.is_running = True

        try:
            # Initialize SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            ssh.connect(remote_host, username=remote_user, password=remote_password)

            # Define the absolute path for the log file
            log_directory = "/home/orantestbed/s-plane_security/tmp"  # Update this path as needed
            abs_filename = f"{log_directory}/{filename}"  # Absolute path for the log file
            logging.info(f'Absolute filename: {abs_filename}')

            # Create the log directory if it doesn't exist
            create_dir_command = f"mkdir -p {log_directory}"
            ssh.exec_command(create_dir_command)

            # Create the log file on the remote server with headers
            create_log_command = f'echo "attack type,attack start,attack end" > {abs_filename}'
            ssh.exec_command(create_log_command)

            # Prepare the command to start the attack in the background
            # Redirect output to /dev/null and use nohup to run it independently of the SSH session
            attack_command = (
                f"echo {remote_password} | sudo -S python3 {attack} "
                f"-i {interface} -d {duration} -s {sleep} -l {abs_filename} "
                f"> /dev/null 2>&1 & echo $!"
            )

            # Execute the attack command
            stdin, stdout, stderr = ssh.exec_command(attack_command)

            # Retrieve the PID of the attack process
            pid = stdout.read().decode().strip()
            if pid:
                st.session_state.attack_pid = pid
                st.write(f'Starting attack: {os.path.basename(attack)} with PID {pid}')
                logging.info(f'Started attack with PID: {pid}')
            else:
                st.error("Failed to retrieve PID. Attack may not have started.")
                st.session_state.is_running = False

        except Exception as e:
            st.error(f"Error starting attack on remote server: {e}")
            st.session_state.is_running = False
            logging.error(f"Exception in start_attack: {e}")

        finally:
            # Close the SSH connection
            ssh.close()


def stop_attack():
    logging.info("Attempting to stop the attack...")

    if st.session_state.attack_pid and st.session_state.is_running:
        remote_host = "10.188.57.241"  # Replace with the actual remote server address
        remote_user = "orantestbed"  # Replace with the actual remote user
        remote_password = "op3nran"  # Replace with the actual remote user's password

        try:
            # Initialize SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            ssh.connect(remote_host, username=remote_user, password=remote_password)

            # Command to kill the process using the stored PID
            kill_command = f"echo {remote_password} | sudo -S kill -9 {st.session_state.attack_pid}"
            ssh.exec_command(kill_command)

            # Use `pgrep` to find all processes related to the script_name
            find_process_command = f"pgrep -f {st.session_state.script_name}"
            stdin, stdout, stderr = ssh.exec_command(find_process_command)
            pids = stdout.read().decode().strip().split()

            if pids:
                logging.info(f"Found PIDs: {pids}")
                for pid in pids:
                    # Kill each process found with the script_name
                    kill_command = f"echo {remote_password} | sudo -S kill -9 {pid}"
                    ssh.exec_command(kill_command)
                    logging.info(f"Sent kill command for PID: {pid}")
                st.success("Attack stopped successfully.")
            else:
                st.warning(f"No running processes found for {st.session_state.script_name}.")
                logging.warning(f"No PIDs found for {st.session_state.script_name}")

            logging.info(f"Sent kill command for PID: {st.session_state.attack_pid}")

            st.session_state.is_running = False
            st.session_state.attack_pid = None
            st.success("Attack stopped successfully.")

        except Exception as e:
            st.error(f"Error stopping the attack: {e}")
            logging.error(f"Exception in stop_attack: {e}")

        finally:
            # Close the SSH connection
            ssh.close()
    else:
        st.warning("No attack is currently running.")
        logging.warning("Stop attack attempted, but no attack is running.")


def main():
    col1, col2, col3 = st.columns([1, 1, 1])

    # Replace 'logo1.png' and 'logo2.png' with the paths or URLs to your logos
    with col1:
        st.image('Images/NEU_logo.png', width=200)
    with col2:
        st.image('Images/purdue_logo.png', width=200)
    with col3:
        st.image('Images/UTA_logo.png', width=200)

    # Streamlit UI with formatted title using HTML
    st.markdown(
        "<h1 style='text-align: center;'>TIMESAFE:</h1>"
        "<div style='text-align: center; font-size: 16px;'>"
        "<u>T</u>iming <u>I</u>nterruption <u>M</u>onitoring and "
        "<u>S</u>ecurity <u>A</u>ssessment for <u>F</u>ronthaul "
        "<u>E</u>nvironments</div>",
        unsafe_allow_html=True
    )

    # Add vertical space
    st.markdown("<br>", unsafe_allow_html=True)  # This adds two line breaks for vertical space

    st.markdown("<h3 style='text-align: center;'>Attack Configuration</h3>", unsafe_allow_html=True)

    interface_a = 'enp1s0f1np1'
    output_folder = 'tmp'
    attack_duration = st.selectbox("Select Duration Test:", ["random", "fixed", "continuous"])

    if attack_duration == "random":
        duration = random.randint(10, 30)
        sleep = random.randint(40, 60)
    elif attack_duration == "continuous":
        duration = 600
        sleep = 0
    else:
        duration = st.number_input("Enter the attack duration in seconds:", min_value=1, value=10)
        sleep = st.number_input("Enter the sleep duration in seconds:", min_value=1, value=5)

    attacks = ["Spoofing Attack", "Replay Attack"]
    selected_attack = st.selectbox("Select Attack Type:", attacks)

    attack_script = None
    if selected_attack == "Replay Attack":
        replay_options = st.selectbox("Select Replay Attack Type:", ["Sync only", "Sync and Follow Up"])
        attack_script = os.path.join(directory, "Sync_Attack.py") if replay_options == "Sync only" else os.path.join(
            directory, "Sync_FollowUp_Attack.py")
    elif selected_attack == "Spoofing Attack":
        attack_script = os.path.join(directory, "Announce_Attack.py")

    col1a, col2a = st.columns([1, 1])

    if not st.session_state.is_running:
        with col1a:
            st.markdown("<h3 style='color:green;'>🟢 No attack running</h3>", unsafe_allow_html=True)
            if st.button("Start Attack"):
                # Extract the script name minus the .py extension
                st.session_state.script_name = os.path.basename(attack_script).replace(".py", "")
                logging.info(f'Script name: {st.session_state.script_name}')

                # Start the attack and generate the log filename
                filename = f'{st.session_state.script_name}.{interface_a}.{duration}.{sleep}.csv'
                logging.info(f'Filename is {filename}')

                # Call the attack function
                logging.info(f'Calling start_attack, is_running = {st.session_state.is_running}')
                st.session_state.terminating = False
                start_attack(attack_script, interface_a, duration, sleep, filename)
                time.sleep(5)
                st.rerun()  # Use experimental_rerun for immediate rerun
        with col2a:
            st.image('Images/spoofing0.png', width=300)

    # Add a spinning indicator when the attack is running
    if st.session_state.is_running:
        with col1a:
            st.markdown("<h3 style='color:red;'>🔴 Attack is running...</h3>", unsafe_allow_html=True)
            nested_col1, nested_col2 = st.columns(2, gap="large")
            with nested_col1:
                if not st.session_state.terminating:
                    if st.button("Stop Attack"):
                        st.session_state.terminating = True
            with nested_col2:
                if st.session_state.terminating:
                    with st.spinner('Terminating Attack...'):
                        stop_attack()
                    st.rerun()
        with col2a:
            if not st.session_state.terminating:
                display_cycling_images()


    # Author footnote
    author_text = """
    <p style='font-size: 12px; text-align: center;'>
    Joshua Groen, Simone Di Valerio<sup>&dagger;</sup>, Imtiaz Karim<sup>&Dagger;</sup>, Davide Villa,
    Yiwei Zhang<sup>&Dagger;</sup>, Leonardo Bonati, Michele Polese, Salvatore D'Oro,<br>
    Tommaso Melodia, Elisa Bertino<sup>&Dagger;</sup>, Francesca Cuomo<sup>&dagger;</sup>, Kaushik Chowdhury<sup>§</sup>
    </p>
    <p style='font-size: 12px; text-align: center;'>
    Northeastern University &nbsp;&nbsp; <sup>&dagger;</sup>Sapienza University of Rome &nbsp;&nbsp; 
    <sup>&Dagger;</sup>Purdue University &nbsp;&nbsp; <sup>§</sup>University of Texas at Austin
    </p>
    """

    # Add vertical space
    st.markdown("<br><br>", unsafe_allow_html=True)  # This adds two line breaks for vertical space

    # Display the author footnote at the bottom
    st.markdown(author_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
