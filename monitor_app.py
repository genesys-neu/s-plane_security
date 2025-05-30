import time
import streamlit as st
import paramiko
import logging
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import yaml


## PARAMETERS ##

# Load configuration from YAML file
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Access the variables from the YAML configuration
REMOTE_HOST = config["REMOTE_HOST"]
REMOTE_USER = config["REMOTE_USER"]
REMOTE_PASS = config["REMOTE_PASS"]

CONTAINERIZED = config["CONTAINERIZED"]
CONTAINER_NAME = config["CONTAINER_NAME"]
INTERFACE_ATTACK = config["INTERFACE_ATTACK"]

ROOT_DIRECTORY = config["ROOT_DIRECTORY"]
MONITOR_DIRECTORY = config["MONITOR_DIRECTORY"]
MODELS_DIRECTORY = config["MODELS_DIRECTORY"]
LOGS_DIRECTORY = config["LOGS_DIRECTORY"]


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(LOGS_DIRECTORY + "monitor_app.log"),
    logging.StreamHandler()
])

# Global variables to manage the attack process and status
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'terminating' not in st.session_state:
    st.session_state.terminating = False


def start_monitor(model_path, interface, timeout):
    prediction_count = 0

    command = (f"docker exec {CONTAINER_NAME} " if CONTAINERIZED else "") +\
              f" python3 {ROOT_DIRECTORY}pipeline_demo2.py -m {model_path} -i {interface}"
    
    if timeout:
        command += f" -t {timeout}"

    st.write("Running command on remote server:", command)

    # Stream real-time output and update UI

    status_placeholder = st.empty()

    graph_placeholder = st.empty()

    # Keep track of last 100 responses for bar chart
    response_history = [-1] * 120
    time_history = []

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=REMOTE_PASS)

        stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
        start = time.time()

        while st.session_state.is_running:
            output_line = stdout.readline()
            logging.info(f'{output_line}')

            if "Predicted label:" in output_line:
                predicted_label = int(output_line.strip().split()[-1])
                time_history.append(predicted_label)

                if predicted_label == 0:
                    status_placeholder.markdown("<h3 style='color:green;'>🟢 SAFE - No Attack detected</h3>",
                                                unsafe_allow_html=True)
                elif predicted_label == 1:
                    status_placeholder.markdown("<h3 style='color:red;'>🔴 Malicious Activity Detected</h3>",
                                                unsafe_allow_html=True)
                if time.time()-start >= 1:
                    # Check the majority in time_history over the last second
                    zeros = time_history.count(0)
                    ones = time_history.count(1)
                    # Append the majority label to response_history
                    if zeros > ones:
                        response_history.append(0)
                    else:
                        response_history.append(1)
                    if len(response_history) > 120:
                        response_history.pop(0)

                    # Clear time_history for the next second
                    time_history.clear()
                    start = time.time()

                    with graph_placeholder.container():
                        plt.close()
                        # Define the color palette to handle -1 (white), 0 (green), and 1 (red)
                        custom_colors = ['#ffffff', 'green', 'red']  # white for None (-1), green for 0, red for 1
                        cmap = sns.color_palette(custom_colors)

                        # Create and display heatmap
                        fig, ax = plt.subplots(figsize=(20, 2))
                        ax = sns.heatmap([response_history], cmap=cmap, cbar=False, xticklabels=False,
                                         yticklabels=False, vmin=-1, vmax=1,)
                        # ax.set_xlabel('Time')
                        ax.set_xticks([0, 120 // 2, 120 - 1])  # 0 (left), 60 (middle), 119 (right)
                        ax.set_xticklabels(['-120', '-60', '0'], fontsize=24)  # Time labels for x-axis
                        ax.set_xlabel('Time (seconds)', fontsize=36)  # Adjust fontsize as needed
                        st.pyplot(fig)

            elif "exited" in output_line:
                logging.info(f'Exited output: {output_line}')
                st.session_state.is_running = False
                break


    except Exception as e:
        st.error(f"Error starting monitor on remote server: {e}")
        st.session_state.is_running = False
        logging.error(f"Exception in start_monitor: {e}")

    finally:
        # logging.info('Reached the finally statement')
        # tail_process.terminate()  # Ensure the tail process is terminated
        return_code = stdout.channel.recv_exit_status()

        if return_code != 0:
            st.error(f"Error: {stderr.read().decode()}")
        else:
            st.success("Process completed successfully.")

        ssh.close()
        st.session_state.is_running = False  # Set to False when the process ends
        time.sleep(5)
        st.rerun()


def kill_processes(ssh, pgrep_name):
    # Use `pgrep` to find all processes related to the script_name
    find_process_command = f"pgrep -f {pgrep_name}"
    stdin, stdout, stderr = ssh.exec_command(find_process_command)
    pids = stdout.read().decode().strip().split()

    if pids:
        logging.info(f"Found PIDs: {pids}")
        for pid in pids:
            # Kill each process found with the script_name
            kill_command = f"echo {REMOTE_PASS} | sudo -S kill -9 {pid}"
            ssh.exec_command(kill_command)
            logging.info(f"Sent kill command for PID: {pid}")
        # st.success("Monitor stopped successfully.")
    else:
        st.warning(f"No running processes found.")
        logging.warning(f"No PIDs found")


def stop_monitor():
    logging.info("Attempting to stop the monitor...")

    if st.session_state.is_running:
        try:
            # Initialize SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            ssh.connect(REMOTE_HOST, username=REMOTE_USER, password=REMOTE_PASS)

            kill_processes(ssh, 'pipeline_demo2')
            kill_processes(ssh, 'tcpdump')

            st.session_state.is_running = False
            st.session_state.terminating = False
            st.success("Monitor stopped successfully.")

        except Exception as e:
            st.error(f"Error stopping the monitor: {e}")
            logging.error(f"Exception in stop_monitor: {e}")

        finally:
            # Close the SSH connection
            ssh.close()
    else:
        st.warning("Monitor is not currently running.")
        logging.warning("Stop monitor attempted, but the monitor is not running.")


def main():
    col1, col2, col3 = st.columns(3, vertical_alignment="center")

    # Replace 'logo1.png' and 'logo2.png' with the paths or URLs to your logos
    with col1:
        st.image('Images/UTA2_logo.png')
    with col2:
        st.image('Images/NEU_logo.png')
    with col3:
        st.image('Images/purdue_logo.png')

    # Streamlit UI with formatted title using HTML
    st.markdown(
        "<h1 style='text-align: center;'>TIMESAFE:"
        "<div style='text-align: center; font-size: 16px;'><u>T</u>iming <u>I</u>nterruption <u>M</u>onitoring and "
        "<u>S</u>ecurity <u>A</u>ssessment for <u>F</u>ronthaul <u>E</u>nvironments</h1></div>",
        unsafe_allow_html=True
    )

    # Add vertical space
    st.markdown("<br>", unsafe_allow_html=True)  # This adds two line breaks for vertical space

    st.markdown("<h3 style='text-align: center;'>Monitor Configuration</h1>", unsafe_allow_html=True)

    timeout = st.text_input("Timeout (in seconds, leave blank for continuous)", help="Leave blank to run continuously")
    model = st.text_input("Model weights path",
                               value="best_model_prod1_tr.2.32.pth",
                               help="Enter the model weights file")
    interface = st.text_input("Network interface to listen on", value="ens6f0",
                              help="Enter the network interface to listen on")

    # status_placeholder = st.empty()
    logging.info(f'Current session state: {st.session_state.is_running}')

    bcol1, bcol2 = st.columns([.3, .7])
    with bcol1:
        if st.button("Start TIMESAFE Detection"):
            if not st.session_state.is_running:
                st.session_state.is_running = True
                logging.info(f'updated session state: {st.session_state.is_running}')
                # time.sleep(5)
                # st.rerun()
        logging.info(f'Stop button session state: {st.session_state.is_running}')

    with bcol2:
        nested_col1, nested_col2 = st.columns(2, gap="large")
        with nested_col1:
            logging.info(f'Current terminating session state: {st.session_state.terminating}')
            if st.button("Stop TIMESAFE Detection"):
                if not st.session_state.terminating and st.session_state.is_running:
                    st.session_state.terminating = True
                    # st.rerun()
        with nested_col2:
            if st.session_state.terminating:
                with st.spinner('Stopping Detection...'):
                    stop_monitor()
                    time.sleep(2)
                st.rerun()

    if st.session_state.is_running:
        start_monitor(MODELS_DIRECTORY + model, interface, timeout)

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
