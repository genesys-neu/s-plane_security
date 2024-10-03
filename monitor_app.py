import time
import streamlit as st
import paramiko
import logging
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("monitor_app.log"),
    logging.StreamHandler()
])

# Global variables to manage the attack process and status
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'terminating' not in st.session_state:
    st.session_state.terminating = False


def start_monitor(model_path, interface, timeout):
    ssh_host = "10.188.57.241"
    ssh_user = "orantestbed"
    ssh_password = "op3nran"
    prediction_count = 0

    command = f"python3 s-plane_security/pipeline_demo2.py -m {model_path} -i {interface}"
    if timeout:
        command += f" -t {timeout}"

    st.write("Running command on remote server:", command)

    # Stream real-time output and update UI

    status_placeholder = st.empty()

    graph_placeholder = st.empty()

    # Keep track of last 100 responses for bar chart
    response_history = []

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ssh_host, username=ssh_user, password=ssh_password)

        stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)

        while st.session_state.is_running:
            output_line = stdout.readline()
            logging.info(f'{output_line}')

            if "Predicted label:" in output_line:
                prediction_count += 1
                predicted_label = int(output_line.strip().split()[-1])
                response_history.append(predicted_label)
                if len(response_history) > 200:
                    response_history.pop(0)

                if predicted_label == 0:
                    status_placeholder.markdown("<h3 style='color:green;'>🟢 SAFE - No Attack detected</h3>",
                                                unsafe_allow_html=True)
                elif predicted_label == 1:
                    status_placeholder.markdown("<h3 style='color:red;'>🔴 Malicious Activity Detected</h3>",
                                                unsafe_allow_html=True)
                if prediction_count > 20:
                    prediction_count = 0
                    with graph_placeholder.container():
                        plt.close()
                        # Check unique values in response_history
                        unique_values = set(response_history)

                        # Define color palette based on unique values
                        if unique_values == {0}:
                            colors = ['green']  # Only 0s
                        elif unique_values == {1}:
                            colors = ['red']  # Only 1s
                        else:
                            colors = ['green', 'red']  # Mix of 0s and 1s
                        # Create and display heatmap
                        fig, ax = plt.subplots(figsize=(20, 2))
                        ax = sns.heatmap([response_history], cmap=sns.color_palette(colors), cbar=False, xticklabels=False,
                                         yticklabels=False)
                        ax.set_xlabel('Time')
                        ax.set_xlabel('Time', fontsize=24)  # Adjust fontsize as needed
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


def stop_monitor():
    logging.info("Attempting to stop the monitor...")

    if st.session_state.is_running:
        remote_host = "10.188.57.241"  # Replace with the actual remote server address
        remote_user = "orantestbed"  # Replace with the actual remote user
        remote_password = "op3nran"  # Replace with the actual remote user's password

        try:
            # Initialize SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect to the remote server
            ssh.connect(remote_host, username=remote_user, password=remote_password)

            # Use `pgrep` to find all processes related to the script_name
            find_process_command = f"pgrep -f pipeline_demo2"
            stdin, stdout, stderr = ssh.exec_command(find_process_command)
            pids = stdout.read().decode().strip().split()

            if pids:
                logging.info(f"Found PIDs: {pids}")
                for pid in pids:
                    # Kill each process found with the script_name
                    kill_command = f"echo {remote_password} | sudo -S kill -9 {pid}"
                    ssh.exec_command(kill_command)
                    logging.info(f"Sent kill command for PID: {pid}")
                # st.success("Monitor stopped successfully.")
            else:
                st.warning(f"No running processes found.")
                logging.warning(f"No PIDs found")

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
    model_path = st.text_input("Model weights path",
                               value="s-plane_security/DU_model/Transformer/best_model_tr_new.3.40.pth",
                               help="Enter the full path to the model weights file")
    interface = st.text_input("Network interface to listen on", value="enp1s0f1np1",
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
        start_monitor(model_path, interface, timeout)

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
