import time
import streamlit as st
import paramiko
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("attack_app.log"),
    logging.StreamHandler()
])


def main():
    col1, col2, col3 = st.columns([.3, .4, .3], vertical_alignment="center")

    # Replace 'logo1.png' and 'logo2.png' with the paths or URLs to your logos
    with col1:
        st.image('Images/purdue_logo.png')
    with col2:
        st.image('Images/NEU_logo.png')
    with col3:
        st.image('Images/UTA_logo.png')

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

    ssh_host = "10.188.57.241"
    ssh_user = "orantestbed"
    ssh_password = "op3nran"

    timeout = st.text_input("Timeout (in seconds, leave blank for continuous)",
                             help="Leave blank to run continuously")
    model_path = st.text_input("Model weights path",
                                value="s-plane_security/Transformer/best_model_tr.3.40.pth",
                                help="Enter the full path to the model weights file")
    interface = st.text_input("Network interface to listen on", value="enp1s0f1np1",
                               help="Enter the network interface to listen on")

    status_placeholder = st.empty()
    if st.button("Start TIMESAFE Detection"):
        command = f"python3 s-plane_security/pipeline_demo2.py -m {model_path} -i {interface}"
        if timeout:
            command += f" -t {timeout}"

        st.write("Running command on remote server:", command)

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ssh_host, username=ssh_user, password=ssh_password)

            stdin, stdout, stderr = ssh.exec_command(command)

            while True:
                output_line = stdout.readline()
                logging.info(f'Output: {output_line}')
                if output_line == '' and stdout.channel.exit_status_ready():
                    break

                if "Predicted label:" in output_line:
                    predicted_label = int(output_line.strip().split()[-1])
                    if predicted_label == 0:
                        status_placeholder.markdown("<h3 style='color:green;'>🟢 SAFE - No Attack detected</h3>", unsafe_allow_html=True)
                    elif predicted_label == 1:
                        status_placeholder.markdown("<h3 style='color:red;'>🔴 Malicious Activity Detected</h3>", unsafe_allow_html=True)

                time.sleep(0.1)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        finally:
            return_code = stdout.channel.recv_exit_status()
            if return_code != 0:
                st.error(f"Error: {stderr.read().decode()}")
            else:
                st.success("Process completed successfully.")

            ssh.close()

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
