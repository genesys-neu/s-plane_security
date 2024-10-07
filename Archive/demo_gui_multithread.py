import streamlit as st
import threading
import time
import queue

# Queue to store monitor and attack outputs safely between threads
monitor_queue = queue.Queue()
attack_queue = queue.Queue()

# Function to run the monitor process
def run_monitor(command):
    time.sleep(2)  # Simulate process delay
    for i in range(10):  # Simulate monitor output for 10 iterations
        monitor_queue.put(f"Monitor running... iteration {i}")
        print(f"Monitor running... iteration {i}")
        time.sleep(1)  # Simulate continuous output

# Function to run the attack process
def run_attack(command):
    time.sleep(2)  # Simulate process delay
    for i in range(10):  # Simulate attack running for 10 iterations
        attack_queue.put(f"Attack running... iteration {i}")
        print(f"Attack running... iteration {i}")
        time.sleep(1)  # Simulate continuous output

# Initialize session state variables if they don't exist
if 'monitor_thread' not in st.session_state:
    st.session_state.monitor_thread = None
if 'monitor_output' not in st.session_state:
    st.session_state.monitor_output = []

if 'attack_thread' not in st.session_state:
    st.session_state.attack_thread = None
if 'attack_output' not in st.session_state:
    st.session_state.attack_output = []

def main():
    # Streamlit UI layout
    st.title("TIMESAFE Monitoring and Attack Demo")

    # Column 1: Monitor
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Configure Monitor")
        model_path = st.text_input("Model weights path", value="s-plane_security/Transformer/best_model_tr.3.40.pth")
        interface = st.text_input("Network interface to listen on", value="enp1s0f1np1")
        timeout = st.text_input("Timeout (leave blank for continuous)")

        # Start monitor button
        if st.button("Start TIMESAFE Detection"):
            if st.session_state.monitor_thread is None or not st.session_state.monitor_thread.is_alive():
                command = f"python3 s-plane_security/pipeline_demo2.py -m {model_path} -i {interface}"
                if timeout:
                    command += f" -t {timeout}"
                st.session_state.monitor_thread = threading.Thread(target=run_monitor, args=(command,))
                st.session_state.monitor_thread.start()

        # Update monitor output by reading from the queue
        while not monitor_queue.empty():
            st.session_state.monitor_output.append(monitor_queue.get())
            print(f'monitor queue length {len(monitor_queue)}')

        # Display monitor output
        st.markdown("### Monitor Output")
        for output in st.session_state.monitor_output:
            st.write(output)

    # Column 2: Attack
    with col2:
        st.markdown("### Configure Attack")
        attacks = ["Spoofing Attack", "Replay Attack"]
        selected_attack = st.selectbox("Select Attack Type:", attacks)

        # Start attack button
        if st.button("Start Attack"):
            if st.session_state.attack_thread is None or not st.session_state.attack_thread.is_alive():
                attack_command = f"sudo python3 {selected_attack}.py"  # Replace with actual command
                st.session_state.attack_thread = threading.Thread(target=run_attack, args=(attack_command,))
                st.session_state.attack_thread.start()

        # Update attack output by reading from the queue
        while not attack_queue.empty():
            st.session_state.attack_output.append(attack_queue.get())
            print(f'attack queue length {len(attack_queue)}')

        # Display attack output
        st.markdown("### Attack Output")
        for output in st.session_state.attack_output:
            st.write(output)

if __name__ == "__main__":
    main()
