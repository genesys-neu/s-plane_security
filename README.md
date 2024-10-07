# TIMESAFE
Timing Interruption Monitoring and Security Assessment for Fronthual Environments

## TIMESAFE Monitor GUI

### Overview
This Streamlit GUI allows you to visually monitor a remote DU. It uses the pipeline_demo2.py script deployed on the remote RU.

### Usage
1. **Requirements**:
   - streamlit, paramiko, seaborn, matplotlib.pyplot

3. **Update SSH Configs**:
   - Ensure you update the SSH configs for the remote RU in both the 'start_monitor' and stop_monitor' functions

3. **Running the Script**:
   - Run the script with the following command:
     ```
     streamlit run monitor_app.py
     ```

## TIMESAFE Attack GUI

### Overview
This Streamlit GUI allows you to choose different attack parameters and execute them from a remote device. 

### Usage
1. **Requirements**:
   - streamlit, paramiko, itertools

3. **Update SSH Configs**:
   - Ensure you update the SSH configs for the remote RU in both the 'start_attack' and stop_attack' functions

3. **Running the Script**:
   - Run the script with the following command:
     ```
     streamlit run attack_app.py
     ```

## Other tools and scripts

- Data Collection and Pre-processing tools are in the ./DataCollectionPTP directory

- All our models trained to be deployed at the DU are in the ./DU_model directory

- The ./Testbed directory contains all our test bed scripts, including the actual attack scripts

- The ./Production_Environment folder contains visualization tools and results from the production environment.
