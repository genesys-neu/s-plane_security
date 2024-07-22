# Testbed Environment

## Testbed Setup
Quick overview of the setup and the structure of each element in the Testbed

### Overview
The Testbed Environment consists of 3 elements: the DU, the RU and the attacker and they are directly connected through a switch. 
- **RU:** Runs the PTP Protocol as Slave Clock and runs the Open Fronthaul Background Traffic 
- **DU:** Runs the PTP Protocol as Master Clock, runs the Open Fronthaul Background Traffic, and can either run the pipeline and the data collection. As for the data collection the DU listens to the proper interface and stores the PTP traffic. It processes the information in a *.csv* file for ease of readability and save memory resources and perform the offline detection of the attack. **TO DETERMINE WHETHER TO USE THE SNIFF MODULE FOR LIVE DETECTION OR TCPDUMP**
- **Attacker:** Performs the attacks on the PTP Protocol
  
### DU
- **Automated Scripts**: This folder consists of a set of scripts that automate the data collection and the pipeline tests. In particular we have a python script to run the Open Fronthaul Background Traffic (*automated_bg_traffic.py*) which initializes the synchronization between the DU and RU and starts the processes, one for the data collection (*automated_data_collection.py*) and another for the benign traffic data collection (*automated_benign_data_collection.py*). The data collection needs the synchronization between the DU and the Attacker while the benign traffic data collection can run independently. Eventually, we also have the script to test the detection of the pipeline (*automated_test_DU.py*).
- **Background Traffic Generator: MUST CONTAIN THE BACKGROUND TRACES, BUT THEY ARE TOO BIG TO UPLOAD ON GITHUB** Contains the script that actually reads the Background Traces and replicates the traffic over the interface. This script is started by the *automated_bg_traffic.py* after the synchronization is established
- **Pipeline Scripts:** Contains the subfolder *Models* which contains all the trained Machine Learning models that can be used by the pipeline for the detection. *train_test.py* script implements an LSTM-based classifier for time-series data. It includes functionalities for training the model, evaluating its performance, and generating confusion matrices. We have two different solutions for the pipeline. *pipeline.py* leverages Machin Learning model for the prediction while *pipeline_heuristic.py* is a rule based solution for attack detection.
- **Process Data:** *pcap_csv_converter.py* converts *.pcap* files stored with all PTP traffic in the interface into a *.csv* file as first phase of data processing. *data_gen.py* takes the new csv file from the previous scripts, formats data to be used as training dataset for the ML model (also performs an offline detection of the attacks and labels each packet, essential step for the training phase) **THE FILE IN THIS FOLDER IS DIFFERENT FROM THE OTHER VERSION ON GITHUB**

### RU

### Attacker

## Running the Scripts

### Open Fronthaul Background Traffic
#### Requirements

### Benign Data Collection
#### Requirements

### Attacks Data Collection
#### Requirements

### Dataset Generation
#### Requirements

### Testing the Pipeline
#### Requirements

