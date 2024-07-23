# Testbed Environment

## Testbed Setup
Quick overview of the setup and the structure of each element in the Testbed

### Overview
The Testbed Environment consists of 3 elements: the DU, the RU and the attacker and they are directly connected through a switch. 
- **RU:** Runs the PTP Protocol as Slave Clock and runs the Open Fronthaul Background Traffic 
- **DU:** Runs the PTP Protocol as Master Clock, runs the Open Fronthaul Background Traffic, and can either run the pipeline and the data collection. As for the data collection the DU listens to the proper interface and stores the PTP traffic. It processes the information in a `.csv` file for ease of readability and save memory resources and perform the offline detection of the attack. **TO DETERMINE WHETHER TO USE THE SNIFF MODULE FOR LIVE DETECTION OR TCPDUMP**
- **Attacker:** Performs the attacks on the PTP Protocol
  
### DU
- **Automated Scripts:** This folder consists of a set of scripts that automate the data collection and the pipeline tests. In particula, one for the data collection (`automated_data_collection.py`) and another for the benign traffic data collection (`automated_benign_data_collection.py`). The data collection needs the synchronization between the DU and the Attacker while the benign traffic data collection can run independently. Eventually, we also have the script to test the detection of the pipeline (`automated_test_DU.py`).
- **Background Traffic Generator: MUST CONTAIN THE BACKGROUND TRACES, BUT THEY ARE TOO BIG TO UPLOAD ON GITHUB** Contains a script to run the Open Fronthaul Background Traffic (`automated_bg_traffic.py`) which initializes the synchronization between the DU and RU and starts the processesthe script that actually reads the Background Traces and replicates the traffic over the interface in the `OFH_tgen.py` script. The traces are contained in the `Cleande_CU__plane_traces` folder
- **Pipeline Scripts:** Contains the subfolder `Models` which contains all the trained Machine Learning models that can be used by the pipeline for the detection. `train_test.py` script implements an LSTM-based classifier for time-series data. It includes functionalities for training the model, evaluating its performance, and generating confusion matrices. We have two different solutions for the pipeline. `pipeline.py` leverages Machin Learning model for the prediction while `pipeline_heuristic.py` is a rule based solution for attack detection.
- **Process Data:** `pcap_csv_converter.py` converts `.pcap` files stored with all PTP traffic in the interface into a `.csv` file as first phase of data processing. `data_gen.py` takes the new csv file from the previous scripts, formats data to be used as training dataset for the ML model (also performs an offline detection of the attacks and labels each packet, essential step for the training phase) **THE FILE IN THIS FOLDER IS DIFFERENT FROM THE OTHER VERSION ON GITHUB**

### RU
- **automated_bg_traffic:** initializes the synchronization between the DU and RU and starts the processes.
- **OFH_tgen.py:** reads the `.csv` files that represent the single Backgrount Traffic traces and replicates them
- **Cleaned_CU_plane_traces:** **THIS FOLDER MUST BE UPLOADED BUT CSV FILES ARE TOO MASSIVE** contains the four Open Fronthaul Background Traces that must be replicated
  
### Attacker
- **Scripts:** This subfolder contains all the scripts to perform the attacks on PTP. We have two versions for the the Spoofing attack. `Announce_attack.py` (used on the testbed) listens to the interface, collects relevant PTP information and crafts new messages. `Craft_Announce_Attack.py` (used on the production ready network)on the other hand does not need to sniff for any incoming packets. All information needed are already stored and this script just sends PTP packets over the network. The Replay attack, `Sync_Attack.py`, is used in the testbed and works for a two-steps PTP. It waits for a pair of Sync and FollowUp packets and replays them over the network. `PTPDoS.py` is not a real DoS attack. It simply crafts malformed packets and sent them to the network. it is used to test the pipeline with unusual PTP traffic that has not been used as training data for the model. The other python scripts (`AnnouncePTP.py`, `SyncPTP.py`) are python classes of the two PTP packets. They contain methods to modify the fields and are imported in the attacks scripts.
- **automated_test_attacker.py:** This script first initiates the connection and synchronization with the DU, then it chooses a random attack and a random value for recovery time and attack time. Then, it runs the attacks and stores log about what type of attack has been performed and the related start and end timestamps

## Running the Scripts

### Open Fronthaul Background Traffic
This process is performed by the DU and RU. The files involved have the same nomenclature for both the machines. The `automated_bg_traffic.py` script first estabish the synchronization between the RU and DU, then it invokes the the `OFH_tgen.py` script that reads the Background Traffic contained in `.csv` files contained in the `Cleaned_CU_plane_traces` folder, crafts every single packet and sends it through the proper interface. The four Background Traces are:
- `run1-12sep-aerial-udpDL.csv`
- `run1-8sep-aerial-increasingDL-withUL.csv`
- `run2-8sep-aerial-increasingDL-noUL.csv`
- `run3-8sep-aerial-maxDLUL.csv`
  
These files are read simultaneously by both the DU and RU to replicate the traffic until the experiment is over.

#### Requirements
- Python 3

#### Usage
For both the DU and RU the `OFH_tgen.py` and `automated_bg_traffic.py` script must be in the same folder along with the `Cleande_CU_plane_traces` folder. The procedure is the same for both the DU and the RU but **the DU must be started first**

1. **Installation**:
   - Ensure that Python 3.x is installed on your system.

2. **Running the Script**:
   - Navigate to the directory containing the script `automated_bg_traffic.py`.
   - in the DU, run the script with the following command:
     ```
     sudo python3 autpmated_bg_traffic.py [-i <ip_destination>] [-d <duration>]
     ```
     - in the RU, run the script with the following command:
     ```
     sudo python3 autpmated_bg_traffic.py [-i <ip_destination>] [-d <duration>] [-r]
     ```
     All arguments are mandatory.
     - Replace `<ip_destination>` with the ip of the other machine you want to exchange traffic with.
     - Replace `<duration>` with the duration of the experiment in seconds, it must be the same for both DU and RU.
     - `-r` is the flag that specifies that the specific device is the RU

3. **Output**:
   - The script will first establish connection between DU and RU.
   - It will cyclically run the `OFH_tgen.py` script for each trace.
   - The traffic will be printed on terminal.

4. **Customization**:
- Adjust the parameters according to your environment and the duration of your experiment.

### Example
```
sudo python3 automated_bg_traffic.py -i 192.168.40.51 -d 1800 
```
This command starts the script in the DU, it connects to the RU at the given address and sets the duration experiment for 1800 seconds (30 minutes) .

```
sudo python3 automated_bg_traffic.py -i 192.168.40.1 -d 1800 -r
```
This command starts the script in the RU, it connects to the DU at the given address and sets the duration experiment for 1800 seconds (30 minutes) .

### Benign Data Collection
This task is only performed by the DU. It is supposed that the Background Traffic mentioned above is running.

#### Requirements
- Python3
- 
#### Usage


### Attacks Data Collection
#### Requirements
#### Usage

### Dataset Generation
#### Requirements
#### Usage

### Testing the Pipeline
#### Requirements
#### Usage

