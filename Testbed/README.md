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

#### Example
```
sudo python3 automated_bg_traffic.py -i 192.168.40.51 -d 1800 
```
This command starts the script in the DU, it connects to the RU at the given address and sets the duration experiment for 1800 seconds (30 minutes) .

```
sudo python3 automated_bg_traffic.py -i 192.168.40.1 -d 1800 -r
```
This command starts the script in the RU, it connects to the DU at the given address and sets the duration experiment for 1800 seconds (30 minutes) .

### Benign Data Collection
For this task it is supposed that the Background Traffic mentioned above is running. The script consists on several tests of customizable duration. During each test the script sniffs with `tcpdump` the chosen interfaces and saves the pcap file. After that, the pcap file is converted to a csv file using the script `pcap_csv_converter.py` in the `ProcessData` folder, filtering all PTP packets and keeping only the relevant information. We suggest to create one output folder for each experiment

#### Requirements
- Python3
- Scapy 2.6.0
  
#### Usage
This task is only performed in the DU

1. **Installation**:
   - Ensure that Python 3.x is installed on your system.
   - Ensure that `scapy` is installed on your system. if not run the following command to install it:
     ```
     pip install scapy
     ```

2. **Running the Script**:
   - Navigate to the directory containing the script `automated_benign_data_collection.py`.
   - Run the script with the following command:
   ```
   sudo python3 automated_benign_data_collection.py [-i <interface>] [-de <experiment_duration>] [-dt <test_duration>] [-o <output_folder>]
   ```
   All arguments are mandatory.
     - Replace `<interface>` with the interface you want to use to sniff the traffic
     - Replace `<experiment_duration>` with the desired duration of the whole experiment. This value should be a multiple of the `<test_duration>`
     - Replace `<test_duration>` with the desired duration of each single test. This value should be a divisor of the `<experiment_duration>`
     - Replace `<output_folder>` with the folder you want your output to be stored
3. **Output**:
   - Each test will generate a pcap file named `dump_test_number.pcap` and the final csv files will be named `dump_{test_number}.csv`. after generating the csv files all pcap files will be deleted for efficient memory usage. All files are stored in the selected folder
   - Each row in the csv file represent a PTP packet with the following structure:
   ```
   ['Time', 'Source', 'Destination', 'Protocol', 'Length', 'SequenceID', 'MessageType']
   ```
4. **Customization**:
   - Adjust the parameters according to your environment, the duration of your experiment and tests.

#### Example
   ```
   sudo python3 automated_benign_data_collection.py -i enp4s0 -de 1800 -dt 300 -o ../DULogs/
   ```
   This command starts the benign data collection, sniffing at the interface `enp4s0`. the duration of the experiment is 1800 seconds (30 minutes) and the duration of each test is 300 seconds (5 minutes). Output files will be stored in the `DULogs` in the same folder of `AutomatedScripts` folder where the scripts are

### Attacks Data Collection
This task is executed by the DU and the Attacker. First they need to synchronize their processes, to start each tests. Then the Attacker selects a random type of attack, random duration and random recovery time, starts the attack and stores the type of attack, start and end timestamps in a csv file. Each test produces one csv file named `test_attacker_{test_number}.csv` in the selected folder. the DU instead, sniff the traffic from the selected interface and produces a pcap file. The pcap file is then converte into a csv file named `test_DU_{test_number}.csv` in the selected folder. We suggest to create one output folder for each experiment
#### Requirements
- Python3
- Scapy 2.6.0
#### Usage
This task is performed using both the DU and the Attacker

1. **Installation**:
   - Ensure that Python 3.x is installed on your system.
   - Ensure that `scapy` is installed on your system. if not run the following command to install it:
     ```
     pip install scapy
     ```
2. **Running the Script**
   - **DU**
     - Navigate to the directory containing the script `automated_data_collection.py`.
     - Run the script with the following command:
     ```
     sudo python3 automated_data_collection.py [-if <interface>] [-de <experiment_duration>] [-dt <test_duration>] [-o <output_folder>]
     ```
     All arguments are mandatory.
       - Replace `<interface>` with the interface you want to use to sniff the traffic
       - Replace `<experiment_duration>` with the desired duration of the whole experiment. This value should be a multiple of the `<test_duration>`
       - Replace `<test_duration>` with the desired duration of each single test. This value should be a divisor of the `<experiment_duration>`
       - Replace `<output_folder>` with the folder you want your output to be stored

   - **Attacker**
       - Navigate to the directory containing the script `automated_test_attacker.py`.
       - Run the script with the following command:
       ```
       sudo python3 automated_test_attacker.py [-if <interface>] [-de <experiment_duration>] [-dt <test_duration>] [-i <DU_ip_address>] [-o <output_folder>]
       ```
       All arguments are mandatory.
       - Replace `<interface>` with the interface you want to use to sniff the traffic
       - Replace `<experiment_duration>` with the desired duration of the whole experiment. This value should be a multiple of the `<test_duration>`
       - Replace `<test_duration>` with the desired duration of each single test. This value should be a divisor of the `<experiment_duration>`
       - Replace `<DU_ip_address>` with the ip address of the DU to connect with for the synchronization
       - Replace `<output_folder>` with the folder you want your output to be stored
4. **Output**
   For the DU, in the selected folder the output will be a set of csv files, each representing a test and named `test_DU_{number_test}.csv`, with all PTP packets captured within the frametime of the test in the following format:
   ```
   ['Time', 'Source', 'Destination', 'Protocol', 'Length', 'SequenceID', 'MessageType']
   ```
   For the Attacker, in the selected folder the output will be a set of csv files, each representing a test and named `test_attacker_{number_test}.csv`, with all the attacks performed within the specific timeframe of the test in the following format:
   ```
   ['attack_type', 'start_timestamp', 'end_timestamp']
   ```
   Files with the same indexes in the DU and the Attacker cover the same test in the same timeframe. For example `test_attacker_1.csv` and `test_DU_1.csv` represent the same synchronized test in the same moment. All PTP packets stored in the `test_DU_1.csv` contain the benign traffic along with the malicious packets crafted by the attacker during the attacks logged in `test_attacker_1.csv`
6. **Customization**
- Adjust the parameters according to your environment, the duration of your experiment and tests and the folders you want to save the logs
#### Example
   ```
   sudo python3 automated_data_collection.py -if enp4s0 -de 1800 -dt 300 -o ../DULogs/
   ```
At the DU side, this command starts the benign data collection, sniffing at the interface `enp4s0`. the duration of the experiment is 1800 seconds (30 minutes) and the duration of each test is 300 seconds (5 minutes). Output files will be stored in the `DULogs` in the same folder of `AutomatedScripts` folder where the scripts are
 ```
sudo python3 automated_test_attacker.py -i 192.168.40.1 -if enp4s0 -de 1800 -dt 300 -o ../AttackerLogs/
 ```
At the Attacker side, this command starts synchronization with the DU at its ip address `192.168.40.1` and starts the attacks, sniffing and sending malicious packets at the interface `enp4s0`. the duration of the experiment is 1800 seconds (30 minutes) and the duration of each test is 300 seconds (5 minutes). Output files will be stored in the `AttackerLogs` in the same folder of `AutomatedScripts` folder where the scripts are.


### Dataset Generation
This section involves the last step for generating a dataset useful to traim ML models. It is composed by one script `data_gen.py` and it requires the `csv` files created either in the **Benign Data Collection** or the **Attacks Data Collection** section, which contain the `PTP` traffic captured and processed during the experiments. 
#### Requirements
- Python3
- Pandas
#### Usage
This task is performed at the DU
1. **Installation**
   - Ensure that Python 3.x is installed on your system.
   - Ensure that `pandas` is installed on your system. if not run the following command to install it:
     ```
     pip install pandas
     ```
3. **Running the Script**
   - Navigate to the directory containing the script `data_gen.py`.
     - Run the script with the following command:
     ```
     sudo python3 data_gen.py [-i <input_folder>]
     ```
     The argument is mandatory.
       - Replace `<input_folder>` with the folder containing the `csv` files to process and where the final dataset will be created

5. **Output**
   The script loops over all csv files in the input folder, which are the different tests of the same experiment. Then it creates in the same folder a new file `dataset.csv`. In the output file, all processed information gathered in the several `csv` files will be appended in order to have a unique final dataset the the whole experiment. A single entry in the final dataset consists of all the previous information in the `csv` files mapped to integer, with an additional column `label` which can be either 0 or 1 in case the packet is malicious (1) or benign (0) as follow:
   ```
   Time,Source,Destination,Protocol,Length,SequenceID,MessageType
   1720555124.1808698,e8:eb:d3:b1:37:e7,01:1b:19:00:00:00,PTPv2,58,43466,0
   1720555124.180984,e8:eb:d3:b1:37:e7,01:1b:19:00:00:00,PTPv2,58,43466,8
   ```
   is the initial entry
   ```
   Source,Destination,Length,SequenceID,MessageType,Time Interval,Label
   0,2,58,43466,0,0.0,0
   0,2,58,43466,8,0.00011420249938964844,0
   ```
   entry processed in the final dataset
6. **Customization**
- Adjust the parameters according to the folder containing the files you want to process
#### Example
 ```
   sudo python3 data_gen.py -i ../DULogs/
```
This command starts the generation of the dataset, it analyzes all csv files in the folder and generates the `dataset.csv` file

### Testing the Pipeline
In order to have the best environment to test the Pipeline, we suggest to also run the Open Fronthaul Background Traffic between the DU and the RU. Other than the Background Traffic, the DU also runs the Pipeline for the detection and the Attacker performs attacks randomly. The file that launches the pipeline is `automated_test_DU.py` which initiates the synchronization with the attacker and launches the pipeline. The `pipeline.py` script is in `PipelineScript` folder. this script also imports the features of the `train_test.py` script. The `Model` floder contains the Machine Learning models the pipeline can use for the detection. The attacker works in the same way as the **Attacks Data Collection** section
#### Requirements
- Python3
- Scapy
- Torch
- Numpy
- Pandas
- Sklearn
#### Usage
1. **Installation**
   - Ensure that Python 3.x is installed on your system.
   - Ensure that `scapy` is installed on your system. if not run the following command to install it:
     ```
     pip install scapy
     ```
   - Ensure that `torch` is installed on your system. if not run the following command to install it:
     ```
     pip install torch
     ```
   - Ensure that `pandas` is installed on your system. if not run the following command to install it:
     ```
     pip install pandas
     ```
   - Ensure that `sklearn` is installed on your system. if not run the following command to install it:
     ```
     pip install sklearn
     ```
3. **Running the Script**
   - Run the Background Traffic as explained in **Open Fronthaul Background Traffic**
   - **DU**
   - Navigate to the directory containing the script `automated_test_DU.py`.
     - Run the script with the following command:
     ```
     sudo python3 automated_test_DU.py [-t <type>] [-if <interface>] [-m <model>] [-o <output_folder>] [-de <experiment_duration>] [-dt <test_duration>]
     ```
     Not all the arguments are mandatory.
       - (Default is `ml`) Replace <type> with `ml` if you want a ML based Pipeline, `he` if you want an heuristic approach
       - Replace <interface> with the interface you want to sniff the traffic with
       - In the case of a ML based Pipeline, replace <model> with the filename of the model you want to use
       - Replace <output_folder> with the output folder you want to store your logs
       - Replace `<experiment_duration>` with the desired duration of the whole experiment. This value should be a multiple of the `<test_duration>`
       - Replace `<test_duration>` with the desired duration of each single test. This value should be a divisor of the `<experiment_duration>`
     - **Attacker**
       - Navigate to the directory containing the script `automated_test_attacker.py`.
       - Run the script with the following command:
       ```
       sudo python3 automated_test_attacker.py [-if <interface>] [-de <experiment_duration>] [-dt <test_duration>] [-i <DU_ip_address>] [-o <output_folder>]
       ```
       All arguments are mandatory.
       - Replace `<interface>` with the interface you want to use to sniff the traffic
       - Replace `<experiment_duration>` with the desired duration of the whole experiment. This value should be a multiple of the `<test_duration>`
       - Replace `<test_duration>` with the desired duration of each single test. This value should be a divisor of the `<experiment_duration>`
       - Replace `<DU_ip_address>` with the ip address of the DU to connect with for the synchronization
       - Replace `<output_folder>` with the folder you want your output to be stored

5. **Output**
   In the DU we will have, in the output folder, a set of `csv` files containing the output of the prediction from the pipeline. Each entry in the file represents the prediciton of one packet and the timestamp of when the packet is received. Each files represent each single test performed during the experiment. The name of each file is `test_DU_{test_number}.csv` and an entry in this file is as follows:
   ```
   1720555124.1808698, tensor([[1.]], grad_fn=<RoundBackward0>
   ```
   For the Attacker, in the selected folder the output will be a set of csv files, each representing a test and named `test_attacker_{number_test}.csv`, with all the attacks performed within the specific timeframe of the test in the following format:
   ```
   ['attack_type', 'start_timestamp', 'end_timestamp']
   ```
   Files with the same indexes in the DU and the Attacker cover the same test in the same timeframe. For example `test_attacker_1.csv` and `test_DU_1.csv` represent the same synchronized test in the same moment. All PTP packets stored in the `test_DU_1.csv` contain the benign traffic along with the malicious packets crafted by the attacker during the attacks logged in `test_attacker_1.csv` 
7. **Customization**
   - Adjust the parameters according to your environment, the duration of your experiment and tests, the folders you want to save the logs, the type of prediction and which particular ML model you want to use
#### Example
 ```
sudo python3 automated_test_DU.py -t ml -if enp4s0 -m best_model_no_ts_tr.1.40.pth -o ../DULogs/ -de 1800 -dt 300 
 ```
At the DU side, this command starts the pipeline detection with a ML approach, sniffing the traffic through the `enp4s0` interface. The ML model chosen is `best_model_no_ts_tr.1.40.pth`and the output folder to store the logs is `../DULogs/`. The duration of the experiment is 1800 seconds (30 minutes) and the duration of each test is 300 seconds (5 minutes)
 ```
sudo python3 automated_test_attacker.py -i 192.168.40.1 -if enp4s0 -de 1800 -dt 300 -o ../AttackerLogs/
 ```
At the Attacker side, this command starts synchronization with the DU at its ip address `192.168.40.1` and starts the attacks, sniffing and sending malicious packets at the interface `enp4s0`. The duration of the experiment is 1800 seconds (30 minutes) and the duration of each test is 300 seconds (5 minutes). Output files will be stored in the `AttackerLogs` in the same folder of `AutomatedScripts` folder where the scripts are.
