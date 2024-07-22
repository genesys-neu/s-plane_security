# Testbed Environment

## Testbed Setup
Quick overview of the setup and the structure of each element in the Testbed

### Overview
The Testbed Environment consists of 3 elements: the DU, the RU and the attacker and they are directly connected through a switch. 
- **RU:** Runs the PTP Protocol as Slave Clock and runs the Open Fronthaul Background Traffic 
- **DU:** Runs the PTP Protocol as Master Clock, runs the Open Fronthaul Background Traffic, and can either run the pipeline and the data collection. As for the data collection the DU listens to the proper interface and stores the PTP traffic. It processes the information in a *.csv* file for ease of readability and save memory resources and perform the offline detection of the attack. **TO DETERMINE WHETHER TO USE THE SNIFF MODULE FOR LIVE DETECTION OR TCPDUMP**
- **Attacker:** Performs the attack on the PTP Protocol
  
### DU

### RU

### Attacker

## Features

## Requirements

## Running the Scripts

### Open Fronthaul Background Traffic

### Benign Data Collection

### Attacks Data Collection

### Dataset Generation

### Testing the Pipeline

