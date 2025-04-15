# TIMESAFE  
**Timing Interruption Monitoring and Security Assessment for Fronthaul Environments**

TIMESAFE is a research-driven framework designed to expose and mitigate vulnerabilities in 5G fronthaul synchronization mechanisms. As modern 5G and beyond systems adopt disaggregated Radio Access Network (RAN) architectures, ensuring secure and reliable timing over Ethernet-based Time-Sensitive Networking (TSN) has become critical. Standards like Precision Time Protocol (PTP) focus heavily on performance but often overlook security, leaving fronthaul links open to serious threats.

In our [accompanying paper](https://arxiv.org/abs/2412.13049), we demonstrate how synchronization-based spoofing and replay attacks can catastrophically disrupt a production-ready O-RAN and 5G-compliant base station in under 2 seconds—forcing manual recovery. To counter such threats, TIMESAFE introduces a Machine Learning (ML)-based monitoring tool capable of detecting various malicious timing disruptions with over **97.5% accuracy**.

> ⚠️ If you use this repository, please consider citing our paper:
>
> ```bibtex
> @misc{groen2024timesafetiminginterruptionmonitoring,
>       title={TIMESAFE: Timing Interruption Monitoring and Security Assessment for Fronthaul Environments}, 
>       author={Joshua Groen and Simone Di Valerio and Imtiaz Karim and Davide Villa and Yiewi Zhang and Leonardo Bonati and Michele Polese and Salvatore D'Oro and Tommaso Melodia and Elisa Bertino and Francesca Cuomo and Kaushik Chowdhury},
>       year={2024},
>       eprint={2412.13049},
>       archivePrefix={arXiv},
>       primaryClass={cs.NI},
>       url={https://arxiv.org/abs/2412.13049}, 
> }
> ```

---

## TIMESAFE Monitor GUI

### Overview
This Streamlit GUI allows you to visually monitor a remote DU. It uses the `pipeline_demo2.py` script deployed on the remote RU.

### Usage
1. **Requirements**:
   - `streamlit`, `paramiko`, `seaborn`, `matplotlib.pyplot`

2. **Update SSH Configs**:
   - Ensure you update the SSH configs for the remote RU in both the `start_monitor` and `stop_monitor` functions

3. **Running the Script**:
   - Run the script with the following command:
     ```
     streamlit run monitor_app.py
     ```

---

## TIMESAFE Attack GUI

### Overview
This Streamlit GUI allows you to choose different attack parameters and execute them from a remote device.

### Usage
1. **Requirements**:
   - `streamlit`, `paramiko`, `itertools`

2. **Update SSH Configs**:
   - Ensure you update the SSH configs for the remote RU in both the `start_attack` and `stop_attack` functions

3. **Running the Script**:
   - Run the script with the following command:
     ```
     streamlit run attack_app.py
     ```

---

## Other Tools and Scripts

- Data collection and pre-processing tools are located in the `DataCollectionPTP` directory.

- All trained models intended for deployment at the DU are in the `DU_model` directory.

- The `Testbed` directory contains testbed automation scripts, including the actual attack scripts.

- The `Production_Environment` folder includes visualization tools and experimental results from our production setup.
