# S-Plane Security
This needs to be updated to better describe the entire project.

## Dataset Preprocessing Script

### Overview
This Python script preprocesses a dataset comprising multiple CSV files containing network traffic data. It applies data cleaning and labeling techniques, aggregates the processed data into a single CSV file, and provides features for further analysis or machine learning tasks.

### Features
- **Data Cleaning**: Removes rows with invalid or irrelevant data, such as missing values or incorrect protocol types.
- **Labeling**: Assigns labels to the dataset based on the type of network traffic, distinguishing between benign and malicious traffic.
- **Categorical Encoding**: Converts categorical variables into numerical representations for modeling purposes.
- **Time Interval Calculation**: Computes the time interval between successive data points for time-series analysis.
- **Command-Line Interface**: Supports command-line arguments for specifying the directory containing the dataset files and the output file name.

### Usage
1. **Installation**:
   - Ensure that Python 3.x is installed on your system.
   - Install the required Python libraries by running `pip install pandas numpy argparse`.

2. **Running the Script**:
   - Navigate to the directory containing the script (`dataset_preprocessing.py`).
   - Run the script with the following command:
     ```
     python dataset_preprocessing.py [-d <dataset_directory>] [-o <output_filename>] [-a <attacker_mac_address>]
     ```
     All arguments are optional.
     - Replace `<dataset_directory>` with the path to the directory containing the dataset files.
     - Replace `<output_filename>` with the desired name for the output CSV file.
     - Replace `<attacker_mac_address>` with the MAC address of the attacker for labeling malicious traffic.

3. **Output**:
   - The script will process all CSV files in the specified directory and subdirectories.
   - It will generate a single CSV file containing the cleaned and labeled dataset.
   - The output file will be saved in the current directory with the name `final_dataset.csv`.

4. **Customization**:
- Adjust any other parameters or preprocessing steps in the script according to your dataset requirements.

### Example
```
python dataset_preprocessing.py -d ./DataCollectionPTP/DU -o final_dataset.csv
```
This command preprocesses all CSV files in the `DataCollectionPTP/DU` directory and generates a file named `final_dataset.csv` containing the cleaned and labeled dataset.
