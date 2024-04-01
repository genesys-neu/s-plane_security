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
     python3 dataset_preprocessing.py [-d <dataset_directory>] [-o <output_filename>] [-a <attacker_mac_address>]
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
python3 dataset_preprocessing.py -d ./DataCollectionPTP/DU -o final_dataset.csv
```
This command preprocesses all CSV files in the `DataCollectionPTP/DU` directory and generates a file named `final_dataset.csv` containing the cleaned and labeled dataset.


## Classifier for Time-Series Data (train_test.py)

This script implements an LSTM-based classifier for time-series data. It includes functionalities for training the model, evaluating its performance, and generating confusion matrices. The main features of the script include:

- **Data Loading and Preprocessing:** The script can load time-series data from a CSV file, split it into training, validation, and testing sets, and prepare it for training.
  
- **Model Training:** It trains an LSTM classifier using the training data. The script supports customization of various model parameters such as input dimension, hidden dimension, and number of epochs.
  
- **Model Evaluation:** After training, the script evaluates the trained model's performance on the validation and testing sets. It calculates metrics such as loss and accuracy and generates confusion matrices.
  
- **Confusion Matrix Visualization:** The script uses seaborn and matplotlib to visualize confusion matrices, providing insights into the model's performance in classifying different classes.

### Requirements

- Python 3
- PyTorch
- pandas
- NumPy
- scikit-learn
- seaborn
- matplotlib

### Usage

To use the script, follow these steps:

1. Install the required dependencies listed in above.
2. Place your time-series data in a CSV file named `final_dataset.csv`.
3. Run the script using the command `python3 train_test.py [-f <file_input>] [-t <trial_version>]`.

### Customization

You can customize the script by adjusting various parameters such as input size, hidden size, number of layers, and optimization parameters directly in the script or through command-line arguments.

### Example
```
python3 train_test.py -t 1.1
```
This command preprocesses all CSV files in the `DataCollectionPTP/DU` directory and generates a file named `final_dataset.csv` containing the cleaned and labeled dataset.
