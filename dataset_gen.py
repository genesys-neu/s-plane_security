import os
import numpy as np
import pandas as pd
import argparse


# List to store DataFrames from all .csv files
dfs = []


def load_data(input_file):
    #input_file = "./DataCollectionPTP/DU/BenignTraffic/run1-12sep-aerial-udpDL.csv"

    df = pd.read_csv(input_file)

    # Remove rows with '-' in Protocol column
    df = df[df['Protocol'] == 'PTPv2']

    # Drop the Protocol column
    df.drop(columns=['Protocol'], inplace=True)
    # print(df.iloc[50:75])

    """
    # Convert the Sequence ID column to numeric, coercing errors to NaN
    df['SequenceID'] = pd.to_numeric(df['SequenceID'], errors='coerce')
    # Replace NaN with -1 because these are not PTP packets
    df['SequenceID'] = df['SequenceID'].fillna(-1).astype(int)
    """

    # Convert categorical to numeric
    # Create a dictionary to map unique labels from both columns to the same integer encoding
    label_map = {}
    unique_labels = sorted(set(df['Source']) | set(df['Destination']))
    for i, label in enumerate(unique_labels):
        label_map[label] = i

    # Encode both 'Source' and 'Destination' columns using the label map
    df['Source'] = df['Source'].map(label_map)
    df['Destination'] = df['Destination'].map(label_map)

    # Get the list of remaining categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    # Convert categorical columns to numerical data
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes

    # Calculate the time interval between rows
    df['Time Interval'] = df['Time'].diff().fillna(0)
    # And drop the Time column
    df.drop(columns=['Time'], inplace=True)

    column_types = df.dtypes

    # print(df.iloc[55:75])
    # print(column_types)
    return df


def label_data(df, malicious):
    if malicious:
        df['Label'] = 1
    else:
        df['Label'] = 0
    return df


if __name__ == "__main__":

    # Directory containing .csv files
    directory = "./DataCollectionPTP/DU"
    parser = argparse.ArgumentParser()

    # Recursively iterate over all files and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a .csv file
            if file.endswith(".csv"):
                # Construct the full path to the .csv file
                file_path = os.path.join(root, file)
                # Check if the path contains 'BenignTraffic'
                if 'BenignTraffic' in root:
                    # print(root)
                    # Read the .csv file into a DataFrame and append it to the benign_dfs list
                    init_df = load_data(file_path)
                    # Add the labels for benign traffic (0)
                    init_df = label_data(init_df, False)
                    # print(init_df)
                    dfs.append(init_df)

    # Concatenate the list of DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    print(final_df)
