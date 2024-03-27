import os
import numpy as np
import pandas as pd
import argparse


# List to store DataFrames from all .csv files
dfs = []
attacker = '68:05:ca:2e:59:73'


def map_categories(df, columns):
    # Initialize a mapping dictionary
    mapping = {}
    code = 0
    # Iterate over the specified columns
    for column in columns:
        # Iterate over unique values in the column
        for value in df[column].unique():
            # Map each unique value to a code if not already mapped
            if value not in mapping:
                mapping[value] = code
                # Increment the code for the next value
                code += 1
    # Apply the mapping to all specified columns
    for column in columns:
        df[column] = df[column].map(mapping)
    return df, mapping


def label_data(df, t_type, mapping):
    if t_type == 'benign':
        df['Label'] = 0
    elif t_type == 'Announce':
        df['Label'] = df['Source'].apply(lambda x: 1 if x == mapping[attacker] else 0)
    return df


def load_data(input_file, attack_type):
    # input_file = "./DataCollectionPTP/DU/BenignTraffic/run1-12sep-aerial-udpDL.csv"
    df = pd.read_csv(input_file)

    # Remove rows with '-' in Protocol column
    df = df[df['Protocol'] == 'PTPv2']
    # Drop the Protocol column
    df.drop(columns=['Protocol'], inplace=True)
    # Apply mapping function to both 'Source' and 'Destination' columns simultaneously
    df, mapping = map_categories(df, ['Source', 'Destination'])

    # Get the list of remaining categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    # Convert categorical columns to numerical data
    for column in categorical_columns:
        df[column] = pd.Categorical(df[column]).codes

    # Calculate the time interval between rows
    df['Time Interval'] = df['Time'].diff().fillna(0)
    # And drop the Time column
    df.drop(columns=['Time'], inplace=True)

    # column_types = df.dtypes
    # print(df.iloc[55:75])
    # print(column_types)
    label_data(df, attack_type, mapping)
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

                # Extract the section after the last directory
                section = os.path.basename(os.path.dirname(root))
                print(root)
                print(section)
                if section == 'Announce':
                    # Read the .csv file into a DataFrame and label it
                    init_df = load_data(file_path, section)
                    print(init_df)

    # Concatenate the list of DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    print(final_df)
