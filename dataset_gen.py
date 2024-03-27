import os
import numpy as np
import pandas as pd
import argparse


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
    if t_type == 'Benign':
        df['Label'] = 0
    elif t_type == 'Announce':
        df['Label'] = df['Source'].apply(lambda x: 1 if x == mapping[attacker] else 0)
    # TODO: Add Sync_FollowUp attack logic here
    return df


def load_data(input_file, attack_type):
    # input_file = "./DataCollectionPTP/DU/BenignTraffic/run1-12sep-aerial-udpDL.csv"
    df = pd.read_csv(input_file)

    # Only keep rows with PTPv2 in Protocol column
    df = df[df['Protocol'].str.contains('PTP', case=False)]
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

    # List to store DataFrames from all .csv files
    dfs = []
    attacker = '68:05:ca:2e:59:73'
    output_file = 'final_dataset.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default='./DataCollectionPTP/DU',
                        help="directory to use when building dataset")
    parser.add_argument("-o", "--output", default='final_dataset.csv',
                        help="output file name for the final dataset")
    parser.add_argument("-a", "--attacker", default='68:05:ca:2e:59:73',
                        help="MAC address of the attacker")
    args = parser.parse_args()

    directory = args.directory
    output_file = args.output
    attacker = args.attacker

    # Recursively iterate over all files and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a .csv file
            if file.endswith(".csv"):
                # Construct the full path to the .csv file
                file_path = os.path.join(root, file)
                # print(root)
                if 'BenignTraffic' in root:
                    init_df = load_data(file_path, 'Benign')
                    # print(init_df)
                    dfs.append(init_df)
                else:
                    # Extract the section after the last directory
                    section = os.path.basename(os.path.dirname(root))
                    # print(root)
                    # print(section)
                    # Read the .csv file into a DataFrame and label it
                    init_df = load_data(file_path, section)
                    # print(init_df)
                    dfs.append(init_df)

    # Concatenate the list of DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    # print(final_df)
    # Save the final DataFrame to a CSV file
    final_df.to_csv(output_file, index=False)
    print(f"Final dataset saved to {output_file}")
