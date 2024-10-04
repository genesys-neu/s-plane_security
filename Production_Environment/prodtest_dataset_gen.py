import os
import numpy as np
import pandas as pd
import argparse
from concurrent.futures import ThreadPoolExecutor


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


def label_data(df):
    # Initialize 'Label' column with 0
    df['Label'] = 0

    # Set 'Label' to 1 where both conditions are true
    # df.loc[(df['Source'] == 'b8:ce:f6:5e:6b:4a') &
    #        (df['Time'] >= 78.506815) &
    #        (df['Time'] <= 102.512281), 'Label'] = 1

    return df


def load_data(input_file):

    df = pd.read_csv(input_file)
    # print(df)
    # Only keep rows with PTPv2 in Protocol column
    df = df[df['Protocol'].str.contains('PTP', case=False)]
    # Drop the Protocol column
    df.drop(columns=['Protocol'], inplace=True)

    # Add labels
    label_data(df)
    # From float to Integers
    df['Label'] = df['Label'].astype(int)

    # Apply mapping function to both 'Source' and 'Destination' columns simultaneously
    df, mapping = map_categories(df, ['Source', 'Destination'])

    # Calculate the time interval between rows
    df['Time Interval'] = df['Time'].diff().fillna(0)
    # And drop the Time column
    df.drop(columns=['Time'], inplace=True)

    # column_types = df.dtypes
    # print(df.iloc[55:75])
    # print(column_types)

    return df


if __name__ == "__main__":


    output_file = 'final_dataset.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default='final_dataset.csv',
                        help="output file name for the final dataset")
    parser.add_argument("-i", "--input", default='../DataCollectionPTP/successful_announce_attack_ptp.csv',
                        help="input file name")
    args = parser.parse_args()

    input_file = args.input

    output_file = args.output

    # Define a list to store tuples of futures and file paths
    future_file_pairs = []

    df = load_data(input_file)
    final_df = df[['Source', 'Destination', 'Length', 'SequenceID', 'MessageType', 'Time Interval', 'Label']]

    # print(final_df)
    # Save the final DataFrame to a CSV file
    final_df.to_csv(output_file, index=False)
    print(f"Final dataset saved to {output_file}")
