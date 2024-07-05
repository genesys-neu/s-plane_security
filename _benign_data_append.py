import pandas as pd
import argparse

# Define the mapping for addresses
address_mapping = {
    'e8:eb:d3:b1:37:e7': 0,
    '01:1b:19:00:00:00': 2,
    'e8:eb:d3:b1:47:02': 1
}

def process_data(input_file, output_file):
    # Read the input CSV file
    df_new = pd.read_csv(input_file)

    # Map the Source and Destination columns
    df_new['Source'] = df_new['Source'].map(address_mapping)
    df_new['Destination'] = df_new['Destination'].map(address_mapping)

    # Calculate the Time Interval
    df_new['Time Interval'] = df_new['Time'].diff().fillna(0)
    # Drop the unnecessary columns
    df_new.drop(columns=['Time', 'Protocol'], inplace=True)

    # Add the Label column with default value 0
    df_new['Label'] = 0

    # Ensure the column order
    column_order = ['Source', 'Destination', 'Length', 'SequenceID', 'MessageType', 'Time Interval', 'Label']
    df_new = df_new[column_order]

    # Save the new DataFrame to a new CSV file
    df_new.to_csv(output_file,mode='a',header=False, index=False)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":


    process_data('./BenignTraces.csv', 'DU_model/final_dataset.csv')
