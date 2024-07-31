import pandas as pd
import argparse
import os
import csv

# Define the initial mapping for addresses and message types

ptp_message_types = {
    0: "Sync",
    1: "Delay_Req",
    2: "Pdelay_Req",
    3: "Pdelay_Resp",
    8: "Follow_Up",
    9: "Delay_Resp",
    10: "Pdelay_Resp_Follow_Up",
    11: "Announce",
    12: "Signaling",
    13: "Management"
}
reverse_ptp_message_types = {v: k for k, v in ptp_message_types.items()}

# Create log file
def create_log_file(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Source', 'Destination', 'Length', 'SequenceID', 'MessageType', 'Time Interval', 'Label'])
        print(f'Created {filename}')

def is_under_attack(row, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up):
    if row['Source'] == address_mapping[args.attacker]:  # When the mac address is the attacker's 
        return 1, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up
    
    if row['MessageType'] == 0:  # Sync message
        if row['SequenceID'] in seen_sync and row['SequenceID']< last_seq_sync: # if the packet's seq ID has already occurred and it is lower than the previous one it is a replayed message
            return 1, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up
        seen_sync.add(row['SequenceID']) # Add value to the queue
        last_seq_sync = row['SequenceID'] # Store last seen value
        return 0, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up
    
    elif row['MessageType'] == 8:  # Follow_Up message
        if row['SequenceID'] in seen_follow_up and row['SequenceID'] < last_seq_follow_up: # if the packet's seq ID has already occurred and it is lower than the previous one it is a replayed message
            return 1, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up
        seen_follow_up.add(row['SequenceID']) # Add value to queue
        last_seq_follow_up = row['SequenceID'] # Store last seen value
        return 0, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up
    
    return 0, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up

def process_data(input_file, output_file, seen_sync, seen_follow_up):
    # Read the input CSV files
    df_new = pd.read_csv(input_file)

    # Function to map addresses and add new indexes if new addresses are found
    def map_address(address):
        if address not in address_mapping:
            address_mapping[address] = len(address_mapping)
        return address_mapping[address]

    # Map the Source and Destination columns
    df_new['Source'] = df_new['Source'].map(map_address)
    df_new['Destination'] = df_new['Destination'].map(map_address)


    last_seq_sync = None
    last_seq_follow_up = None
    # Use the fixed mapping for MessageType
    df_new['MessageType'] = df_new['MessageType'].map(reverse_ptp_message_types)

    # Apply the is_under_attack function to each row in the DataFrame
    df_new['Label'] = 0
    for index, row in df_new.iterrows():
        df_new.at[index, 'Label'], seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up = is_under_attack(row, seen_sync, seen_follow_up, last_seq_sync, last_seq_follow_up)

    # Calculate the Time Interval
    df_new['Time Interval'] = df_new['Time'].diff().fillna(0)

    # Drop the unnecessary columns
    df_new.drop(columns=['Time', 'Protocol'], inplace=True)
    # Ensure the column order
    column_order = ['Source', 'Destination', 'Length', 'SequenceID', 'MessageType', 'Time Interval', 'Label']
    df_new = df_new[column_order]

    # Save the new DataFrame to a new CSV file
    df_new.to_csv(output_file, mode='a', header=False, index=False)
    print(f"Processed dataset saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="enter the folder where to find csv files", type=str, required=True)
    parser.add_argument("-a", "--attacker", help="enter the MAC address of the attacker", type=str, required=True)
    args = parser.parse_args()
    
    address_mapping = {args.attacker:0}
    
    # Define output 
    filename = args.input_folder+'dataset.csv'
    
    create_log_file(filename)
    files = [f for f in os.listdir(args.input_folder) if f.endswith('.csv') and not f.endswith('dataset.csv')] # checks csv files and discards the new created final dataset
    for i, file in enumerate(files):
            print(f'file number {i}')
            # Load the CSV files
            du = os.path.join(args.input_folder, file)
            print(du)  # This prints the path to the CSV file
            # Initialize tracking sets and variables for sync and follow-up messages
            seen_sync = set()
            seen_follow_up = set()
            process_data(du, filename, seen_sync, seen_follow_up)
