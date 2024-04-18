import csv

# Function to modify a single row of the CSV file
def modify_row(row):
    # Modify the row as per your requirements

    if row[4]== '0':
        row[4]='11'
    if row[4]== '4':
        row[4]='0'
    if row[4]== '3':
        row[4]='8'
    if row[4]== '1':
        row[4]= '1'
    if row[4]== '2':
        row[4]='9'

    return row

# Path to the existing CSV file
file_path = 'final_dataset.csv'

# Temporary file path for writing modified data
temp_file_path = 'temp_file.csv'

# Open existing CSV file for reading and temporary file for writing
with open(file_path, 'r', newline='') as input_file, \
     open(temp_file_path, 'w', newline='') as temp_file:

    # Create CSV reader and writer objects
    csv_reader = csv.reader(input_file)
    csv_writer = csv.writer(temp_file)

    # Read and process each row of the existing CSV file
    for row in csv_reader:
        # Modify the row
        modified_row = modify_row(row)
        
        # Write the modified row to the temporary file
        csv_writer.writerow(modified_row)

# Replace the original file with the temporary file
import shutil
shutil.move(temp_file_path, file_path)

print("CSV file modified successfully!")
