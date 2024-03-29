import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
from concurrent.futures import ThreadPoolExecutor
import torch.optim as optim


np.random.seed(17)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()

        # Define the LSTM layer with batch_first=True
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Forward propagate through LSTM layer
        lstm_out, _ = self.lstm(x)
        # print("lstm_out shape:", lstm_out.shape)
        # Only take the output from the final time step
        # lstm_out = lstm_out[:, -1, :]
        # Forward propagate through the output layer
        out = self.fc(lstm_out)
        # Apply sigmoid activation function
        out = torch.sigmoid(out)
        return out


class PTPDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.long)  # Assuming labels are integers
        return sample, label


def load_data(file, sequence):
    def load_chunk(start_idx, end_idx):
        # Read a chunk of data from the CSV file
        data_chunk = pd.read_csv(file, skiprows=start_idx, nrows=end_idx - start_idx)
        return data_chunk

    # Read column names from the CSV file
    with open(file) as f:
        column_names = f.readline().strip().split(',')

    # Read the data from the CSV file into a DataFrame to calculate its total length
    total_length = sum(1 for _ in open(file)) - 1  # Subtract 1 for the header row

    # Calculate the number of chunks needed
    num_chunks = total_length // sequence

    # Generate start and end indices for each chunk
    chunk_indices = [(i * sequence, (i + 1) * sequence) for i in range(num_chunks)]

    # Read chunks of data in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_chunk, start, end) for start, end in chunk_indices]
        chunks = [future.result() for future in futures]

    # Initialize lists to store training and validation data
    training_data = []
    validation_data = []

    # Randomly select chunks for validation
    validation_chunk_indices = np.random.choice(num_chunks, size=int(num_chunks * 0.2), replace=False)

    # Iterate over each chunk and determine if it belongs to training or validation
    for i, chunk in enumerate(chunks):
        if i in validation_chunk_indices:
            validation_data.extend(chunk.values.tolist())  # Append rows to validation_data
            #print(chunk)
        else:
            training_data.extend(chunk.values.tolist())  # Append rows to training_data

    # Convert the lists of rows to DataFrames
    # print(training_data)
    training_data = pd.DataFrame(training_data, columns=chunk.columns)
    validation_data = pd.DataFrame(validation_data, columns=chunk.columns)

    # Assign column names to the DataFrames
    training_data.columns = column_names
    validation_data.columns = column_names
    # print(training_data)

    # Extract the corresponding labels for training and validation data
    training_labels = training_data['Label']
    validation_labels = validation_data['Label']

    # Remove the labels from the DataFrames
    training_data.drop(columns=['Label'], inplace=True)
    validation_data.drop(columns=['Label'], inplace=True)

    return training_data, validation_data, training_labels, validation_labels


# Define additional metrics (e.g., accuracy)
def accuracy(output, target):
    # Calculate accuracy
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_input", default='final_dataset.csv',
                        help="file containing all the training data")
    parser.add_argument("-s", "--sequence_length", type=int, default=10,
                        help="length of sequence to use")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="batch size for training and validation data loaders")
    args = parser.parse_args()

    input_file = args.file_input
    sequence = args.sequence_length
    batch_size = args.batch_size

    train_data, validate_data, train_label, validate_label = load_data(input_file, sequence)

    train_dataset = PTPDataset(train_data, train_label)
    val_dataset = PTPDataset(validate_data, validate_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    input_size = train_data.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 100

    # Instantiate the model and move it to the GPU
    model = LSTMClassifier(input_size, hidden_size).to(device)

    # Define the loss function (Binary Cross-Entropy Loss)
    criterion = nn.BCELoss()

    # Define optimizer (Adam) and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')  # Initialize best validation loss

    # Training loop with evaluation on the validation set
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Move inputs and labels to the GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Define outputs here
            loss = criterion(outputs.squeeze(), labels.float())  # No need to convert labels to float
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy(outputs, labels)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():  # Disable gradient calculation during validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
                val_accuracy += accuracy(outputs, labels)

        # Adjust learning rate
        scheduler.step()

        # Print average loss and accuracy for each epoch
        print(f'Epoch {epoch + 1}, '
              f'Training Loss: {running_loss / len(train_loader):.4f}, '
              f'Training Accuracy: {running_accuracy / len(train_loader):.4f}, '
              f'Validation Loss: {val_loss / len(val_loader):.4f}, '
              f'Validation Accuracy: {val_accuracy / len(val_loader):.4f}')

        # Save model if validation loss decreases
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')