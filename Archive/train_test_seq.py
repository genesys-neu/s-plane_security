import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import argparse
from concurrent.futures import ThreadPoolExecutor
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json


np.random.seed(17)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim  # Define hidden_dim as an attribute
        # Define the LSTM layer with batch_first=True
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Define the output layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Print the shape of the input tensor
        # print("Input shape:", x.shape)

        # Reshape input tensor to include batch size dimension
        x = x.unsqueeze(0)  # Add dummy dimension for batch size
        # print("Reshaped input shape:", x.shape)

        # Initialize hidden state with zeros
        h0 = torch.zeros(1, 1, self.hidden_dim).to(x.device)
        c0 = torch.zeros(1, 1, self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        # print("LSTM output shape:", lstm_out.shape)

        # Only take the output from the final time step
        lstm_out = lstm_out[:, -1, :]

        # Forward propagate through the output layer
        out = self.fc(lstm_out)

        # Apply sigmoid activation function
        out = torch.sigmoid(out)
        # print("Output shape:", out.shape)
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


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_fp=1, weight_fn=2):
        super(WeightedBCELoss, self).__init__()
        self.weight_fp = weight_fp
        self.weight_fn = weight_fn

    def forward(self, inputs, targets):
        # Compute binary cross-entropy loss
        bce_loss = nn.BCELoss()(inputs, targets)

        # Compute custom weighted loss
        loss = (targets * self.weight_fn * bce_loss) + ((1 - targets) * self.weight_fp * bce_loss)

        return loss.mean()


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

    # Randomly select chunks for validation
    valid_test_indices = np.random.choice(num_chunks, size=int(num_chunks * 0.2), replace=False)

    # Separate validation and testing indices
    valid_indices = valid_test_indices[:len(valid_test_indices) // 2]
    test_indices = valid_test_indices[len(valid_test_indices) // 2:]

    # Initialize lists to store training, validation, and testing data
    training_data = []
    validation_data = []
    testing_data = []

    # Iterate over each chunk and determine if it belongs to training, validation, or testing
    for i, chunk in enumerate(chunks):
        if i in valid_indices:
            validation_data.extend(chunk.values.tolist())  # Append rows to validation_data
        elif i in test_indices:
            testing_data.extend(chunk.values.tolist())  # Append rows to testing_data
        else:
            training_data.extend(chunk.values.tolist())  # Append rows to training_data

    # Convert the lists of rows to DataFrames
    training_data = pd.DataFrame(training_data, columns=column_names)
    validation_data = pd.DataFrame(validation_data, columns=column_names)
    testing_data = pd.DataFrame(testing_data, columns=column_names)

    # Extract the corresponding labels for training, validation, and testing data
    training_labels = training_data['Label']
    validation_labels = validation_data['Label']
    testing_labels = testing_data['Label']

    # Remove the labels from the DataFrames
    training_data.drop(columns=['Label'], inplace=True)
    validation_data.drop(columns=['Label'], inplace=True)
    testing_data.drop(columns=['Label'], inplace=True)

    return training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels


# Define additional metrics (e.g., accuracy)
def accuracy(output, target):
    # Calculate accuracy
    correct = (output == target).sum().item()
    total = target.size(0)
    return correct / total


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_input", default='final_dataset.csv',
                        help="file containing all the training data")
    parser.add_argument("-t", "--trial_version", default='',
                        help="add identifer for this trial")

    args = parser.parse_args()
    input_file = args.file_input
    t_v = args.trial_version

    chunk_size = 100
    training_metrics = {'epochs': [], 'training_loss': [], 'training_accuracy': [], 'validation_loss': [],
                        'validation_accuracy': [], 'confusion_matrix': []}

    train_data, validate_data, test_data, train_label, validate_label, test_label = load_data(input_file, chunk_size)

    train_dataset = PTPDataset(train_data, train_label)
    val_dataset = PTPDataset(validate_data, validate_label)
    test_dataset = PTPDataset(test_data, test_label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    input_size = train_data.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 200

    # Instantiate the model and move it to the GPU
    model = LSTMClassifier(input_size, hidden_size).to(device)

    # Define the loss function (Binary Cross-Entropy Loss)
    # criterion = nn.BCELoss()
    # Define the loss function (Custom Weighted Binary Cross-Entropy Loss)
    criterion = WeightedBCELoss(weight_fp=1, weight_fn=2)

    # Define optimizer (Adam) and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=2000, mode='triangular')

    best_val_loss = float('inf')  # Initialize the best validation loss
    patience = 20  # Number of epochs to wait for improvement
    counter = 0  # Counter for patience

    # Training loop with evaluation on the validation set
    for epoch in range(num_epochs):
        # Randomly select batch size between 10 and 40
        batch_size = np.random.randint(10, 31)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training phase
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # Modify labels if any label in the batch is 1
            new_label = torch.tensor(1 if 1 in labels else 0, dtype=torch.float).to(device)

            try:
                # Move inputs to the GPU
                inputs = inputs.to(device)

                # Forward pass
                outputs = model(inputs)
                # print(f'Outputs: {outputs}, Label: {new_label}')
                # Round the predictions to 0 or 1
                predicted = torch.round(outputs)

                # Repeat the scalar label to match the shape of the model output tensor
                new_label = new_label.expand_as(outputs)
                # print(f'Outputs: {outputs.shape}, New Label: {new_label.shape}')

                # Use raw probabilities in the loss calculation
                loss = criterion(outputs, new_label.float())

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_accuracy += accuracy(predicted, new_label)

            except ValueError as e:
                print(f'Error occurred in training epoch {epoch + 1}: {e}')
                continue

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_accuracy = 0.0
        # iterate over the validation data loader
        for inputs, labels in val_loader:
            # Move inputs to the GPU
            inputs = inputs.to(device)

            # Modify labels if any label in the batch is 1
            new_label = torch.tensor(1 if 1 in labels else 0, dtype=torch.float).to(device)

            with torch.no_grad():  # Disable gradient calculation during validation
                try:
                    # Forward pass
                    outputs = model(inputs)
                    # print(f'Validation Outputs: {outputs}, Label: {new_label}')
                    # Round the predictions to 0 or 1
                    predicted = torch.round(outputs)

                    # Repeat the scalar label to match the shape of the model output tensor
                    new_label = new_label.expand_as(outputs)
                    # print(f'Validation Outputs: {outputs.shape}, New Label: {new_label.shape}')

                    # Use raw probabilities in the loss calculation
                    loss = criterion(outputs, new_label.float())

                    val_loss += loss.item()
                    val_accuracy += accuracy(predicted, new_label)

                except ValueError as e:
                    print(f'Error occurred in validation epoch {epoch + 1}: {e}')
                    continue

        # Adjust learning rate
        scheduler.step()

        # Print average loss and accuracy for each epoch
        print(f'Epoch {epoch + 1}, Batch Size: {batch_size}, '
              f'Training Loss: {running_loss / len(train_loader):.4f}, '
              f'Training Accuracy: {running_accuracy / len(train_loader):.4f}, '
              f'Validation Loss: {val_loss / len(val_loader):.4f}, '
              f'Validation Accuracy: {val_accuracy / len(val_loader):.4f}')
        training_metrics['epochs'].append(epoch + 1)
        training_metrics['training_loss'].append(running_loss / len(train_loader))
        training_metrics['training_accuracy'].append(running_accuracy / len(train_loader))
        training_metrics['validation_loss'].append(val_loss / len(val_loader))
        training_metrics['validation_accuracy'].append(val_accuracy / len(val_loader))

        # Save model if validation loss decreases
        if val_loss < best_val_loss and epoch >= 5:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_seq_{t_v}.pth')
            counter = 0  # Reset counter if there's improvement
        else:
            # Increment counter if there's no improvement
            counter += 1

        # Check early stopping condition
        if counter >= patience:
            print(f'Validation loss has not improved for {patience} epochs. Stopping training.')
            break

    # Test phase
    model.eval()  # Set model to evaluation mode
    batch_size = 25
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_predictions = []
    test_targets = []

    with torch.no_grad():  # Disable gradient calculation during testing
        for inputs, labels in test_loader:
            # Modify labels if any label in the batch is 1
            new_label = torch.tensor(1 if 1 in labels else 0, dtype=torch.float).to(device)
            print(f'inputs: {inputs}, new label: {new_label}')
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted = torch.round(outputs)  # Round the predictions to 0 or 1
            print(f'predicted: {predicted}')
            test_predictions.append(predicted.cpu().numpy())
            print(f'test_predictions: {test_predictions}')
            test_targets.append(new_label.cpu().numpy())

    # Generate confusion matrix
    conf_matrix = confusion_matrix(test_targets, test_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    # Convert confusion matrix from NumPy array to list
    conf_matrix_list = conf_matrix.tolist()
    training_metrics['confusion_matrix'].append(conf_matrix_list)

    # Save the dictionary to a JSON file
    with open(f'training_log_seq_{t_v}.json', 'w') as jsonfile:
        json.dump(training_metrics, jsonfile)

    # Plot confusion matrix
    '''
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    '''
