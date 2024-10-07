import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np
import re

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from Archive.train_test import TransformerNN, LSTMClassifier, CNNModel2D  # Adjust the imports based on your module


class PTPDataset(Dataset):
    def __init__(self, data, labels, sequence_length, stride=2):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.sequence_length) // self.stride + 1

    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        sample = self.data[start_idx:end_idx]
        label_tensor = self.labels[start_idx:end_idx]
        label = torch.tensor(1 if torch.any(label_tensor) else 0, dtype=torch.long)
        return sample, label


def load_data(input_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Separate features and labels
    features = df.iloc[:, :-1]
    labels = df['Label']

    return features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default='./',
                        help="directory containing all the models to be tested")
    parser.add_argument("-f", "--file_input", default='final_dataset.csv',
                        help="file containing all the training data")
    parser.add_argument("-m", "--model", default='Transformer', help="Choose Transformer, CNN, or LSTM")

    args = parser.parse_args()
    model_dir = args.directory
    input_file = args.file_input
    model_type = args.model

    # Load the data
    features, labels = load_data(input_file)

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels.values, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    input_size = features.shape[1]
    hidden_size = 64  # for LSTM
    num_layers = 2
    output_size = 1

    # List to store models and their corresponding confusion matrices
    model_list = []

    for root, _, files in os.walk(model_dir):
        for model_file in files:
            # Check if the file is a .pth file
            if model_file.endswith(".pth"):
                # Extract model name from the file name
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(root, model_file)

                # Define the path for saving the confusion matrix plot
                conf_matrix_plot_path = os.path.join(model_dir, f"{model_name}_confusion_matrix.png")

                # Check if the confusion matrix plot already exists
                if os.path.exists(conf_matrix_plot_path):
                    print(f"Confusion matrix plot already exists for model: {model_name}. Skipping...")
                    continue

                # Load the model weights
                # Instantiate the model and move it to the GPU
                if model_type == 'LSTM':
                    print('Using LSTM')
                    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size).to(device)
                elif model_type == "CNN":
                    print('Using CNN')
                    slice_length = 32
                    model = CNNModel2D(slice_len=slice_length).to(device)
                else:
                    # Extract slice size from the model name
                    slice_size_match = re.findall(r'\.(\d+)', model_name)
                    if len(slice_size_match) == 2:
                        n_heads = int(slice_size_match[0])
                        slice_length = int(slice_size_match[1])
                        print(f'Using Transformer with slice size {slice_length} and {n_heads} heads')
                        model = TransformerNN(slice_len=slice_length, nhead=n_heads).to(device)
                    else:
                        print(f"Invalid model name format: {model_name}. Skipping...")
                        continue
                model.load_state_dict(torch.load(model_path))

                # Add the model and its name to the list as a tuple
                model_list.append((model, model_name, slice_length))

    for model, model_name, slice_length in model_list:
        # Set the model to evaluation mode
        print(f'Model name: {model_name}, slice length: {slice_length}')
        model.eval()
        # Create a DataLoader for the test data
        batch_size = 1000
        test_dataset = PTPDataset(features_tensor, labels_tensor, slice_length, stride=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Lists to store predictions and targets
        test_predictions = []
        test_targets = []
        # Iterate through the test data
        with torch.no_grad():  # Disable gradient calculation during testing
            for inputs, labels in test_loader:
                # Reshape the inputs for CNN (add a channel dimension)
                if isinstance(model, CNNModel2D):
                    inputs = inputs.unsqueeze(1)  # Shape becomes (batch_size, 1, slice_len, num_features)

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = torch.round(outputs)  # Round the predictions to 0 or 1
                test_predictions.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
        # Generate the confusion matrix
        conf_matrix = confusion_matrix(test_targets, test_predictions)
        # Normalize the confusion matrix to display percentages
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        # Print the confusion matrix
        print("Confusion Matrix:")
        print(conf_matrix_normalized)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=False, annot_kws={"size": 24})
        plt.xlabel('Predicted Label', fontsize=18)
        plt.ylabel('True Label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # plt.title(f'Confusion Matrix - Model {model_name}')
        # Save the confusion matrix plot as a .png file
        plt.savefig(os.path.join(model_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
