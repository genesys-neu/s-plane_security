import os
import torch
from torch.utils.data import DataLoader
from train_test import load_data, PTPDataset, LSTMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np


np.random.seed(17)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default='./',
                        help="directory containing all the models to be tested")
    parser.add_argument("-f", "--file_input", default='final_dataset.csv',
                        help="file containing all the training data")

    args = parser.parse_args()
    model_dir = args.directory
    input_file = args.file_input
    chunk_size = 100

    # Load the data
    _, _, test_data, _, _, test_label = load_data(input_file, chunk_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    input_size = test_data.shape[1]
    hidden_size = 64
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

                # Load the model weights
                model = LSTMClassifier(input_size, hidden_size).to(device)
                model.load_state_dict(torch.load(model_path))

                # Add the model and its name to the list as a tuple
                model_list.append((model, model_name))

    for model, model_name in model_list:
        # Set the model to evaluation mode
        model.eval()
        # Create a DataLoader for the test data
        batch_size = 25
        test_dataset = PTPDataset(test_data, test_label)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Lists to store predictions and targets
        test_predictions = []
        test_targets = []
        # Iterate through the test data
        with torch.no_grad():  # Disable gradient calculation during testing
            for inputs, labels in test_loader:
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
        sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - Model {model_name}')
        # Save the confusion matrix plot as a .png file
        plt.savefig(f"{model_name}_confusion_matrix.png")
        plt.close()
