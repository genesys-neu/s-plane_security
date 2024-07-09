import os
import torch
from torch.utils.data import DataLoader
from train_test import load_data, PTPDataset, LSTMClassifier, TransformerNN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse
import numpy as np
import re


np.random.seed(17)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", default='./',
                        help="directory containing all the models to be tested")
    parser.add_argument("-f", "--file_input", default='final_dataset.csv',
                        help="file containing all the training data")
    parser.add_argument("-m", "--model", default='Transformer', help="Chose Transformer or LSTM")

    args = parser.parse_args()
    model_dir = args.directory
    input_file = args.file_input
    model_type = args.model
    chunk_size = 1000

    # Load the data
    _, _, test_data, _, _, test_label = load_data(input_file, chunk_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    input_size = test_data.shape[1]
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
                    model = LSTMClassifier(input_size, hidden_size).to(device)
                else:
                    # Extract slice size from the model name
                    # print(f'model name: {model_name}')
                    slice_size = re.findall(r'\.(\d+)', model_name)
                    # print(f'Slice Size {slice_size}')
                    slice_length = int(slice_size[1])
                    n_heads = int(slice_size[0])
                    print(f'Using Transformer with slice size {slice_length} and {n_heads} heads')
                    model = TransformerNN(slice_len=slice_length, nhead=n_heads).to(device)
                model.load_state_dict(torch.load(model_path))

                # Add the model and its name to the list as a tuple
                model_list.append((model, model_name, slice_length))

    for model, model_name, slice_length in model_list:
        # Set the model to evaluation mode
        print(f'Model name: {model_name}, slice length: {slice_length}')
        model.eval()
        # Create a DataLoader for the test data
        batch_size = 1000
        test_dataset = PTPDataset(test_data, test_label, slice_length)
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
        plt.figure(figsize=(6, 3))
        sns.heatmap(toverall_confusion_matrix_bg, annot=True, fmt='.2%', cmap='Blues', cbar=False,
                    annot_kws={"size": 24})
        plt.xlabel('Predicted Label', fontsize=18)
        plt.ylabel('True Label', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # Remove internal gridlines
        plt.grid(False)
        # Save the confusion matrix plot as a .png file
        plt.savefig(os.path.join(model_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()
