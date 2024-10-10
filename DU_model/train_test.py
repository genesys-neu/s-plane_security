import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset
import argparse
from concurrent.futures import ThreadPoolExecutor
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


np.random.seed(17)


class SklearnMLModel:
    def __init__(self, model, verbose=0):
        self.model = model
        self.verbose = verbose

    def train(self, X_train, y_train):
        start_time = time.time()

        # Check if the model supports the verbose parameter
        if hasattr(self.model, 'verbose'):
            self.model.verbose = self.verbose

        if self.verbose > 0:
            print("Starting training...")

        # Train the model
        self.model.fit(X_train, y_train)

        if self.verbose > 0:
            end_time = time.time()
            print(f"Training completed in {end_time - start_time:.2f} seconds")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            # For models like SVM that do not support predict_proba by default
            return self.model.decision_function(X)


class CNNModel2D(nn.Module):
    def __init__(self, slice_len=32, num_features=6, num_classes=1):
        super(CNNModel2D, self).__init__()

        # Define a simpler 2D convolutional network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # Reduces size by a factor of 2

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))  # Reduces size by a factor of 2

        # Compute the size of the flattened layer
        flattened_size = (slice_len // 4) * (num_features // 4) * 32

        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        # print("Input shape:", x.shape)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # print("After conv1 + pool1:", x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print("After conv2 + pool2:", x.shape)

        x = x.view(x.size(0), -1)
        # print("After flattening:", x.shape)

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


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

        # Forward propagate through the output layer
        out = self.fc(lstm_out)
        # print(f'out shape: {out.shape}')
        # Apply sigmoid activation function
        out = torch.sigmoid(out)
        return out


class TransformerNN(nn.Module):
    def __init__(self, classes: int = 1, num_feats: int = 6, slice_len: int = 32, nhead: int = 3, nlayers: int = 2,
                 dropout: float = 0.2, use_pos: bool = False):
        super(TransformerNN, self).__init__()
        self.norm = nn.LayerNorm(num_feats)

        # define the encoder layers
        encoder_layers = TransformerEncoderLayer(num_feats, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = num_feats

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(num_feats * slice_len, 256)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(256, classes)


    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classes log probabilities
        """
        # src = self.norm(src) should not be necessary since output can be already normalized
        # pass through encoder layers
        t_out = self.transformer_encoder(src)
        # flatten already contextualized KPIs
        t_out = torch.flatten(t_out, start_dim=1)
        # Pass through MLP classifier
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = torch.sigmoid(output)
        return output


class PTPDataset(Dataset):
    def __init__(self, data, labels, sequence_length):
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        sample = torch.tensor(self.data.iloc[idx:end_idx].values, dtype=torch.float32)
        # Convert labels from Pandas Series to NumPy array, then to tensor
        label_tensor = torch.tensor(self.labels.iloc[idx:end_idx].values, dtype=torch.long)
        # Determine label for the sequence
        label = torch.tensor(1 if torch.any(label_tensor) else 0, dtype=torch.long)
        return sample, label


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_fp=1, weight_fn=1):
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
    valid_test_indices = np.random.choice(num_chunks, size=int(num_chunks * 0.1), replace=False)

    # Separate validation and testing indices
    # Modified to increase training size: we use the same selection for validation and testing
    valid_indices = valid_test_indices
    test_indices = valid_test_indices

    # Initialize lists to store training, validation, and testing data
    training_data = []
    validation_data = []
    testing_data = []

    # Iterate over each chunk and determine if it belongs to training, validation, or testing
    for i, chunk in enumerate(chunks):
        if i in valid_indices:
            validation_data.extend(chunk.values.tolist())  # Append rows to validation_data
            testing_data.extend(chunk.values.tolist())  # Append rows to testing_data
        # elif i in test_indices:
            # testing_data.extend(chunk.values.tolist())  # Append rows to testing_data
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
    parser.add_argument("-m", "--model", default='Transformer',
                        help="Choose Transformer, LSTM, CNN, LogisticRegression, KNN, DecisionTree, GBC, NaiveBayes, RandomForest")
    parser.add_argument("-s", "--slice_length", type=int, default=32, help="Slice length for the Transformer")
    parser.add_argument("-nh", "--num_heads", type=int, default=3, help="Number of Transformer heads")

    args = parser.parse_args()
    input_file = args.file_input
    t_v = args.trial_version
    model_type = args.model
    slice_length = args.slice_length
    num_heads = args.num_heads

    chunk_size = 500
    training_metrics = {'epochs': [], 'training_loss': [], 'training_accuracy': [], 'validation_loss': [],
                        'validation_accuracy': [], 'confusion_matrix': []}

    train_data, validate_data, test_data, train_label, validate_label, test_label = load_data(input_file, chunk_size)

    train_dataset = PTPDataset(train_data, train_label, slice_length)
    val_dataset = PTPDataset(validate_data, validate_label, slice_length)
    test_dataset = PTPDataset(test_data, test_label, slice_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model parameters
    input_size = train_data.shape[1]
    hidden_size = 64  # For LSTM
    num_layers = 2
    output_size = 1
    num_epochs = 100

    # Instantiate the model based on the user's choice
    if model_type == 'LSTM':
        print('Using LSTM')
        model = LSTMClassifier(input_size, hidden_size).to(device)
    elif model_type == 'Transformer':
        print(f'Using Transformer with {num_heads} heads and slice size {slice_length}')
        model = TransformerNN(slice_len=slice_length, nhead=num_heads).to(device)
    elif model_type == 'CNN':
        print(f'Using CNN with slice size {slice_length}')
        model = CNNModel2D(slice_len=slice_length).to(device)
    elif model_type == 'LogisticRegression':
        print('Using Logistic Regression')
        model = SklearnMLModel(LogisticRegression(), verbose=1)
    elif model_type == 'KNN':
        print('Using k-Nearest Neighbors')
        model = SklearnMLModel(KNeighborsClassifier(), verbose=1)
    elif model_type == 'DecisionTree':
        print('Using Decision Tree')
        model = SklearnMLModel(DecisionTreeClassifier(), verbose=1)
    elif model_type == 'GBC':
        print('Using Gradient Boosting Classifier')
        model = SklearnMLModel(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, verbose=1))
    elif model_type == 'NaiveBayes':
        print('Using Naive Bayes')
        model = SklearnMLModel(GaussianNB(), verbose=1)
    elif model_type == 'RandomForest':
        print('Using Random Forest')
        model = SklearnMLModel(RandomForestClassifier(), verbose=1)
    else:
        raise ValueError(
            "Invalid model type chosen. Please choose from: "
            "Transformer, LSTM, CNN, LogisticRegression, KNN, DecisionTree, SVM, NaiveBayes, or RandomForest.")

    # Handle the training and evaluation for sklearn models
    if isinstance(model, SklearnMLModel):
        print("Loading training and test data...")

        # Convert DataFrames to NumPy arrays
        X_train = train_data.to_numpy()  # Convert training data to NumPy array
        y_train = train_label.to_numpy()  # Convert training labels to NumPy array

        X_test = test_data.to_numpy()  # Convert test data to NumPy array
        y_test = test_label.to_numpy()  # Convert test labels to NumPy array

        # Print shapes and data types for debugging
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

        print(f"Data type of X_train: {X_train.dtype}")
        print(f"Data type of y_train: {y_train.dtype}")
        print(f"Data type of X_test: {X_test.dtype}")
        print(f"Data type of y_test: {y_test.dtype}")

        # Train the scikit-learn model
        print("Starting training...")
        model.train(X_train, y_train)

        # Evaluate the scikit-learn model
        print("Starting evaluation...")
        predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, predictions)
        test_recall = recall_score(y_test, predictions, average='binary')
        print(f'Test Accuracy for {model_type}: {test_accuracy:.4f}')
        print(f"Recall: {test_recall:.4f}")

    else:

        # Define the loss function (Binary Cross-Entropy Loss)
        # criterion = nn.BCELoss()
        # Define the loss function (Custom Weighted Binary Cross-Entropy Loss)
        criterion = WeightedBCELoss(weight_fp=1, weight_fn=1)

        # Define optimizer (Adam) and learning rate scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_val_loss = float('inf')  # Initialize the best validation loss
        patience = 15  # Number of epochs to wait for improvement
        counter = 0  # Counter for patience

        # Training loop with evaluation on the validation set
        for epoch in range(num_epochs):
            # Randomly select batch size between 10 and 40
            # reset the random seed?
            # np.random.seed()
            if model_type == 'LSTM':
                batch_size = np.random.randint(10, 41)
            else:
                batch_size = 1000
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Training phase
            model.train()
            running_loss = 0.0
            running_accuracy = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()

                try:
                    # Move inputs and labels to the GPU
                    # print(f'Input dimensions {inputs.size()}, Labels dimensions {labels.size()}')
                    inputs, labels = inputs.to(device), labels.to(device)
                    # Print input dimensions before any modification
                    # print(f"Original input shape: {inputs.shape}")

                    # Reshape the inputs for CNN (add a channel dimension)
                    if isinstance(model, CNNModel2D):
                        inputs = inputs.unsqueeze(1)  # Shape becomes (batch_size, 1, slice_len, num_features)
                        # print(f"Input shape after unsqueeze for CNN: {inputs.shape}")

                    # Forward pass
                    outputs = model(inputs)
                    # print(f'Output shape: {outputs.shape}')
                    # print(f'Outputs: {outputs}')

                    # Round the predictions to 0 or 1
                    predicted = torch.round(outputs)
                    predicted = predicted.flatten()
                    # print(f'Predicted shape: {predicted.shape}')
                    # print(f'Predicted: {predicted}')
                    # Adjust shapes for the last batch

                    outputs = outputs.flatten()  # Flatten the output tensor
                    # labels = labels.float().view(-1)  # Flatten the label tensor
                    # print(f'Labels shape: {labels.shape}')
                    # Use raw probabilities in the loss calculation
                    loss = criterion(outputs, labels.float())
                    running_accuracy += accuracy(predicted, labels)

                    # Use rounded predictions in the loss calculation
                    # loss = criterion(predicted.squeeze(), labels.float())  # Use predicted instead of outputs
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                except ValueError as e:
                    print(f'Error occurred in training epoch {epoch +1}: {e}')
                    continue

            # Validation phase
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():  # Disable gradient calculation during validation
                for inputs, labels in val_loader:
                    try:
                        inputs, labels = inputs.to(device), labels.to(device)

                        # Reshape the inputs for CNN (add a channel dimension)
                        if isinstance(model, CNNModel2D):
                            inputs = inputs.unsqueeze(1)  # Shape becomes (batch_size, 1, slice_len, num_features)

                        outputs = model(inputs)

                        # Round the predictions to 0 or 1
                        predicted = torch.round(outputs)
                        predicted = predicted.flatten()

                        # print(f'Outputs: {outputs.shape}, Labels: {labels.shape}')
                        outputs = outputs.flatten()  # Flatten the output tensor
                        # labels = labels.float().view(-1)  # Flatten the label tensor
                        # print(f'Outputs: {outputs.shape}, Labels: {labels.shape}')
                        # print(f'Outputs: {outputs.shape}, labels: {labels.shape}')
                        loss = criterion(outputs, labels.float())
                        val_accuracy += accuracy(predicted, labels)

                        val_loss += loss.item()

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
            if (val_loss / len(val_loader)) < best_val_loss:
                best_val_loss = (val_loss / len(val_loader))
                # print(f'Best validation loss updated: {best_val_loss}')
                torch.save(model.state_dict(), f'best_model_{t_v}.pth')
                counter = 0  # Reset counter if there's improvement
            else:
                # Increment counter if there's no improvement
                counter += 1
            # print(f'Counter: {counter}')
            # Check early stopping condition
            if counter >= patience:
                print(f'Validation loss has not improved for {patience} epochs. Stopping training.')
                break

        # Test phase
        # Load the best model saved during training
        model.load_state_dict(torch.load(f'best_model_{t_v}.pth'))

        model.eval()  # Set model to evaluation mode
        batch_size = 1000
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        test_predictions = []
        test_targets = []

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

        # Generate confusion matrix
        conf_matrix = confusion_matrix(test_targets, test_predictions)
        print("Confusion Matrix:")
        print(conf_matrix)
        # Convert confusion matrix from NumPy array to list
        conf_matrix_list = conf_matrix.tolist()
        training_metrics['confusion_matrix'].append(conf_matrix_list)

        # Save the dictionary to a JSON file
        with open(f'training_log_{t_v}.json', 'w') as jsonfile:
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
