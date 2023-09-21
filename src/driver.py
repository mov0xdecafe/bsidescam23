import argparse

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from torch.utils import data

from dataset import FunctionIdentificationDataset  # Import the dataset module
from model import CNNModel  # Import the CNN model module

KERNEL_SIZE = 20  # Kernel size for convolutional layer

def train_model(model, training_dataset):
    """
    Train the CNN model on the training dataset.

    Args:
        model (nn.Module): The CNN model.
        training_dataset (data.Dataset): The training dataset.
    """
    loss_function = nn.NLLLoss()  # Negative Log Likelihood loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    training_loader = data.DataLoader(training_dataset, shuffle=True)  # DataLoader for training data
    model.train()  # Set the model to training mode
    for sample, features in tqdm.tqdm(training_loader):  # Iterate through training data
        sample = sample[0]
        features = features[0]
        model.zero_grad()

        feature_scores = model(sample)  # Get feature scores from the model

        loss = loss_function(feature_scores, features)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

def test_model(model, test_dataset):
    """
    Test the trained CNN model on the test dataset.

    Args:
        model (nn.Module): The trained CNN model.
        test_dataset (data.Dataset): The test dataset.
    """
    test_loader = data.DataLoader(test_dataset)  # DataLoader for test data
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        all_features = []
        all_feature_scores = []

        for sample, features in tqdm.tqdm(test_loader):  # Iterate through test data
            sample = sample[0]
            features = features[0]

            feature_scores = model(sample)  # Get feature scores from the model

            all_features.extend(features.numpy())
            all_feature_scores.extend(feature_scores.numpy())
        
        all_features = numpy.array(all_features)
        all_feature_scores = numpy.array(all_feature_scores).argmax(axis=1)

        # Calculate and print evaluation metrics
        accuracy = accuracy_score(all_features, all_feature_scores)
        pr = precision_score(all_features, all_feature_scores, average='macro')
        recall = recall_score(all_features, all_feature_scores, average='macro')
        f1 = f1_score(all_features, all_feature_scores, average='macro')

        print("accuracy: {}".format(accuracy))
        print("precision: {}".format(pr))
        print("recall: {}".format(recall))
        print("f1-score: {}".format(f1))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="Path to directory containing binaries to include in dataset")

    args = arg_parser.parse_args()

    print("Function Boundary Identification")

    # Create the dataset using the specified path
    dataset = FunctionIdentificationDataset(args.dataset_path, block_size=1000, padding_amount=KERNEL_SIZE-1)

    # Split the dataset into training and test sets
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    training_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

    # Create the CNN model
    model = CNNModel(vocab_size=258, embedding_dim=64, hidden_dim=16, kernel_size=KERNEL_SIZE, featureset_size=3)

    print("[*] Training Model")
    train_model(model, training_dataset)
    print("[+] Model Trained")

    print("[*] Testing Model")
    test_model(model, test_dataset)
