import argparse

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
from torch.utils import data

from dataset import FunctionIdentificationDataset
from model import CNNModel

KERNEL_SIZE = 20

def train_model(model, training_dataset):
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    training_loader = data.DataLoader(training_dataset, shuffle=True)
    model.train()
    for sample, features in tqdm.tqdm(training_loader):
        sample = sample[0]
        features = features[0]
        model.zero_grad()

        feature_scores = model(sample)

        loss = loss_function(feature_scores, features)
        loss.backward()
        optimizer.step()

def test_model(model, test_dataset):
    test_loader = data.DataLoader(test_dataset)
    model.eval()
    with torch.no_grad():
        all_features = []
        all_feature_scores = []

        for sample, features in tqdm.tqdm(test_loader):
            sample = sample[0]
            features = features[0]

            feature_scores = model(sample)

            all_features.extend(features.numpy())
            all_feature_scores.extend(feature_scores.numpy())
        
        all_features = numpy.array(all_features)
        all_feature_scores = numpy.array(all_feature_scores).argmax(axis=1)

        accuracy = accuracy_score(all_features, all_feature_scores)
        pr = precision_score(all_features, all_feature_scores)
        recall = recall_score(all_features, all_feature_scores)
        f1 = f1_score(all_features, all_feature_scores)

        print("accuracy: {}".format(accuracy))
        print("pr: {}".format(pr))
        print("recall: {}".format(recall))
        print("f1: {}".format(f1))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dataset_path", help="Path to directory contain binaries to include in dataset")

    args = arg_parser.parse_args()

    print("Function Boundary Identification")
    dataset = FunctionIdentificationDataset(args.dataset_path, block_size=1000, padding_amount=KERNEL_SIZE-1)

    train_size = int(len(dataset) * 0.9)
    test_size = int(dataset) - train_size
    training_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    model = CNNModel(vocab_size=258, embedding_dim=64, hidden_dim=16, kernel_size=KERNEL_SIZE, featureset_size=2)

    print("[*] Training Model")
    train_model(model, training_dataset)
    print("[+] Model Trained")

    print("[*] Testing Model")
    test_model(model, test_dataset)