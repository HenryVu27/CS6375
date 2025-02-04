import config
from feature_extraction import zuco_reader
import ml_helpers
from data_helpers import save_results, load_matlab_files
import numpy as np

from datetime import timedelta
import time
from conformer import Conformer
import torch

from sklearn.model_selection import KFold
import sklearn.metrics
import torch.optim as optim
import torch.nn as nn
device='cuda'
from model import MultiInputModel
from dataset import TextEEGDataset, dict2features
from torch.utils.data import DataLoader, random_split

def main():
    print('Starting Loop')
    start = time.time()
    count = 0

    text_features, text_masks, eeg_features, labels=dict2features()
    dataset = TextEEGDataset(text_features, text_masks, eeg_features, labels)
    
    # text features (400,60)
    # eeg features (400, 41, 105)
    
    model = MultiInputModel(text_embedding_size=60, eeg_feature_size=105, hidden_size=32, num_classes=3)
    model.cuda()
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for seed in config.random_seed_values:
        train(dataset, model, criterion,optimizer, seed)


def train(dataset, model, criterion,optimizer, seed):
    kf = KFold(n_splits=config.folds, random_state=seed, shuffle=True)
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
    # Split train_val set into train and validation
        train_val_subset = torch.utils.data.Subset(dataset, train_val_idx)
        len_train = int(len(train_val_subset) * 0.9)
        len_val = len(train_val_subset) - len_train
        train_subset, val_subset = random_split(train_val_subset, [len_train, len_val])

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=1, shuffle=False)

        # Training loop for the fold
        train_loss=0.0
        for epoch in range(config.epochs):
            # Training step
            model.train()
            
            for text_inputs, text_masks, eeg_inputs, labels in train_loader:

                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass to get outputs
                predictions = model(text_inputs, eeg_inputs)
                
                
                # Compute the loss based on predictions and actual labels
                labels=torch.nn.functional.one_hot(labels, num_classes=3,).float()
                loss = criterion(predictions, labels)
                
                # Backward pass to calculate the gradient
                loss.backward()
                
                # Update the parameters with optimizer
                optimizer.step()
                
                # Sum up the loss for the epoch
                train_loss += loss.item()
            
            # Average loss for the epoch
            train_loss /= len(train_loader)

            # Validation step
            val_loss = 0.0
            correct = 0
            total = 0
            model.eval()
            
            with torch.no_grad():
                for text_inputs, text_masks, eeg_inputs, labels in val_loader:
                    # Forward pass
                    predictions = model(text_inputs, eeg_inputs)
                    
                    # Compute validation loss
                    onehot_labels=torch.nn.functional.one_hot(labels, num_classes=3).float()
                    loss = criterion(predictions, onehot_labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(predictions.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
            # Average validation loss for the epoch
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            
            # Print statistics
            print(f'Epoch {epoch+1}/{config.epochs}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Validation Loss: {val_loss:.4f}, '
                f'Validation Accuracy: {val_accuracy:.2f}%')

        # After all epochs
        print(f"Finished fold {fold}")
    
        # Testing step
        correct = 0
        total = 0
        model.eval()
        preds=[]
        labels=[]
        with torch.no_grad():
            for text_input, eeg_input, label in test_loader:
                # Forward pass
                prediction = model(text_input, eeg_input)
                
                # Calculate accuracy
                _, predicted = torch.max(prediction.data, 1)
                preds.append(predicted)
                labels.append(label)
                total += labels.size(0)
                correct += (predicted == label).sum().item()

        # Average test loss and accuracy for the fold
        test_accuracy = 100 * correct / total
        print(f"test accuracy is: {test_accuracy}")
        p, r, f, _ = sklearn.metrics.precision_recall_fscore_support(label, preds,
                                                                           average='micro')
        print(p, r, f)





if __name__ == '__main__':
    main()