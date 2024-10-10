import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from collections import Counter
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        # self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss()    # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.activation(self.W1(input_vector))
        # hidden = self.dropout(hidden)
        # [to fill] obtain output layer representation
        hidden = self.activation(self.W2(hidden))
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(hidden)
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val

def compute_summary_statistics(data, dat_name):
    N = len(data)
    labels = Counter()
    word_count = 0
    doc_lengths = []
    for document, label in data:
        labels[label] += 1
        doc_length = len(document)
        word_count += doc_length
        doc_lengths.append(doc_length)
    avg_doc_length = word_count/N
    vocab_size = len(set(word for document, _ in data for word in document))

    print("========== {} Summary Statistics ==========".format(dat_name))
    print("Number of samples: {}".format(N))
    print("Label distribution: {}".format(labels))
    print("Vocabulary size: {}".format(vocab_size))

    # plt.figure()
    # labels, counts = zip(*sorted(labels.items()))
    # plt.bar(labels, counts, tick_label=[label + 1 for label in labels])
    # plt.title("{} Label Distribution".format(dat_name))
    # plt.xlabel("Rating")
    # plt.ylabel("Number of Samples")
    # plt.show()

    # plt.figure()
    # plt.hist(doc_lengths, bins=50)
    # plt.title("{} Input Length Distribution".format(dat_name))
    # plt.xlabel("Input Length (words)")
    # plt.ylabel("Number of Documents")
    # plt.show()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    # Compute summary statistics
    compute_summary_statistics(train_data, "Training Data")
    compute_summary_statistics(valid_data, "Validation Data")
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    
    print("========== Training for {} epochs ==========".format(args.epochs))
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        epoch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_acc.append(correct/total)
        train_loss.append(epoch_loss/total)

        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        model.eval()
        loss = None
        correct = 0
        total = 0
        val_l = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        true_val_labels = []
        predicted_val_labels = []
        N = len(valid_data) 
        with torch.no_grad():
            for minibatch_index in tqdm(range(N // minibatch_size)):
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    true_val_labels.append(gold_label)
                    predicted_val_labels.append(predicted_label)
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                val_l += loss.item()
        val_acc.append(correct/total)
        val_loss.append(val_l/total)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
    print("========== Classification Report for Validation Dataset ==========".format(epoch + 1))
    print(classification_report(true_val_labels, predicted_val_labels, labels=[0, 1, 2, 3, 4]))

    # Plots for training and validation
    plt.figure()
    epochs = range(1, args.epochs + 1)
    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
   #  write out to results/test.out
    