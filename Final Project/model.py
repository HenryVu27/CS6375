import torch.nn as nn
import torch

from conformer import Conformer

class MultiInputModel(nn.Module):
    def __init__(self, text_embedding_size, eeg_feature_size, hidden_size, num_classes):
        super(MultiInputModel, self).__init__()

        self.Conformer = Conformer()
        self.fc1 = nn.Linear(text_embedding_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.eeg_model=Conformer()

    def forward(self, text_input, eeg_input): #eeg input (batch size x time dimension x channel)
        eeg_input=torch.transpose(eeg_input, 1,2) 
        eeg_input=eeg_input.unsqueeze(1).float() #Now  (batch size x 1 x channel x time dimension)
        eeg_input=self.Conformer(eeg_input)
        combined = torch.cat((text_input, eeg_input), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.fc2(x)
        x = self.softmax(x)
        return x