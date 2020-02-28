import torch
from torch import nn
from torch.nn.utils.rnn import *


class Model(nn.Module):

    def __init__(self, input_size, output_size, num_layers, hidden_size):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.output = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.output1 = nn.Linear(hidden_size * 4, output_size)

    def forward(self, X, lengths):
        #X = self.embed(X)
        #print("X:",X)
        #print("length:",lengths)
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        # Log softmax after output layer is required for use in `nn.CTCLoss`.
        out = self.output(out).relu()
        out = self.output1(out).log_softmax(2)
        #print("out: ", out.shape)
        #print("out_len",out_lens.shape)
        #assert(False)
        return out, out_lens