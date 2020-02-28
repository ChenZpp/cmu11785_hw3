import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class TrainDataset(Dataset):
    def __init__ (self, xpath, ypath):
        self.xpath = xpath
        self.ypath = ypath
        try:
            self.x = np.load(xpath, allow_pickle=True, encoding = "latin1")
            self.y = np.load(ypath, allow_pickle=True, encoding = "latin1")+1
        except:
            print("error train data set path: ")
            print("xpath: ", self.xpath)
            print("ypath: ", self.ypath)

        assert(self.x.shape[0] == self.y.shape[0])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])

class PredictDataset(Dataset):
    def __init__ (self, xpath):
        self.xpath = xpath
        try:
            self.x = np.load(xpath, allow_pickle=True, encoding = "latin1")
        except:
            print("error predict data set path: ")
            print("xpath: ", self.xpath)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index])

def Train_collate_packed(seq_list):

    inputs, targets = zip(*seq_list)
    lens = [seq.shape[0] for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = [targets[i] for i in seq_order]
    return inputs, targets

def Predict_collate_packed(seq_list):

    inputs = seq_list
    lens = [seq.shape[0] for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    return inputs
    '''
    inputs = torch.cat([s.unsqueeze(1) for s in seq_list], dim=1)
    return inputs
    '''

def Train_collate_pad(batch):
    inputs, targets = zip(*batch)
    len_xs = torch.LongTensor([x.shape[0] for x in inputs])
    len_ys = torch.LongTensor([y.shape[0] for y in targets])

    #print("pad inputs: ", len(inputs))
    #print(inputs[0])
    pad_inputs = pad_sequence(inputs)
    pad_targets = pad_sequence(targets, batch_first=True)

    return pad_inputs, len_xs, pad_targets, len_ys

def Predict_collate_pad(batch):

    inputs = batch
    len_xs = torch.LongTensor([x.shape[0] for x in inputs])
    pad_inputs = pad_sequence(inputs)
    return pad_inputs, len_xs


if __name__ == "__main__":

    xpath = r"HW3P2_Data/wsj0_dev.npy"
    ypath = r"HW3P2_Data/wsj0_dev_merged_labels.npy"
    train_dataset = TrainDataset(xpath, ypath)
    print("train data set has a sample size of: ", train_dataset.__len__())
    print("The first x has a shape of: ", train_dataset.__getitem__(0)[0].shape)
    print("The first y has a shape of: ", train_dataset.__getitem__(0)[1].shape)

    xpath = r"HW3P2_Data/wsj0_test.npy"
    predict_dataset = PredictDataset(xpath)
    print("Predict data set has a size of: ", predict_dataset.__len__())


