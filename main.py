# `batch_first=True` is required for use in `nn.CTCLoss`.

from torch.utils.data import DataLoader
import pandas as pd

from phoneme_list import *
from model import *
from dataloader import *
from train import *
from decoder import *

if __name__ == "__main__":

    """Parameters"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    """Data loaders"""
    path_train_dataset_x = r"HW3P2_Data/wsj0_dev.npy"
    path_train_dataset_y = r"HW3P2_Data/wsj0_dev_merged_labels.npy"
    path_vali_dataset_x = r"HW3P2_Data/wsj0_dev.npy"
    path_vali_dataset_y = r"HW3P2_Data/wsj0_dev_merged_labels.npy"
    path_pred_dataset_x = r"HW3P2_Data/wsj0_test.npy"
    train_batch_size = 2
    vali_batch_size = 32


    """Data Loaders"""

    train_loader = DataLoader( TrainDataset(path_train_dataset_x, path_train_dataset_y), shuffle=False, batch_size=train_batch_size, collate_fn = Train_collate_pad)
    vali_loader = DataLoader( TrainDataset(path_vali_dataset_x, path_vali_dataset_y), shuffle=False, batch_size=vali_batch_size, collate_fn = Train_collate_pad, drop_last=True )
    pred_loader = DataLoader( PredictDataset(path_pred_dataset_x), shuffle=False, batch_size=vali_batch_size, collate_fn = Predict_collate_pad)

    """"""
    #(in_vocab, out_vocab, layers, hidden_size):
    model = Model(40, 47, 1, 10)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    criterion = nn.CTCLoss()

    train_epoch(model, optimizer, criterion, train_loader, vali_loader, DEVICE)
    torch.save(model.state_dict(), "model.pt")
    model.load_state_dict(torch.load("model.pt"))
    prediction = predict(model, pred_loader, DEVICE)
    print(prediction)

    df = pd.read_csv(r"HW3P2_Data/sample_submission.csv")
    df["Predicted"] = prediction
    df.to_csv(r"prediction.csv",index=False)







