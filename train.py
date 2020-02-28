from torch import nn
import time
import numpy as np
from decoder import *

def train_epoch(model, optimizer, criterion, train_loader, val_loader, DEVICE):
    criterion = criterion.to(DEVICE)
    before = time.time()
    print("training", len(train_loader), "number of batches")
    for batch_idx, (inputs, inputs_length, targets, targets_length) in enumerate(train_loader):
        optimizer.zero_grad()
        if batch_idx == 0:
            first_time = time.time()
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs, outputs_length = model(inputs, inputs_length)
        #print("output: ", outputs)
        #print(outputs.shape, targets.shape, outputs_length.shape, targets_length.shape)
        #print(targets_length)
        #print (targets)
        loss = criterion(outputs, targets, outputs_length, targets_length)
        #loss = criterion(outputs.view(-1, outputs.size(2)), targets.view(-1))  # Loss of the flattened outputs

        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print("Time elapsed", time.time() - first_time)

        if batch_idx % 100 == 0 and batch_idx != 0:
            after = time.time()
            print("Time: ", after - before)
            print("Loss per word: ", loss.item())
            print("Perplexity: ", np.exp(loss.item()))
            after = before

    val_loss = 0
    batch_id = 0
    for inputs, inputs_length, targets, targets_length in val_loader:
        batch_id += 1
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        outputs, outputs_length = model(inputs, inputs_length)
        loss = criterion(outputs, targets, outputs_length, targets_length)
        val_loss += loss.item()
    val_lpw = val_loss / batch_id
    print("\nValidation loss per word:", val_lpw)
    print("Validation perplexity :", np.exp(val_lpw), "\n")
    return val_lpw

def predict(model, predict_loader, DEVICE):

    result = []
    for batch_idx, (inputs, inputs_length) in enumerate(predict_loader):

        inputs = inputs.to(DEVICE)
        outputs, outputs_length = model(inputs, inputs_length)

        outputs = outputs.to("cpu")
        outputs_length = outputs_length.to("cpu")
        res = CTCDecode(outputs, outputs_length)
        result += res

    return result