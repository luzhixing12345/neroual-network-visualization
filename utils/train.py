

import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
sys.path.append('.')
from model.dataset import get_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class Runthread(QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(list)

    def __init__(self,model,train_dataloader,test_dataloader,graph):
        super(Runthread,self).__init__()
        self.model = model
        self.train_dataloader=train_dataloader
        self.test_dataloader = test_dataloader
        self.graph = graph
    
    def run(self):
        loss_fn = nn.CrossEntropyLoss()  
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        epochs = 10
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(self.train_dataloader, self.model, loss_fn, optimizer)
            test(self.test_dataloader,self.model, loss_fn,self.graph)
        print("Done!")

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn,graph):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    graph.show_train_process(model)
    graph.update_pic()
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


