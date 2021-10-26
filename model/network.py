import torch
import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):
    def __init__(self,c=1,h=28,w=28) -> None:
        super().__init__()
        
        self.layers = []
        self.layers_arguments =[{'orgin':[c,h,w]}]
    
    def link_layer(self):
        self.sequential = nn.Sequential()
        for id,layer in enumerate(self.layers):
            self.sequential.add_module(f"{id}",layer)


    def forward(self,x):
        x = self.sequential(x)
        return x


class example_net_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 15, kernel_size=4)
        self.conv2 = nn.Conv2d(15, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view([-1, 320])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x 

class example_net_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
