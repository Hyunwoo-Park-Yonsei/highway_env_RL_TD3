import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Hyperparameters
learning_rate = 0.1
gamma         = 0.98


# Reinforce Algorithm 

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        # v,a,
        self.fc1 = nn.Linear(400,128)
        self.fc2 = nn.Linear(128,4)
        self.optimizer = optim.Adam(self.parameters(), lr = learning_rate)

        torch.nn.init.zeros_(self.fc1.weight.data)
        torch.nn.init.zeros_(self.fc2.weight.data)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def put_data(self, item):
        self.data.append(item)
    
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            print("r",r)
            print("prob",prob)
            R = r + gamma + R
            loss = -R * torch.log(prob)
            print("loss ",loss)
            loss.backward()
        self.optimizer.step()
        self.data = []

