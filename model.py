import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self,grid_num) -> None:
        super().__init__()
        self.device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.target_num = 2
        self.ego_state_num = 5
        # Wx, Wh, b should be tensor using GPU
        self.linear = nn.Linear(grid_num + self.target_num + self.ego_state_num,1)
        print(self.linear)
        quit()
        
    # getting the y from the x
    def forward(self,x):
        return self.linear(x)
