from cmath import nan
from xml.etree.ElementTree import PI
import scipy.stats
import numpy as np
import math
import torch

def printObs(obs):
    print("================================================================================")
    for a in obs:
        for b in a:
            s = '%-5s' % str(round(b*100,1))
            # if col_count == col or row_count == row:
            #     s = '%-s' % " "
            print(s,end=" ")
        print(" ")
    print()


def calculateProb(output, steer, accel):
    mean_steer = output[0]
    dev_steer = output[1] + 0.00000001
    mean_accel = output[2]
    dev_accel = output[3] + 0.00000001
    # torch.nan_to_num(output,nan=0.00001)

    
    steer_prob = 1/(dev_steer*(2*math.pi)**0.5) * torch.exp(-0.5*((steer-mean_steer)/dev_steer)**2)
    accel_prob = 1/(dev_accel*(2*math.pi)**0.5) * torch.exp(-0.5*((accel-mean_accel)/dev_accel)**2)
    
    return steer_prob*accel_prob


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )
