
import torch.nn as nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self):
        pass

    def forward(self, data_in, data_out):
        loss = 0.0
        loss += F.l1_loss(data_in['depth'], data_out['depth'])




