
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.unet = smp.Unet('resnet34', encoder_weights='imagenet', activation='sigmoid')

    def forward(self, x):
        return self.unet(x)
