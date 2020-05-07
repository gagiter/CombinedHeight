
import torch.nn as nn
import segmentation_models_pytorch as smp


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.net = smp.Unet('resnet34', encoder_weights=None, activation='sigmoid')

    def forward(self, x):
        return self.net(x['color'])
