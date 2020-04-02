

import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net
import dataset
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch implementation of CombinedHeight')

parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--data_dir', type=str, default='data/Postdam',
                    help='where to load data')
parser.add_argument('--resume', action='store_true', default=True,
                    help='resume training from checkpoint')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu_id', type=int, default=1, metavar='N',
                    help='which gpu want to use')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--epoch_start', type=int, default=0, metavar='N',
                    help='number of start epoch')
parser.add_argument('--epoch_num', type=int, default=5000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--summary_freq', type=int, default=1, metavar='N',
                    help='how frequency to summary')

args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda:%d' % args.gpu_id if use_cuda else 'cpu')

train_data = dataset.Data(args.data_dir, 'train')
val_data = dataset.Data(args.data_dir, 'val')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size)


model = smp.Unet('resnet34', encoder_weights=None, activation='sigmoid')
model = model.to(device)
optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
writer = SummaryWriter()

if args.resume:
    model.load_state_dict(torch.load('checkpoint/model.pth'))
    optimiser.load_state_dict(torch.load('checkpoint/optimiser.pth'))
    args.epoch_start = torch.load('checkpoint/epoch.pth')['epoch']

for epoch in range(args.epoch_start, args.epoch_start + args.epoch_num):
    model.train()
    train_losses = []
    for data in train_loader:
        color = data[0].to(device=device)
        label = data[1].to(device=device)
        depth = data[2].to(device=device)
        optimiser.zero_grad()
        output = model(color)
        # plannar, overlap = regularize(output, label)
        loss = F.l1_loss(output, depth)
        loss.backward()
        train_losses.append(loss.item())
        optimiser.step()

    if epoch % args.summary_freq == 0:
        loss = sum(train_losses) / len(train_losses)
        print(epoch, loss)
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_image('image/color', color[0], global_step=epoch)
        writer.add_image('image/label', label[0], global_step=epoch)
        writer.add_image('image/depth', depth[0], global_step=epoch)
        writer.add_image('image/predict', output[0], global_step=epoch)

        torch.save(model.state_dict(), 'checkpoint/model.pth')
        torch.save(optimiser.state_dict(), 'checkpoint/optimiser.pth')
        torch.save({'epoch': epoch}, 'checkpoint/epoch.pth')
        # torch.save(train_losses, 'train_losses.pth')

writer.close()
#
# def regularize(depth, label):
#



