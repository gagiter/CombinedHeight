

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
import regular


parser = argparse.ArgumentParser(description='PyTorch implementation of CombinedHeight')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dataset', type=str, default='Postdam',
                    help='where to load data')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu_id', type=int, default=0, metavar='N',
                    help='which gpu want to use')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--epoch_start', type=int, default=0, metavar='N',
                    help='number of start epoch')
parser.add_argument('--epoch_num', type=int, default=50000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--summary_freq', type=int, default=1, metavar='N',
                    help='how frequency to summary')
parser.add_argument('--image_size', type=int, default=512,
                    help='image size to train')

args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda:%d' % args.gpu_id if use_cuda else 'cpu')

train_data = dataset.Data(os.path.join('data', args.dataset), size=args.image_size, mode='train')
val_data = dataset.Data(os.path.join('data', args.dataset), 'val')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=args.batch_size)


model = smp.Unet('resnet34', encoder_weights=None, activation='sigmoid')
model = model.to(device)
optimiser = optim.Adam(model.parameters(), lr=args.lr)
writer = SummaryWriter()

if args.resume:
    model.load_state_dict(torch.load(os.path.join('checkpoint', args.dataset, 'model.pth')))
    optimiser.load_state_dict(torch.load(os.path.join('checkpoint', args.dataset, 'optimiser.pth')))
    args.epoch_start = torch.load(os.path.join('checkpoint', args.dataset, 'epoch.pth'))['epoch']

for epoch in range(args.epoch_start, args.epoch_start + args.epoch_num):
    model.train()
    train_losses = []
    for data in train_loader:
        color = data[0].to(device=device)
        label = data[1].to(device=device)
        depth = data[2].to(device=device)
        optimiser.zero_grad()
        output = model(color)
        normal, dist = regular.plane(output)
        overlap = regular.overlap(output, label)
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
        writer.add_image('image/normal', normal[0], global_step=epoch)
        writer.add_image('image/overlap', overlap[0], global_step=epoch)
        writer.add_image('image/dist', dist[0], global_step=epoch)

        # torch.save(model.state_dict(), os.path.join('checkpoint', args.dataset, 'model.pth'))
        # torch.save(optimiser.state_dict(), os.path.join('checkpoint', args.dataset, 'optimiser.pth'))
        # torch.save({'epoch': epoch}, os.path.join('checkpoint', args.dataset, 'epoch.pth'))
        # torch.save(train_losses, 'train_losses.pth')

writer.close()
#
# def regularize(depth, label):
#




