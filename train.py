

import argparse
import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net


parser = argparse.ArgumentParser(description='PyTorch implementation of CombinedHeight')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before checkpointing')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

data_path = os.path.join(os.path.expanduser('~'), '.torch', 'datasets', 'mnist')
train_data = datasets.MNIST(data_path, train=True, download=True,
                            transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))]))
test_data = datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))]))

train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size,
                         num_workers=4, pin_memory=True)


model = Net().to(device)
optimiser = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

if args.resume:
    model.load_state_dict(torch.load('model.pth'))
    optimiser.load_state_dict(torch.load('optimiser.pth'))

model.train()
train_losses = []

for i, (data, target) in enumerate(train_loader):
    data = data.to(device=device, non_blocking=True)
    target = target.to(device=device, non_blocking=True)
    optimiser.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    train_losses.append(loss.item())
    optimiser.step()

    if i % 10 == 0:
        print(i, loss.item())
        torch.save(model.state_dict(), 'model.pth')
        torch.save(optimiser.state_dict(), 'optimiser.pth')
        torch.save(train_losses, 'train_losses.pth')


