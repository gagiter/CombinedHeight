

import argparse
import os
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import dataset
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from model import Model
from criterion import Criterion


parser = argparse.ArgumentParser(description='PyTorch implementation of CombinedHeight')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--model_name', type=str, default='model',
                    help='model name for load, save and summary')
parser.add_argument('--dataset', type=str, default='Postdam',
                    help='where to load data')
parser.add_argument('--resume', type=int, default=1,
                    help='resume training from checkpoint')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu_id', type=int, default=0, metavar='N',
                    help='which gpu want to use')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--epoch_start', type=int, default=0, metavar='N',
                    help='number of start epoch')
parser.add_argument('--epoch_num', type=int, default=50000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--summary_freq', type=int, default=10, metavar='N',
                    help='how frequency to summary')
parser.add_argument('--save_freq', type=int, default=100, metavar='N',
                    help='how frequency to save')
parser.add_argument('--eval_freq', type=int, default=100, metavar='N',
                    help='how frequency to eval')
parser.add_argument('--image_size', type=int, default=512,
                    help='image size to train')


args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda:%d' % args.gpu_id if use_cuda else 'cpu')

train_data = dataset.Data(os.path.join('data', args.dataset), size=args.image_size, mode='train', device=device)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

val_data = dataset.Data(os.path.join('data', args.dataset), size=args.image_size, mode='val')
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

model = Model()
model = model.to(device)
criterion = Criterion()
optimiser = optim.Adam(model.parameters(), lr=args.lr)

date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
writer = SummaryWriter(os.path.join('runs', date_time + args.model_name))

load_dir = os.path.join('checkpoint', args.model_name)
if args.resume > 0 and os.path.exists(load_dir):
    model.load_state_dict(torch.load(os.path.join(load_dir, 'model.pth')))
    optimiser.load_state_dict(torch.load(os.path.join(load_dir, 'optimiser.pth')))
    args.epoch_start = torch.load(os.path.join(load_dir, 'epoch.pth'))['epoch']


for epoch in range(args.epoch_start, args.epoch_start + args.epoch_num):
    model.train()
    train_losses = []
    for data_in in train_loader:
        data_out = model(data_in)
        loss = criterion(data_in, data_out)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        train_losses.append(loss.item())

    if epoch % args.summary_freq == 0:
        loss = sum(train_losses) / len(train_losses)
        print(epoch, loss)
        writer.add_scalar('loss', loss, global_step=epoch)
        writer.add_image('image/color', color[0], global_step=epoch)
        writer.add_image('image/depth', depth[0], global_step=epoch)
        writer.add_image('image/predict', output[0], global_step=epoch)

    if epoch % args.eval_freq == 0:
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for data in val_loader:
                color = data[0].to(device=device)
                depth = data[1].to(device=device)
                output = model(color)
                loss = F.l1_loss(output, depth)
                eval_losses.append(loss.item())

        loss = sum(eval_losses) / len(eval_losses)
        print(epoch, loss, 'eval')
        writer.add_scalar('eval_loss', loss, global_step=epoch)
        writer.add_image('eval/color', color[0], global_step=epoch)
        writer.add_image('eval/depth', depth[0], global_step=epoch)
        writer.add_image('eval/predict', output[0], global_step=epoch)

    if epoch % args.save_freq == 0:
        save_dir = os.path.join('checkpoint', args.model_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        torch.save(optimiser.state_dict(), os.path.join(save_dir, 'optimiser.pth'))
        torch.save({'epoch': epoch}, os.path.join(save_dir, 'epoch.pth'))
        print('saved to ' + save_dir)


writer.close()
#
# def regularize(depth, label):
#




