from pathlib import Path
import argparse
import json
import time
from torch import nn, optim
import torch
import torchvision.transforms as transforms
from data_utils.data_folder import ECGDatasetFolder
from data_utils.multi_view_data_injector import MultiViewDataInjector
from data_utils.augmentations import RandomResizeCropTimeOut, ToTensor
from models.vgg_1d import VGG16

parser = argparse.ArgumentParser(description='LFBT Pretraining')
parser.add_argument('--data-dir', type=Path, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--num-leads', default=8, type=int, metavar='N', help="the number of leads")
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--gamma', default=0.8, type=float, metavar='L',
                    help='balance parameter of the loss')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='2048-2048-2048', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LFBT(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone_group = list()
        for i in range(args.num_leads):
            backbone = VGG16(ch_in=1, alpha=0.125)
            backbone.fc = nn.Identity()
            self.backbone_group.append(backbone.to(self.device))

        sizes = [64] + list(map(int, args.projector.split('-')))
        self.projector_group = list()
        for i in range(args.num_leads):
            layers = []
            for j in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[j], sizes[j + 1], bias=False))
                layers.append(nn.BatchNorm1d(sizes[j + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
            self.projector_group.append(nn.Sequential(*layers).to(self.device))
        self.bn_group = list()
        for i in range(args.num_leads):
            self.bn_group.append(nn.BatchNorm1d(sizes[-1], affine=False).to(self.device))

    def forward(self, y1, y2):
        z1_list = list()
        z2_list = list()
        for i in range(self.args.num_leads):
            z1_list.append(self.projector_group[i](self.backbone_group[i](y1[:, [i], :])))
            z2_list.append(self.projector_group[i](self.backbone_group[i](y2[:, [i], :])))
        loss_r = 0
        loss_t = 0
        for i in range(self.args.num_leads):
            for j in range(self.args.num_leads):
                c = self.bn_group[i](z1_list[i]).T @ self.bn_group[j](z2_list[j])
                c.div_(self.args.batch_size)
                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = off_diagonal(c).pow_(2).sum()
                ls = on_diag + self.args.lambd * off_diag
                if i == j:
                    loss_r += ls
                else:
                    loss_t += ls
        loss_r = loss_r / self.args.num_leads
        loss_t = loss_t / (self.args.num_leads * (self.args.num_leads - 1))
        loss = self.args.gamma * loss_r + (1 - self.args.gamma) * loss_t
        return loss, loss_r, loss_t


def main_worker(gpu, args):

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = LFBT(args)

    param_weights = []
    param_biases = []
    for md in model.backbone_group:
        for param in md.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

    for md in model.projector_group:
        for param in md.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

    for md in model.bn_group:
        for param in md.parameters():
            if param.ndim == 1:
                param_biases.append(param)
            else:
                param_weights.append(param)

    parameters = param_weights + param_biases
    optimizer = optim.Adam(parameters, lr=args.learning_rate)

    t = transforms.Compose([
        RandomResizeCropTimeOut(),
        ToTensor()
    ])
    dataset = ECGDatasetFolder(args.data_dir, transform=MultiViewDataInjector([t, t]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
        pin_memory=True)

    for epoch in range(0, args.epochs):
        total_loss = 0
        total_loss_r = 0
        total_loss_t = 0
        ep_start_time = time.time()
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            optimizer.zero_grad()
            loss, loss_r, loss_t = model.forward(y1, y2)
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                stats = dict(epoch=epoch, step=step,
                             loss=loss.item(),
                             loss_r=loss_r.item(),
                             loss_t=loss_t.item())
                print(json.dumps(stats))
            total_loss += loss.item()
            total_loss_r += loss_r.item()
            total_loss_t += loss_t.item()

        total_loss /= len(loader)
        total_loss_r /= len(loader)
        total_loss_t /= len(loader)
        ep_end_time = time.time()

        print("\nEpoch end. Time: %f - Average loss %f - loss_r %f - loss_t %f.\n" % (
            ep_end_time - ep_start_time, total_loss, total_loss_r, total_loss_t))

    torch.save({
        'backbone_state_dict': model.backbone_group,
    }, args.checkpoint_dir / 'encoder_group.pth')


def main():
    args = parser.parse_args()
    print("Pretraining Setting ======================================")
    print(args)
    print("==========================================================")
    main_worker(0, args)


if __name__ == '__main__':
    main()
