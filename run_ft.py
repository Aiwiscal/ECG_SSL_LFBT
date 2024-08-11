import time
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from data_utils.cls_datasets import get_data_loaders
from models.mbn import MultiBranchNet
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description='Lead-Fusion Barlow Twins Fine Tuning')
parser.add_argument('--data-dir', type=str, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--checkpoint', type=Path, required=True,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--num-classes', type=int, required=True, metavar='N', help="the number of classes")
parser.add_argument('--fraction', default=1.0, type=float, metavar='L',
                    help='The fraction of training set used for training.')
parser.add_argument('--model-dir', default='./', type=Path,
                    metavar='DIR', help='path to save models')
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.0001, type=float, metavar='LR',
                    help='learning rate')


class FineTuning(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = MultiBranchNet(args.num_classes, checkpoint=args.checkpoint).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)

    def train(self, data_loader, optimizer):
        self.model.train()
        for batch_idx, (data, target) in tqdm(enumerate(data_loader)):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            optimizer.step()

    def validate(self, data_loader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()  # sum up batch loss
        val_loss /= len(data_loader)
        print('\nVal loss: {:.4f}\n'.format(val_loss))
        return val_loss

    def test(self, data_loader):
        self.model.eval()
        y_pred_list = list()
        y_true_list = list()
        y_pred_s_list = list()
        with torch.no_grad():
            for data, target in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output_softmax = nn.functional.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                y_true_list.append(target.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())
                y_pred_s_list.append(output_softmax.cpu().numpy())
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        y_pred_s = np.concatenate(y_pred_s_list, axis=0)
        y_true_s = np.eye(len(np.unique(y_true)))[y_true]
        auroc = roc_auc_score(y_true_s, y_pred_s, average="macro")
        auprc = average_precision_score(y_true_s, y_pred_s)
        conf_mat = confusion_matrix(y_true, y_pred)
        print("Test Performance -----------------------------------------")
        print("AUROC: ", auroc)
        print("AUPRC: ", auprc)
        print("Confusion Matrix: \n", conf_mat)

    def run(self):
        train_loader, val_loader, test_loader = get_data_loaders(self.args.data_dir,
                                                                 self.args.batch_size,
                                                                 self.args.workers,
                                                                 train_ratio=self.args.fraction)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        min_loss = 999999999
        for ep in range(1, self.args.epochs + 1):
            print("\nEpoch %d ----------------------------" % ep)
            self.train(train_loader, opt)
            v_loss = self.validate(val_loader)
            if v_loss < min_loss:
                min_loss = v_loss
                print("Save checkpoint with minimum val_loss (%f)." % v_loss)
                torch.save(self.model, self.args.model_dir / "ft_best_ckpt.pth")
        self.model.load_state_dict(torch.load(self.args.model_dir / "ft_best_ckpt.pth").state_dict())
        self.test(test_loader)


def main():
    args = parser.parse_args()
    print("Fine Tuning Setting ======================================")
    print(args)
    print("==========================================================")
    fine_tuning = FineTuning(args)
    fine_tuning.run()


if __name__ == '__main__':
    main()
