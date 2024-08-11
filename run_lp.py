from pathlib import Path
import argparse
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from models.linear import LinearClassifier
from data_utils.cls_datasets import get_data_loaders
from data_utils.cls_datasets import LinearClsDataset
from models.vgg_1d import VGG16

parser = argparse.ArgumentParser(description='Lead-Fusion Barlow Twins Linear Probing')
parser.add_argument('--data-dir', type=str, required=True,
                    metavar='DIR', help='data path')
parser.add_argument('--checkpoint', type=Path, required=True,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--num-classes', type=int, required=True, metavar='N', help="the number of classes")
parser.add_argument('--feat-dir', default='./feat/', type=Path,
                    metavar='DIR', help='path to save features')
parser.add_argument('--num-leads', default=8, type=int, metavar='N', help="the number of leads")
parser.add_argument('--workers', default=6, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR',
                    help='learning rate')


class LinearProbing(object):
    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder_list = list()
        load_params = torch.load(args.checkpoint, map_location=self.device)
        lead_names = ["ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6"]
        for i in range(args.num_leads):
            encoder = VGG16(ch_in=1, alpha=0.125)
            if 'backbone_state_dict' in load_params:
                missing_keys, unexpected_keys = encoder.load_state_dict(
                    load_params['backbone_state_dict'][i].state_dict(), strict=False)
                assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
                print("lead-%s - successfully load weights." % lead_names[i])
            self.encoder_list.append(torch.nn.Sequential(*list(encoder.children())[:-1]).to(self.device))
        self.num_classes = args.num_classes
        self.num_leads = args.num_leads

    def infer_feature(self):
        train_loader, val_loader, test_loader = get_data_loaders(self.args.data_dir, self.args.batch_size,
                                                                 self.args.workers,
                                                                 False)
        self.args.feat_dir.mkdir(parents=True, exist_ok=True)
        train_feat_list = list()
        train_label_list = list()
        print("Infer train feature ......\n")

        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                feat_ld_list = list()
                for j in range(self.num_leads):
                    enc = self.encoder_list[j]
                    enc.eval()
                    x_ld = x[:, [j], :]
                    feat_ld = enc(x_ld).cpu().numpy()
                    feat_ld = np.squeeze(feat_ld)
                    feat_ld_list.append(feat_ld)
                feat_ld_cat = np.concatenate(feat_ld_list, axis=-1)
                lb = y.cpu().numpy()
                train_feat_list.append(feat_ld_cat)
                train_label_list.append(lb)

        train_feat = np.concatenate(train_feat_list, axis=0)
        train_label = np.concatenate(train_label_list, axis=0)
        del train_feat_list, train_label_list
        np.save(self.args.feat_dir / "X_train.npy", train_feat)
        np.save(self.args.feat_dir / "y_train.npy", train_label)
        print("\ntrain feat shape: ", train_feat.shape)
        print("train label shape: ", train_label.shape)
        del train_feat, train_label

        print("\nInfer val feature ......\n")
        val_feat_list = list()
        val_label_list = list()
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(val_loader)):
                x, y = x.to(self.device), y.to(self.device)
                feat_ld_list = list()
                for j in range(self.num_leads):
                    x_ld = x[:, [j], :]
                    enc = self.encoder_list[j]
                    enc.eval()
                    feat_ld = enc(x_ld).cpu().numpy()
                    feat_ld = np.squeeze(feat_ld)
                    feat_ld_list.append(feat_ld)
                feat_ld_cat = np.concatenate(feat_ld_list, axis=-1)
                lb = y.cpu().numpy()
                val_feat_list.append(feat_ld_cat)
                val_label_list.append(lb)
        val_feat = np.concatenate(val_feat_list, axis=0)
        val_label = np.concatenate(val_label_list, axis=0)
        del val_feat_list, val_label_list
        np.save(self.args.feat_dir / "X_val.npy", val_feat)
        np.save(self.args.feat_dir / "y_val.npy", val_label)
        print("\nval feat shape: ", val_feat.shape)
        print("val label shape: ", val_label.shape)
        del val_feat, val_label

        print("\nInfer test feature ......\n")
        test_feat_list = list()
        test_label_list = list()

        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(test_loader)):
                x, y = x.to(self.device), y.to(self.device)
                feat_ld_list = list()
                for j in range(self.num_leads):
                    x_ld = x[:, [j], :]
                    enc = self.encoder_list[j]
                    enc.eval()
                    feat_ld = enc(x_ld).cpu().numpy()
                    feat_ld = np.squeeze(feat_ld)
                    feat_ld_list.append(feat_ld)
                feat_ld_cat = np.concatenate(feat_ld_list, axis=-1)
                lb = y.cpu().numpy()
                test_feat_list.append(feat_ld_cat)
                test_label_list.append(lb)
        test_feat = np.concatenate(test_feat_list, axis=0)
        test_label = np.concatenate(test_label_list, axis=0)
        del test_feat_list, test_label_list
        np.save(self.args.feat_dir / "X_test.npy", test_feat)
        np.save(self.args.feat_dir / "y_test.npy", test_label)
        print("\ntest feat shape: ", test_feat.shape)
        print("test label shape: ", test_label.shape)
        del test_feat, test_label

        print("\nFeature inference done.")

    def classify(self):
        feat_path = self.args.feat_dir
        train_loader = data.DataLoader(LinearClsDataset(feat_path / "X_train.npy",
                                                        feat_path / "y_train.npy"),
                                       batch_size=128, shuffle=True)

        val_loader = data.DataLoader(LinearClsDataset(feat_path / "X_val.npy",
                                                      feat_path / "y_val.npy"),
                                     batch_size=128, shuffle=False)

        num_classes = self.num_classes
        classifier = LinearClassifier(feat_dim=512, num_classes=num_classes).to(self.device)
        epoch = 100
        opt = torch.optim.Adam(classifier.parameters(), lr=self.args.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        min_val_loss = 1e9

        print("Start training ......")
        for ep in range(epoch):
            print("\nEpoch %d ----------------------------" % (ep + 1))
            classifier.train()
            for x, y in tqdm(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                out = classifier(x)
                loss = loss_fn(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            classifier.eval()
            with torch.no_grad():
                val_loss = 0.0
                for xv, yv in val_loader:
                    xv, yv = xv.to(self.device), yv.to(self.device)
                    yv_ = classifier(xv)
                    _, pred = torch.max(yv_, dim=1)
                    loss = loss_fn(yv_, yv)
                    val_loss += loss.item()
                val_loss = val_loss / len(val_loader)
                print("\nVal_loss: %f." % val_loss)
                if val_loss < min_val_loss:
                    print("Save checkpoint with minimum val_loss (%f)." % val_loss)
                    min_val_loss = val_loss
                    torch.save(classifier.state_dict(), feat_path / "classifier_best_ckpt.pth")

        classifier.load_state_dict(torch.load(feat_path / "classifier_best_ckpt.pth"))
        classifier.eval()

        test_loader = data.DataLoader(
            LinearClsDataset(feat_path / "X_test.npy",
                             feat_path / "y_test.npy"),
            batch_size=128, shuffle=False)

        print("\nStart test ......")
        y_list_ = list()
        y_one_hot_list_ = list()
        with torch.no_grad():
            for x, y in tqdm(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                out = classifier(x)
                _, pred = torch.max(out, dim=1)
                out = F.softmax(out, dim=1)
                pred = pred.cpu().numpy()
                out = out.cpu().numpy()
                y_list_.append(pred)
                y_one_hot_list_.append(out)

        y_ = np.concatenate(y_list_, axis=0)
        y_one_hot_ = np.concatenate(y_one_hot_list_, axis=0)
        y = np.load(feat_path / "y_test.npy")
        y_one_hot = np.eye(num_classes)[y]
        auroc = roc_auc_score(y_one_hot, y_one_hot_, average="macro")
        auprc = average_precision_score(y_one_hot, y_one_hot_)
        conf_mat = confusion_matrix(y, y_)
        print("\nTest Performance -----------------------------------")
        print("AUROC = ", auroc)
        print("AUPRC = ", auprc)
        print("Confusion Matrix = \n", conf_mat)

    def run(self):
        self.infer_feature()
        self.classify()


def main():
    args = parser.parse_args()
    print("Linear Probing Setting ======================================")
    print(args)
    print("==========================================================")
    linear_probing = LinearProbing(args)
    linear_probing.run()


if __name__ == '__main__':
    main()
