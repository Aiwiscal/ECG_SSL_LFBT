import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
