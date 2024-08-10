import torch
import torch.nn as nn
# from models.net_base_network import EncoderNet
from models.vgg_1d import VGG16


class MultiBranchNet(nn.Module):
    def __init__(self, num_classes, num_leads=8, checkpoint=None):
        super(MultiBranchNet, self).__init__()
        load_params = dict()
        if checkpoint is not None:
            load_params = torch.load(checkpoint)
        encoder_ii = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_ii.load_state_dict(
                load_params["backbone_state_dict"][0].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-ii - successfully load weights.")
        self.encoder_ii = torch.nn.Sequential(*list(encoder_ii.children())[:-1])
        del encoder_ii

        encoder_iii = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_iii.load_state_dict(
                load_params["backbone_state_dict"][1].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-iii - successfully load weights.")
        self.encoder_iii = torch.nn.Sequential(*list(encoder_iii.children())[:-1])
        del encoder_iii

        encoder_v1 = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_v1.load_state_dict(
                load_params["backbone_state_dict"][2].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-v1 - successfully load weights.")
        self.encoder_v1 = torch.nn.Sequential(*list(encoder_v1.children())[:-1])
        del encoder_v1

        encoder_v2 = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_v2.load_state_dict(
                load_params["backbone_state_dict"][3].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-v2 - successfully load weights.")
        self.encoder_v2 = torch.nn.Sequential(*list(encoder_v2.children())[:-1])
        del encoder_v2

        encoder_v3 = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_v3.load_state_dict(
                load_params["backbone_state_dict"][4].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-v3 - successfully load weights.")
        self.encoder_v3 = torch.nn.Sequential(*list(encoder_v3.children())[:-1])
        del encoder_v3

        encoder_v4 = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_v4.load_state_dict(
                load_params["backbone_state_dict"][5].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-v4 - successfully load weights.")
        self.encoder_v4 = torch.nn.Sequential(*list(encoder_v4.children())[:-1])
        del encoder_v4

        encoder_v5 = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_v5.load_state_dict(
                load_params["backbone_state_dict"][6].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-v5 - successfully load weights.")
        self.encoder_v5 = torch.nn.Sequential(*list(encoder_v5.children())[:-1])
        del encoder_v5

        encoder_v6 = VGG16(ch_in=1, alpha=0.125)
        if "backbone_state_dict" in load_params:
            missing_keys, unexpected_keys = encoder_v6.load_state_dict(
                load_params["backbone_state_dict"][7].state_dict(), strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []
            print("lead-v6 - successfully load weights.")
        self.encoder_v6 = torch.nn.Sequential(*list(encoder_v6.children())[:-1])
        del encoder_v6

        self.num_classes = num_classes
        self.num_leads = num_leads
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x_ii = x[:, [0], :]
        feat_ii = self.encoder_ii(x_ii)
        x_iii = x[:, [1], :]
        feat_iii = self.encoder_iii(x_iii)
        x_v1 = x[:, [2], :]
        feat_v1 = self.encoder_v1(x_v1)
        x_v2 = x[:, [3], :]
        feat_v2 = self.encoder_v2(x_v2)
        x_v3 = x[:, [4], :]
        feat_v3 = self.encoder_v3(x_v3)
        x_v4 = x[:, [5], :]
        feat_v4 = self.encoder_v4(x_v4)
        x_v5 = x[:, [6], :]
        feat_v5 = self.encoder_v5(x_v5)
        x_v6 = x[:, [7], :]
        feat_v6 = self.encoder_v6(x_v6)
        feat = torch.concat([feat_ii, feat_iii, feat_v1, feat_v2, feat_v3, feat_v4, feat_v5, feat_v6], dim=1)
        feat_ = feat.view(feat.shape[0], feat.shape[1])
        y = self.fc(feat_)
        return y  # , feat

