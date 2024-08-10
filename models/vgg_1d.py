import torch.nn as nn


def conv_layer(ch_in, ch_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv1d(ch_in, ch_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm1d(ch_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(int(in_list[i]), int(out_list[i]), k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool1d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self, ch_in=8, n_classes=1000, alpha=0.5):
        super(VGG16, self).__init__()
        self.alpha = alpha
        self.model = nn.Sequential(
            vgg_conv_block([ch_in, 64 * alpha], [64 * alpha, 64 * alpha], [3, 3], [1, 1], 2, 2),
            vgg_conv_block([64 * alpha, 128 * alpha], [128 * alpha, 128 * alpha], [3, 3], [1, 1], 2, 2),
            vgg_conv_block([128 * alpha, 256 * alpha, 256 * alpha], [256 * alpha, 256 * alpha, 256 * alpha], [3, 3, 3],
                           [1, 1, 1], 2, 2),
            vgg_conv_block([256 * alpha, 512 * alpha, 512 * alpha], [512 * alpha, 512 * alpha, 512 * alpha], [3, 3, 3],
                           [1, 1, 1], 2, 2),
            vgg_conv_block([512 * alpha, 512 * alpha, 512 * alpha], [512 * alpha, 512 * alpha, 512 * alpha], [3, 3, 3],
                           [1, 1, 1], 2, 2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(int(512 * alpha), n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(512 * self.alpha))
        x = self.fc(x)
        return x
