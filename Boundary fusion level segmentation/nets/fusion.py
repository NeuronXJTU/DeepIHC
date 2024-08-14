import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def get_activation_b(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv_b(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm_b(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm_b(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm_b(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm_b, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation_b(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock_b(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock_b, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv_b(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class UpBlock_b(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock_b, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), 2)
        self.nConvs = _make_nConv_b(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)



class AttU_Net_UNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.last_activation_dist = nn.ReLU()


        # Question here
        in_channels = 64
        self.inc_b = ConvBatchNorm_b(img_ch, in_channels)
        self.down1_b = DownBlock_b(in_channels, in_channels * 2, nb_Conv=2)
        self.down2_b = DownBlock_b(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3_b = DownBlock_b(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4_b = DownBlock_b(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4_b = UpBlock_b(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3_b = UpBlock_b(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2_b = UpBlock_b(in_channels * 4, in_channels, nb_Conv=2)
        self.up1_b = UpBlock_b(in_channels * 2, in_channels, nb_Conv=2)
        self.outc_b = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.last_activation_bound = nn.Sigmoid()

    def forward(self, x_d, x_b):
        # encoding path
        # ===================dist============================
        x1_d = self.Conv1(x_d)

        x2_d = self.Maxpool(x1_d)
        x2_d = self.Conv2(x2_d)

        x3_d = self.Maxpool(x2_d)
        x3_d = self.Conv3(x3_d)

        x4_d = self.Maxpool(x3_d)
        x4_d = self.Conv4(x4_d)

        x5_d = self.Maxpool(x4_d)
        x5_d = self.Conv5(x5_d)

        # decoding + concat path
        d5_d = self.Up5(x5_d)
        x4_d = self.Att5(g=d5_d, x=x4_d)
        d5_d = torch.cat((x4_d, d5_d), dim=1)
        d5_d = self.Up_conv5(d5_d)

        d4_d = self.Up4(d5_d)
        x3_d = self.Att4(g=d4_d, x=x3_d)
        d4_d = torch.cat((x3_d, d4_d), dim=1)
        d4_d = self.Up_conv4(d4_d)

        d3_d = self.Up3(d4_d)
        x2_d = self.Att3(g=d3_d, x=x2_d)
        d3_d = torch.cat((x2_d, d3_d), dim=1)
        d3_d = self.Up_conv3(d3_d)

        d2_d = self.Up2(d3_d)
        x1_d = self.Att2(g=d2_d, x=x1_d)
        d2_d = torch.cat((x1_d, d2_d), dim=1)
        d2_d = self.Up_conv2(d2_d)

        d1_d = self.Conv_1x1(d2_d)
        d1_d = self.last_activation_dist(d1_d)

        # ====================bound=====================
        x_b = x_b.float()

        x_b_1 = self.inc_b(x_b)

        x_b_2 = self.down1_b(x_b_1)

        x_b_3 = self.down2_b(x_b_2)

        x_b_4 = self.down3_b(x_b_3)

        x_b_5 = self.down4_b(x_b_4)

        x_b = self.up4_b(x_b_5, x_b_4)

        x_b = self.up3_b(x_b, x_b_3)

        x_b = self.up2_b(x_b, x_b_2)

        x_b = self.up1_b(x_b, x_b_1)


        d1_b = self.last_activation_bound(self.outc_b(x_b))


        return d1_d, d1_b

