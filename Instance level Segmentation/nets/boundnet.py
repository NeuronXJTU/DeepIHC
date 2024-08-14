import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.resnet import ResNet
import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels//2,in_channels//2,(2,2),2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)




class FPN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # ----------------------------------#
        #   C3、C4、C5通道数均调整成256
        # ----------------------------------#
        self.lat_layers = nn.ModuleList(
            [
                nn.Conv2d(x, 256, kernel_size=1) for x in self.in_channels
            ]
        )

        # ----------------------------------#
        #   特征融合后用于进行特征整合
        # ----------------------------------#
        self.pred_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)) for _ in
                self.in_channels
            ]
        )

        # ----------------------------------#
        #   对P5进行下采样获得P6和P7
        # ----------------------------------#
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2),
                    nn.ReLU(inplace=True)
                )
            ]
        )

    def forward(self, backbone_features):
        P5 = self.lat_layers[2](backbone_features[2])
        P4 = self.lat_layers[1](backbone_features[1])
        P3 = self.lat_layers[0](backbone_features[0])

        P5_upsample = F.interpolate(P5, size=(backbone_features[1].size()[2], backbone_features[1].size()[3]),
                                    mode='nearest')
        P4 = P4 + P5_upsample

        P4_upsample = F.interpolate(P4, size=(backbone_features[0].size()[2], backbone_features[0].size()[3]),
                                    mode='nearest')
        P3 = P3 + P4_upsample

        P5 = self.pred_layers[2](P5)
        P4 = self.pred_layers[1](P4)
        P3 = self.pred_layers[0](P3)

        P6 = self.downsample_layers[0](P5)
        P7 = self.downsample_layers[1](P6)

        return P3, P4, P5, P6, P7



class Yolact(nn.Module):
    def __init__(self, num_classes, coef_dim=32, pretrained=False, train_mode=True):
        super().__init__()
        self.backbone = ResNet(layers=[3, 4, 6, 3])
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/resnet50_backbone_weights.pth"))
        self.fpn = FPN([512, 1024, 2048])
        self.n_channels = 64
        self.n_classes = 1
        # Question here
        in_channels = 64
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)

        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)

        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)

        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)

        self.outc = nn.Conv2d(in_channels, self.n_classes, kernel_size=(1, 1))

        self.last_activation = nn.Sigmoid()

        self.upsample=nn.Upsample(scale_factor=8, mode="nearest")

        self.x_1_conv=nn.Conv2d(256,64,3,padding=1)

        self.x_2_conv=nn.Conv2d(256,128,3,padding=1)

        self.x_3_conv=nn.Conv2d(256,256,3,padding=1)

        self.x_4_conv=nn.Conv2d(256,512,3,padding=1)

        self.x_5_conv=nn.Conv2d(256,512,3,padding=1)

    def forward(self, x):
        '''
        主干特征提取网络获得三个初步特征 (n, 512, 68, 68)
                                        (n, 1024, 34, 34)
                                        (n, 2048, 17, 17)
        '''
        features = self.backbone(x)
        '''
        构建特征金字塔，获得五个有效特征层 (n, 256, 68, 68) P3
                                          (n, 256, 34, 34) P4
                                          (n, 256, 17, 17) P5
                                          (n, 256, 9, 9)   P6
                                          (n, 256, 5, 5)   P7
        '''
        features = self.fpn.forward(features)
        x_1=self.x_1_conv(self.upsample(features[0]))
        x_2=self.x_2_conv(self.upsample(features[1]))
        x_3=self.x_3_conv(self.upsample(features[2]))
        x_4=self.x_4_conv(self.upsample(features[3]))
        x_5=self.x_5_conv(self.upsample(features[4]))

        x = self.up4(x_5, x_4)
        print(x.shape)

        x = self.up3(x, x_3)
        print(x.shape)

        x = self.up2(x, x_2)
        print(x.shape)

        x = self.up1(x, x_1)
        print(x.shape)

        boundpred= self.last_activation(self.outc(x))
        print(boundpred)
from PIL import Image
import numpy as np
import torch
model = Yolact(1, pretrained = False)
image_path="D:/ki67/40x/choose-20/512-1/img/slide-2022-06-13t09-25-27-r1-s4_88320_23760_0.png"
image = Image.open(image_path)
if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
    image=image
else:
    image = image.convert('RGB')
image = np.array(image, np.float32)
image=np.transpose(image, [2, 0, 1])
image=np.expand_dims(image, axis=0)
x=torch.tensor(image)
model.forward(x)
