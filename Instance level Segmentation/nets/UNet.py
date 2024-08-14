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



class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)

        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)#64-->128

        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)

        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)

        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)

        self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)

        self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)

        self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)

        self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))

        self.last_activation = nn.Sigmoid()

    def forward(self, x1):
        # Question here
        x = x1.float()
        print(x.shape)
        x_1 = self.inc(x)
        print(x_1.shape)

        x_2 = self.down1(x_1)
        print(x_2.shape)
        x_3 = self.down2(x_2)
        print(x_3.shape)

        x_4 = self.down3(x_3)
        print(x_4.shape)

        x_5 = self.down4(x_4)
        print(x_5.shape)

        x = self.up4(x_5, x_4)
        print(x.shape)

        x = self.up3(x, x_3)
        print(x.shape)

        x = self.up2(x, x_2)
        print(x.shape)

        x = self.up1(x, x_1)
        print(x.shape)

        logits=self.last_activation(self.outc(x))
        

        return logits


from PIL import Image
import numpy as np
import torch
model = UNet()
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