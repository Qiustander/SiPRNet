import torch  # to load PyTorch library
import torch.nn as nn  # to load PyTorch library
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        in_dim = 1
        out_dim = 2
        in_size = 128
        dim = 32
        num_layer_res = 1

        self.in_layer = FeatExt(in_size=in_size)
        self.preupsample = nn.Sequential(
            ConvBlock(in_dim, dim, 3, 1, 1, isUseNorm=True),
            self._make_up_block(dim, kernel_size=3, num_layer=num_layer_res),
            nn.PixelShuffle(2)
            # ConvBlock(dim, dim * 4, 3, 1, 1, isUseNorm=True),
        )
        self.upsample1 = nn.Sequential(
                self._make_up_block2(dim, kernel_size=3, num_layer=num_layer_res),
                nn.PixelShuffle(2)
            )
        # self.upsample2 = self._make_up_block(dim, kernel_size=3, num_layer=num_layer_res)
        self.last = nn.Sequential(
                ConvBlock(dim//2, out_dim, 3, 1, 1, isUseNorm=False),
                nn.Conv2d(out_dim, out_dim, 3, 1, 1)
            )

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_up_block(self, in_channels, kernel_size, num_layer):
        upsample_channel = ConvBlock(in_channels, in_channels * 4, kernel_size=kernel_size, stride=1,
                                     padding=1, isUseNorm=True)
        layers = []
        for i in range(1, num_layer):
            layers.append(UpsampleBlock(in_channels, in_channels, kernel_size=kernel_size, upsample=None))
        layers.append(
            UpsampleBlock(in_channels, in_channels * 4, kernel_size=kernel_size, upsample=upsample_channel))
        return nn.Sequential(*layers)

    def _make_up_block2(self, in_channels, kernel_size, num_layer):
        upsample_channel = ConvBlock(in_channels, in_channels * 2, kernel_size=kernel_size, stride=1,
                                     padding=1, isUseNorm=True)
        layers = []
        for i in range(1, num_layer):
            layers.append(UpsampleBlock(in_channels, in_channels, kernel_size=kernel_size, upsample=None))
        layers.append(
            UpsampleBlock(in_channels, in_channels*2, kernel_size=kernel_size, upsample=upsample_channel))
        return nn.Sequential(*layers)

    def forward(self, in_img):  # "self" stores the variables (modules) that is defined in initialization function,
        m = self.in_layer(in_img)
        img = self.preupsample(m)
        img = self.upsample1(img)
        # img = self.upsample2(img)
        img = self.last(img)
        return img


class ConvBlock(nn.Module):  # contruct the ConvBlock module based on the structure of nn.Module
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, isUseNorm=False):  # define the initialization function.
        # "self" is the variable pool of the ConvBlock class
        # The function is conducted automatically when build the ConvBlock module,

        super(ConvBlock, self).__init__()  # call the initialization function of the father class ("nn.Module")
        self.isUseNorm = isUseNorm
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)  # define a convolutional layer
        if self.isUseNorm: self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.PReLU()  # define a activation layer

        return

    def forward(self, x):  # "self" stores the variables (modules) that is defined in initialization function,
        # x is the received input

        m = self.conv(x)  # call the "self.conv" module as defined in initialization function
        if self.isUseNorm: m = self.norm(m)
        out = self.act(m)  # call the "self.act" module as defined in initialization function

        return out  # output the result (store in "out")


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=None):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size, stride=1, padding=1, isUseNorm=True),
            nonlocalblock(channel=out_channels),
            ConvBlock(out_channels, out_channels, kernel_size, stride=1, padding=1, isUseNorm=True),
            nonlocalblock(channel=out_channels),
        )
        self.activation = nn.PReLU()
        self.upsample_chn = upsample

    def forward(self, x):

        if self.upsample_chn is not None:
            shortcut = self.upsample_chn(x)
        else:
            shortcut = x
        out = self.upsample(x)
        out += shortcut
        out = self.activation(out)

        return out


def upsample(x):
    b,c,h,w = x.shape[0:4]
    avg = nn.AvgPool2d([4,4],stride=4)
    output = avg(x).view(b,c,-1)
    return output


class nonlocalblock(nn.Module):
    def __init__(self,channel=32,avg_kernel=2):
        super(nonlocalblock,self).__init__()
        self.channel = channel//2
        self.theta = nn.Conv2d(channel,self.channel,1)
        self.phi = nn.Conv2d(channel,self.channel,1)
        self.g = nn.Conv2d(channel,self.channel,1)
        self.conv = nn.Conv2d(self.channel,channel,1)
        self.avg = nn.AvgPool2d([avg_kernel,avg_kernel],stride=avg_kernel)
    def forward(self,x):
        H,W = x.shape[2:4]
        u=self.avg(x)
        b,c,h,w = u.shape[0:4]
        theta_x = self.theta(u).view(b,self.channel,-1).permute(0,2,1)
        phi_x = self.phi(u)
        phi_x = upsample(phi_x)
        g_x = self.g(u)
        g_x = upsample(g_x).permute(0,2,1)
        theta_x = torch.matmul(theta_x,phi_x)
        theta_x = F.softmax(theta_x,dim=-1)

        y = torch.matmul(theta_x,g_x)
        y = y.permute(0,2,1)
        y = y.view(b,self.channel,h,w)
        y = self.conv(y)
        y = F.interpolate(y,size=[H,W])
        return y

class FeatExt(nn.Module):
    def __init__(self, in_size=64, basic_channels=1024):
        super(FeatExt, self).__init__()
        total_pixel = in_size * in_size
        self.extractor = nn.Sequential(
            ConvBlock(in_channels=total_pixel, out_channels=basic_channels, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2),
            ConvBlock(in_channels=basic_channels, out_channels=basic_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=basic_channels, out_channels=basic_channels, kernel_size=1, stride=1, padding=0),
        )
        return

    def forward(self, in_img):
        b, c, w, h = in_img.shape
        flat_data = in_img.reshape(b, c * w * h, 1, 1)
        feat = self.extractor(flat_data)
        return feat.reshape(b, c, 32, 32)
