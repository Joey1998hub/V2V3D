import torch.nn as nn
import torch
from torch.nn import functional as F
from utils import warp_feats
import numpy as np

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Upscale(nn.Module):
    def __init__(self,in_channels,out_channels,mode,scale=None):
        super().__init__()
        upscale = []
        if mode == 'subpixel':
            upscale.append(nn.PixelShuffle(scale))
            upscale.append(nn.Conv2d(in_channels=int(in_channels/(scale**2)),out_channels=out_channels,kernel_size=3,stride=1,padding=1))
            self.upscale = nn.Sequential(*upscale)

        if mode == 'upconv':
            upscale.append(nn.UpsamplingNearest2d(scale_factor=scale))
            upscale.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1))
            self.upscale = nn.Sequential(*upscale)
    def forward(self,x):
        x = self.upscale(x)
        return x


class Encoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        conv = []
        conv.append(nn.LeakyReLU(0.01))
        conv.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1))

        self.conv = nn.Sequential(*conv)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)

        return x

class Decoder(nn.Module):
    def __init__(self,in_channels,out_channels,scale):
        super().__init__()
        decoder = []
        decoder.append(nn.LeakyReLU(0.01))
        decoder.append(Upscale(in_channels=in_channels,out_channels=out_channels,scale=scale,mode='upconv'))
        self.decoder = nn.Sequential(*decoder)

    def forward(self,x):
        x = self.decoder(x)
        return x

class conv_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(conv_2d, self).__init__()
        pad = kernel_size // 2 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad)

    def forward(self, x):
        return self.conv(x) 

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  dilation,  downsample=None):
        super(conv_block, self).__init__()
        
        self.downsample = downsample
        block = nn.ModuleList()
        block.append(nn.Conv2d(in_channels, out_channels, 3,1,1, dilation=dilation))
        block.append(nn.LeakyReLU(0.01, True))
        block.append(nn.Conv2d(out_channels, out_channels, 3,1,1, dilation=dilation))
        self.conv = nn.Sequential(*block)

    def forward(self, x):
        x_skip = x
        if self.downsample is not None:
            x_skip = self.downsample(x)
        return self.conv(x) + x_skip

class Feature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Feature, self).__init__()
        self.relu = nn.LeakyReLU(0.01, True)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 4, 3, 1,1), nn.LeakyReLU(0.01, True))
        
        self.layer1 = self._make_layer(4, 4, 2, 1, 1)
        self.layer2 = self._make_layer(4, 8, 2, 1, 1)
        self.layer3 = self._make_layer(8, 16, 2, 1, 1)
       
        self.branch1 = nn.Sequential(nn.AvgPool2d(2,2), conv_2d(16, 4,1,1), nn.LeakyReLU(0.01, True), nn.UpsamplingBilinear2d(scale_factor=2))
        self.branch2 = nn.Sequential(nn.AvgPool2d(4,4), conv_2d(16, 4,1,1), nn.LeakyReLU(0.01, True), nn.UpsamplingBilinear2d(scale_factor=4))
        self.branch3 = nn.Sequential(nn.AvgPool2d(8,8), conv_2d(16, 4,1,1), nn.LeakyReLU(0.01, True), nn.UpsamplingBilinear2d(scale_factor=8))
       
        self.lastconv = nn.Sequential(conv_2d(28, 16, 3, 1), nn.LeakyReLU(0.01, True), nn.Conv2d(16,out_channels,1,1),nn.LeakyReLU(0.01, True))

    def _make_layer(self, in_c, out_c, blocks, stride, dilation):
        downsample = None
        if stride != 1 or in_c != out_c:
            downsample = conv_2d(in_c, out_c,1,1) 
        
        layers = []
        layers.append(conv_block(in_c, out_c, 3, stride, dilation, downsample))
        for _ in range(1, blocks):
            layers.append(conv_block(out_c, out_c, 3, stride, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        
        x = torch.cat([l3, self.branch1(l3), self.branch2(l3), self.branch3(l3)], 1)
        x = self.lastconv(x)
        return x
    
class Unet(nn.Module):
    def __init__(self,n_slices,input_channel):
        super().__init__()
        channels_interp = 128
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=8*channels_interp,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01), 
            nn.Conv2d(in_channels=8*channels_interp,out_channels=4*channels_interp,kernel_size=3,stride=1,padding=1), 
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=4*channels_interp,out_channels=2*channels_interp,kernel_size=3,stride=1,padding=1), 
            nn.LeakyReLU(0.01),
            nn.Conv2d(in_channels=2*channels_interp,out_channels=channels_interp,kernel_size=3,stride=2,padding=1), 
            nn.LeakyReLU(0.01))

        self.encoder_1 = Encoder(in_channels=channels_interp,out_channels=128)
        self.encoder_2 = Encoder(in_channels=128,out_channels=256)
        self.encoder_3 = Encoder(in_channels=256,out_channels=384)
        self.encoder_4 = Encoder(in_channels=384,out_channels=512)

        self.upscale_2 = Upscale(in_channels=512,out_channels=512,mode='upconv',scale=2)

        self.decoder_1 = Decoder(in_channels=896,out_channels=384,scale=2)
        self.decoder_2 = Decoder(in_channels=640,out_channels=256,scale=2)
        self.decoder_3 = Decoder(in_channels=384,out_channels=128,scale=2)
        self.decoder_4 = Decoder(in_channels=256,out_channels=128,scale=2)

        self.conv_final = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01), 
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01), 
            nn.Conv2d(in_channels=128,out_channels=n_slices,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.01))

    def forward(self,x):
        #upscale
        f = self.init_conv(x)
        
        #encoder
        encoder_layers = []
        encoder_layers.append(f)
        f = self.encoder_1(f)
        encoder_layers.append(f)
        f = self.encoder_2(f)
        encoder_layers.append(f)
        f = self.encoder_3(f)
        encoder_layers.append(f)
        f = self.encoder_4(f)

        #decoder
        f = F.leaky_relu(f,0.05)
        f = self.upscale_2(f)
        f = torch.cat((encoder_layers[3],f),dim=1)
        f = self.decoder_1(f)
        f = torch.cat((encoder_layers[2],f),dim=1)
        f = self.decoder_2(f)
        f = torch.cat((encoder_layers[1],f),dim=1)
        f = self.decoder_3(f)
        f = torch.cat((encoder_layers[0],f),dim=1)
        f = self.decoder_4(f)
        f = self.conv_final(f)

        return f
    
class V2V3D(nn.Module):
    def __init__(self,warp_psfs,n_slice,select_v,remain_v,use_views=13,feat_ch=4):
        super().__init__() 
        self.use_v = use_views
        self.feat_ch = feat_ch
        self.select_v = select_v
        self.remain_v = remain_v

        self.warp_psfs = warp_psfs
        self.feat_extract = Feature(in_channels=1,out_channels=feat_ch)
        self.unet1 = Unet(n_slices=n_slice,input_channel=feat_ch*select_v.shape[0]*n_slice)
        self.unet2 = Unet(n_slices=n_slice,input_channel=feat_ch*remain_v.shape[0]*n_slice)
        self.weight_init(mean=0.0, std=0.02)
    
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
    
    def forward(self,x):

        V,H,W = x.shape
        x = torch.unsqueeze(x,1) #v,c,h,w
        feats = self.feat_extract(x)
        feats = warp_feats(self.warp_psfs,feats)

        volume1 = self.unet1(feats[self.select_v,...].reshape(1,len(self.select_v)*feats.shape[1]*feats.shape[2],H,W))
        volume2 = self.unet2(feats[self.remain_v,...].reshape(1,len(self.remain_v)*feats.shape[1]*feats.shape[2],H,W))
        volume = (volume1+volume2)/2

        return volume1,volume2,volume