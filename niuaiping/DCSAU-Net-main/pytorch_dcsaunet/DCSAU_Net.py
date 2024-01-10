import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_dcsaunet.encoder import CSA

csa_block = CSA()

class Up(nn.Module):
    """Upscaling 向上缩放"""

    def __init__(self):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor缩放因子，指放大两倍

    def forward(self, x1, x2):
        x1 = self.up(x1)  #上采样，将其尺寸调整为与 x2 相同
        # input is CHW
        
        diffY = x2.size()[2] - x1.size()[2]  #计算了x2和调整后的x1的尺寸差异。diffY表示x2在垂直方向上的尺寸差异
        diffX = x2.size()[3] - x1.size()[3]

        #通过使用F.pad函数对x1进行填充，使其尺寸与 x2相同。填充操作是在 x1 的空间维度（高度和宽度）上进行的，通过计算差值的一半来确定填充的数量，以保证填充后的尺寸与x2相同
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

    
class PFC(nn.Module):
    def __init__(self,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.input_layer = nn.Sequential(
                    nn.Conv2d(3, channels, kernel_size = 3, padding=  3 // 2),
                    #nn.Conv2d(3, channels, kernel_size=3, padding= 1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, groups=channels, padding= kernel_size // 2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.pointwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x
    
class LKA(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(input_dim, dim, 5, padding=2, groups=1)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        # u = x.clone()        
        attl = self.conv0(x)
        attnl = self.conv_spatial(attl)
        attnl = self.conv1(attnl)

        attr = self.conv0(x)
        attnr = self.conv_spatial(attr)
        attnr = self.conv1(attnr)

        return attnl + attnr

class Attention(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(input_dim, d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
# inherit nn.module
class Model(nn.Module):
    def __init__(self,img_channels=3, n_classes=1):
       super(Model, self).__init__()
       self.pfc = PFC(64)
    #    self.lka = LKA(3, 64)
       self.atte = Attention(64, 64)
       self.img_channels = img_channels
       self.n_classes = n_classes
       self.maxpool = nn.MaxPool2d(kernel_size=2)
       self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)
       self.up_conv1 = Up()
       self.up_conv2 = Up()
       self.up_conv3 = Up()
       self.up_conv4 = Up()
       self.down1 = csa_block.layer1
       self.down2 = csa_block.layer2
       self.down3 = csa_block.layer3
       self.down4 = csa_block.layer4
       self.up1 = csa_block.layer5
       self.up2 = csa_block.layer6
       self.up3 = csa_block.layer7
       self.up4 = csa_block.layer8

       
    def forward(self, x):
  
        x1 = self.pfc(x)
        # x1 = self.lka(x)
        # print(x1.shape)      #[16, 64, 256, 256]
       
        
        x2 = self.maxpool(x1)
        # print(x2.shape)
        
        # x1_1 = self.atte(x2)
        # print(x1_1.shape)
        
        x3 = self.down1(x2)    
        x4 = self.maxpool(x3)
        
        x5 = self.down2(x4)     
        x6 = self.maxpool(x5)
        
        x7 = self.down3(x6)    
        x8 = self.maxpool(x7)
        
        x9 = self.down4(x8)

        x10 = self.up_conv1(x9,x7)

        x11 = self.up1(x10)
   
        x12 = self.up_conv2(x11,x5)    
        x13 = self.up2(x12)
       
        x14 = self.up_conv3(x13,x3)   
        x15 = self.up3(x14)
        
        x16 = self.up_conv4(x15,x1)
        x17 = self.up4(x16)
        
        x18 = self.out_conv(x17)
        
        # x19 = torch.sigmoid(x18)
        return x18
