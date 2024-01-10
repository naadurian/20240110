import torch
import torch.nn  as nn
import torch.nn.functional as F 
import resnet

class UP(nn.Module):
    def __init__(self) -> None:
        super(UP,self).__init__()
        self.up = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
    
    def forward(self,x1,x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1,[diffX//2,diffX - diffX//2,
                       diffY//2,diffY-diffY//2])
        
        x = torch.cat([x2,x1],dim=1)
        return x


class PFC(nn.Module):
    def __init__(self,channels,kernel_size=7) -> None:
        super(PFC,self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(3,channels,kernel_size=3,padding=3//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.dwise = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size = 7,groups=channels,padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels))
        self.pwise = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels)
        )
                
    def forward(self,x):
        x = self.input(x)
        
        residual = x
        x = self.dwise(x)
        x = x+residual
        x = self.pwise(x)
        return x
"""_summary_
class PFC(nn.Module):
    def __init__(self,channels, kernel_size=7):
        super(PFC, self).__init__()
        self.channels = channels
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
        
        # 
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size = 3, padding=  3 // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.batchn = nn.BatchNorm2d(self.channels)
    def forward(self, x):
        # x = self.input_layer(x)
        x = self.conv1(x)
        # x, range(-, +) --> (0, 255) 8bit 
        # 0000 0000
        # 1111 1111  
        # 1+2+4+8+16+32+64+128=255 
        # 2^8-1=255
        # x, range(-, +) --> (0, 255) 8bit 
        # 1 x, --> (0, 1) --> *255 (0, 255)
        
        x = self.relu1(x)
        x = self.batchn(x)
        
        
        residual = x
        x = self.depthwise(x)
        x += residual
        x = self.pointwise(x)
        return x
"""
class LKA(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(LKA,self).__init__()
        self.dwconv = nn.Conv2d(in_channels,out_channels,5,stride=1,padding=2,groups=1)
        self.dwdconv =nn.Conv2d(out_channels,out_channels,7,stride=1,padding=9,groups=out_channels,dilation=3)
        self.conv1 = nn.Conv2d(out_channels,out_channels,1)
    
    def forward(self,x):
        residual = x
        x = self.dwconv(x)
        x = self.dwdconv(x)
        x = self.conv1(x)
        
        x = x * residual
        
class Attention(nn.Module):
    def __init__(self,in_chennels,out_chennels) -> None:
        super(Attention,self).__init__()
        
        self.conv1 = nn.Conv2d(in_chennels,out_chennels,1)
        self.GELU = nn.GELU()
        self.LKA = LKA(out_chennels,out_chennels)
        self.conv2 = nn.Conv2d(out_chennels,out_chennels,1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.GELU(x)
        x = self.LKA(x)
        x = self.conv2(x)
        return x



 
ResNet = resnet.ResNet
Bottleneck = resnet.Bottleneck

def CSA(pretrained=False, root='~/.encoding/models', **kwargs):
    model = ResNet(Bottleneck, [2, 2, 2, 2],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
   
    return model
 
csa_block = CSA()    
class model(nn.Module):
    def __init__(self,in_channels=3,out_channels=1) -> None:
        super(model,self).__init__()
        self.conv3 = nn.Conv2d(in_channels,out_channels,3)
        self.PFC = PFC(64)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.csa_down1 = csa_block.layer1
        self.csa_down2 = csa_block.layer2
        self.csa_down3 = csa_block.layer3
        self.csa_down4 = csa_block.layer4
        self.csa_up1 = csa_block.layer5
        self.csa_up2 = csa_block.layer6
        self.csa_up3 = csa_block.layer7
        self.csa_up4 = csa_block.layer8
        self.up1 = UP()
        self.up2 = UP()
        self.up3 = UP()
        self.up4 = UP()
        self.out_conv = nn.Conv2d(64,out_channels,kernel_size=1,stride=1,padding=0)
    
    """
        # x.shape: batch_size, channels, height, width.
    # x1.shape: 
    # def feature_map_save(x1):
        # faeture_map_data = x1[1, :5, :, :]去掉第一个维度
        # for file in feature_map_data:
        #     cv.imwrite("./cengshu_i", file)
         
    """
    def forward(self,x):
        # x = self.conv3(x)
        # print("输出x的形状",x.shape)        #([16, 1, 254, 254])
        x = self.PFC(x)
        
        x_connation1 = x
        
        x = self.maxpool(x)
        
        x = self.csa_down1(x)
        x_connation2 = x
        x = self.maxpool(x)
        
        x = self.csa_down2(x)
        x_connation3 = x
        x = self.maxpool(x)
        
        x = self.csa_down3(x)
        x_connation4 = x
        x = self.maxpool(x)     #[16, 512, 16, 16
        x = self.csa_down4(x)
        # print("输出x的形状",x.shape)
        
        x = self.up1(x,x_connation4)
        
        
        x = self.csa_up1(x)
        
        x = self.up2(x,x_connation3)
        # x = x + x_connation3
        x = self.csa_up2(x)
        
        x = self.up3(x,x_connation2)
        # x = x + x_connation2
        x = self.csa_up3(x)
        
        x = self.up4(x,x_connation1)
        # x = x + x_connation1
        x = self.csa_up4(x)
        
        x = self.out_conv(x)
        
        return x        
                        