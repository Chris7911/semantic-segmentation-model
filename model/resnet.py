
# NOTE! OS: output stride, the ratio of input image resolution to final output resolution 
# (OS16: output size is (img_h/16, img_w/16)) (OS8: output size is (img_h/8, img_w/8))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def make_layer(block, in_channels, out_channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, out_channels=out_channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * out_channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels*BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels*BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    
    def forward(self, x):
        a = self.residual_function(x)
        b = self.shortcut(x)
        return nn.ReLU(inplace=True)(a+b)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, Bottleneck.expansion * out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(Bottleneck.expansion * out_channels)
        )
        self.shortcut = nn.Sequential()

        if (stride != 1) or (in_channels != Bottleneck.expansion * out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, Bottleneck.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * out_channels)
            )

    def forward(self, x):
        a = self.residual_function(x)
        b = self.shortcut(x)
        return nn.ReLU(inplace=True)(a+b)



class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_BasicBlock_OS8, self).__init__()
        if num_layers == 18:
            print ("pretrained resnet, 18")
            resnet = models.resnet18(pretrained=True)

            # remove fully connected layer, avg pool, layer4 and layer5:
            
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])
            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2
        elif num_layers == 34:
            print ("pretrained resnet, 34")

            resnet = models.resnet34(pretrained=True)

            # remove fully connected layer, avg pool, layer4 and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, out_channels=256,
                            num_blocks=num_blocks_layer_4, stride=1, dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, out_channels=512,
                            num_blocks=num_blocks_layer_5, stride=1, dilation=4)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c3 = self.resnet(x) # (shape: (batch_size, 128, h/8, w/8)) (it's called c3 since 8 == 2^3)

        output = self.layer4(c3) # (shape: (batch_size, 256, h/8, w/8))
        output = self.layer5(output) # (shape: (batch_size, 512, h/8, w/8))

        return output

class ResNet_BasicBlock_OS16(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_BasicBlock_OS16, self).__init__()
        if num_layers == 18:
            print ("pretrained resnet, 18")

            resnet = models.resnet18(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
            num_blocks = 2
        elif num_layers == 34:
            print ("pretrained resnet, 34")
            resnet = models.resnet34(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

            num_blocks = 3
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer5 = make_layer(BasicBlock, in_channels=256, out_channels=512, num_blocks=num_blocks, stride=1, dilation=2)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c4 = self.resnet(x) # (shape: (batch_size, 256, h/16, w/16)) (it's called c4 since 16 == 2^4)

        output = self.layer5(c4) # (shape: (batch_size, 512, h/16, w/16))

        return output

class ResNet_Bottleneck_OS16(nn.Module):
    def __init__(self, num_layers):
        super(ResNet_Bottleneck_OS16, self).__init__()
        if num_layers == 50:
            print ("pretrained resnet, 50")
            resnet = models.resnet50(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

        elif num_layers == 101:
            print ("pretrained resnet, 101")
            resnet = models.resnet101(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])

        elif num_layers == 152:
            print ("pretrained resnet, 152")
            resnet = models.resnet152(pretrained=True)
            # remove fully connected layer, avg pool and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-3])
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.layer5 = make_layer(Bottleneck, in_channels=4*256, out_channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c4 = self.resnet(x) # (shape: (batch_size, 4*256, h/16, w/16)) (it's called c4 since 16 == 2^4)

        output = self.layer5(c4) # (shape: (batch_size, 4*512, h/16, w/16))

        return output


def ResNet18_OS8():
    return ResNet_BasicBlock_OS8(num_layers=18)

def ResNet34_OS8():
    return ResNet_BasicBlock_OS8(num_layers=34)

def ResNet18_OS16():
    return ResNet_BasicBlock_OS16(num_layers=18)

def ResNet34_OS16():
    return ResNet_BasicBlock_OS16(num_layers=34)

def ResNet50_OS16():
    return ResNet_Bottleneck_OS16(num_layers=50)

def ResNet101_OS16():
    return ResNet_Bottleneck_OS16(num_layers=101)

def ResNet152_OS16():
    return ResNet_Bottleneck_OS16(num_layers=152)


def test_resnet():
    model = ResNet18_OS8().cuda()
    model = ResNet34_OS8().cuda()
    model = ResNet18_OS16().cuda()
    model = ResNet34_OS16().cuda()
    
    model = ResNet50_OS16().cuda()
    model = ResNet101_OS16().cuda()
    model = ResNet152_OS16().cuda()
    # print(model)
    x = torch.randn(1, 3, 512, 512).cuda()
    out = model(x)
    print(out.shape)

