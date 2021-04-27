import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

sys.path.append("./model")
from resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from aspp import ASPP, ASPP_Bottleneck

def double_conv(in_channels, out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DeeplabV3(nn.Module):
    def __init__(self, model_id, project_dir, num_classes, mode = 0):
        super(DeeplabV3, self).__init__()
        self.num_classes = num_classes

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_dirs()

        # NOTE! specify the type of ResNet here
        # self.resnet = ResNet34_OS16()
        if mode == 0:
            self.resnet = ResNet18_OS16()
        elif mode == 1:
            self.resnet = ResNet34_OS16()
        elif mode == 2:
            self.resnet = ResNet18_OS8()
        elif mode == 3:
            self.resnet = ResNet34_OS8()
        elif mode == 4:
            self.resnet = ResNet50_OS16()
        elif mode == 5:
            self.resnet = ResNet101_OS16()
        elif mode == 6:
            self.resnet = ResNet152_OS16()
        
        
        if mode in [0, 1, 2, 3]:
            self.aspp = ASPP(num_classes=self.num_classes)
        else:
            self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)
        # (shape: (batch_size, 512, h/16, w/16)) 
        # (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. 
        # If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). 
        # If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)(output)
        # output = F.upsample(output, size=(h, w), mode="bilinear") 
        # (shape: (batch_size, num_classes, h, w))

        return output

    def create_dirs(self):
        self.logs_dir = self.project_dir + "training_logs"
        self.model_dir = self.logs_dir + "/{}".format(self.model_id)
        self.checkpoints_dir = self.model_dir + "/checkpoints"

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

class Resnet34Unet(nn.Module):
    def __init__(self, model_id, project_dir, num_classes):
        super(Resnet34Unet, self).__init__()
        self.model_id = model_id
        self.project_dir = project_dir
        self.create_dirs()

        resnet34 = models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(*list(resnet34.children())[:3])
        self.pool1 = resnet34.maxpool
        self.conv2_down = nn.Sequential(*list(resnet34.children())[4])
        self.conv3_down = nn.Sequential(*list(resnet34.children())[5])
        self.conv4_down = nn.Sequential(*list(resnet34.children())[6])
        self.conv5_down = nn.Sequential(*list(resnet34.children())[7])
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_up = double_conv(256 + 512, 256)
        self.conv2_up = double_conv(128 + 256, 128)
        self.conv3_up = double_conv(64 + 128, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        temp = self.pool1(out_conv1)
        out_conv2 = self.conv2_down(temp)
        out_conv3 = self.conv3_down(out_conv2)
        out_conv4 = self.conv4_down(out_conv3)
        bottle = self.conv5_down(out_conv4)

        up_x = self.upsample(bottle)
        up_x = torch.cat([up_x, out_conv4], dim=1)
        up_x = self.conv1_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv3], dim=1)
        up_x = self.conv2_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv2], dim=1)
        up_x = self.conv3_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv1], dim=1)
        out = self.dconv_last(up_x)

        return out

    def create_dirs(self):
        self.logs_dir = self.project_dir + "training_logs"
        self.model_dir = self.logs_dir + "/model_{}".format(self.model_id)
        self.checkpoints_dir = self.model_dir + "/checkpoints"

        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

if __name__ == "__main__":



    model = DeeplabV3("1", "./", 66).cuda()
    x = torch.randn(5, 3, 512, 512).cuda()
    out = model(x)
    print(out.shape)


