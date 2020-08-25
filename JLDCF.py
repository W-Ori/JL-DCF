import torch
from torch import nn
import torch.nn.functional as F

from .resnet import ResNet,Bottleneck

k = 64

class JLModule(nn.Module):
    def __init__(self,backbone):
        super(JLModule, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        cp = []
        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))

        cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))

        cp.append(nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))

        cp.append(nn.Sequential(nn.Conv2d(1024, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))

        cp.append(nn.Sequential(nn.Conv2d(2048, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))

        self.CP = nn.ModuleList(cp)


    def load_petrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        self.vgg_conv1.load_state_dict(torch.load('pretrained/vgg_conv1.pth'), strict=True)

    def forward(self,x):
        feature_extract = []
        feature_extract.append(self.CP[0](self.vgg_conv1(x)))
        x = self.backbone(x)
        for i in range(5):
            feature_extract.append(self.CP[i + 1](x[i]))
        return feature_extract



class CMLayer(nn.Module):
    def __init__(self):
        super(CMLayer, self).__init__()

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            part1 = list_x[i][0]
            part2 = list_x[i][1]
            sum = (part1 + part2 + (part1 * part2)).unsqueeze(dim=0)
            resl.append(sum)
        return resl



class FAModule(nn.Module):





class ScoreLayer(nn.Module):




class JL_DCF(nn.Module):




def build_model(base_model_cfg='resnet'):
