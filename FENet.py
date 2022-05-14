import torch
import torch.nn as nn
from models.lib.resnet_for_psp import resnet50,resnet18,resnet34,resnet101
import torch.nn.functional as F
from attention.senet import SEBlock
# from attention.cbam import ChannelGate
from attention.sknet import SKNet
from attention.sknet import SKBlock
from attention.fcanet import FcaLayer
from attention.fcanet import FcaLayer
from attention.ocr import _ObjectAttentionBlock
from attention.self_attn import Self_Attn
from attention.danet import PAM_Module
from models.dfn import RRB
from models.lib.gausk import GAUModule
# from attention.cbam import ChannelGate
from models.lib.channel8 import ChannelGate
from attention.self_attn import Self_Attn
class ABCNet(nn.Module):
    def __init__(self, layers=50 ,classes=3,  pretrained = False):
        super(ABCNet, self).__init__()
        assert layers in [50,101,152]
        assert classes > 1
        resnet = resnet18(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.classifier = nn.Sequential(nn.Conv2d(64,classes,1,1))
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        # self.se1 = Self_Attn(64)
        # self.SK2 = SKBlock(256,128)
        # self.df1 = RRB(256,128)
        # self.df2 = RRB(128,64)
        self.gaus1 = GAUModule(256,512,256)
        self.gaus2 = GAUModule(128,256,128)
        self.gaus3 = GAUModule(64,128,64)
        self.cbr1 = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 256, 3, 1, 1),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU()
                                 )
        self.cbr2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 128, 3, 1, 1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU()
                                  )
        self.cbr3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, 1, 1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU()
                                  )
        self.c1 = ChannelGate(256)
        self.c2 = ChannelGate(128)
        # self.se3 = Self_Attn(64)
        # self.conv2=nn.Conv2d(128,128,1)
        self.se1 = Self_Attn(512)
        # self.conv1 = nn.Conv2d(512,1,1)
    def forward(self,x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        # print(layer3.shape)
        # df1 = self.df1(layer3)
        # df2 = self.df2(layer2)
        # SK2 = self.SK2(layer3)
        # se1= self.se1(layer2)
        # print(layer4.shape)
        # print(df1.shape)
        # print(layer4.shape)
        # conv1 = self.conv1(layer4)
        # print(conv1.shape)
        se1 = self.se1(layer4)
        c1 = self.c1(layer3)

        # print(c1.shape)

        # conv2 = self.conv2(layer2)
        # print(conv2.shape)
        # print(layer2.shape)
        c2 = self.c2(layer2)

        # print(c1.shape)

        # se3 = self.se3(layer1)
        # print(c1.shape)
        # print(se1.shape)
        gaus1 = self.gaus1(c1,se1)
        cbr1 = self.cbr1(gaus1)

        gaus2 =self.gaus2(c2,cbr1)
        cbr2 = self.cbr2(gaus2)
        gaus3 =self.gaus3(layer1,cbr2)
        cbr3 = self.cbr3(gaus3)



        # se1 = self.se1(gaus3)



        # up1 = self.up(df1)
        # cat1 = torch.cat([up1,SK2],1)

        # up2 = self.up(cat1)

        # print(up2.shape)

        # cat2= torch.cat([up2,layer2],1)

        # up3 = self.up(cat2)
        # print(up3.shape)
        # se1 = self.up(up3)    # 960,64,64

        up = F.interpolate(cbr3,size=256,mode='bilinear',align_corners=True)    #960,256,256

        classifier = self.classifier(up)
        return classifier

if __name__ == '__main__':
    from etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number

    model = ABCNet(classes=2)
    batch = torch.FloatTensor(3, 3, 256, 256)

    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(batch)

    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))


