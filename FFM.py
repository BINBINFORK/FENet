import torch.nn as nn
import torch

from attention.sknet import SKConv

# class Conv2dBn(nn.Module):
#
#     def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
#         super(Conv2dBn, self).__init__()
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
#             nn.BatchNorm2d(out_ch)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

class GAUModule(nn.Module):
    def __init__(self, in_ch1,in_ch2, out_ch):  #
        super(GAUModule, self).__init__()

        # self.conv1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     Conv2dBn(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid()
        # )

        self.conv2 = Conv2dBnRelu(in_ch1, out_ch, kernel_size=3, stride=1, padding=1)  # 3*3卷积，尺寸不变
        self.skc = SKConv(in_ch2,out_ch,stride=1,M=2,r=16,L=32)
        self.conv3 = Conv2dBnRelu(in_ch2, out_ch, kernel_size=3, stride=1, padding=1)

    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        y_up = self.conv3(y_up)
        y_skc = self.skc(y)
        y_skc_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_skc)
        x = self.conv2(x)
        # y = self.conv1(y)
        # print(y_skc.shape)
        # print(x.shape)
        z = torch.mul(x, y_skc_up)  # x与y 对应元素相乘。

        return y_up + z

if __name__ == '__main__':
    x = torch.FloatTensor(3,256,32,32)
    y = torch.FloatTensor(3,512,16,16)
    b = GAUModule(256,512,128)
    out = b(x,y)
    print(out.shape)