import torch
import torch.nn as nn

# https://arxiv.org/pdf/1409.4842v1.pdf

class Inception(nn.Module):
    def __init__(self, channels_in:int, channels_1x1:int, channels_3x3:list, channels_5x5:list, channels_pool:int):
        super(Inception).__init__()
        self.channels_1x1 = channels_1x1
        
        # Parallel Layers
        # Branch A
        self.conv_1x1a = NormalConv2d(in_channels=channels_in, out_channels=channels_1x1, kernel_size=1, stride=1)
        
        # Branch B
        self.conv_1x1b = NormalConv2d(in_channels=channels_in, out_channels=channels_3x3[0], kernel_size=1, stride=1)
        self.conv_3x3b = NormalConv2d(in_channels=channels_3x3[0], out_channels=channels_3x3[1], kernel_size=3, stride=1, padding=1)
        
        # Branch C
        self.conv_1x1c = NormalConv2d(in_channels=channels_in, out_channels=channels_5x5[0], kernel_size=1, stride=1)
        self.conv_5x5c = NormalConv2d(in_channels=channels_5x5[0], out_channels=channels_5x5[1], kernel_size=3, stride=1, padding=2)
        
        # Branch D
        self.pool_3x3d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_1x1d = NormalConv2d(in_channels=channels_in, out_channels=channels_pool, kernel_size=1, stride=1)
        
    def forward(self, x):
        p1 = nn.ReLU(self.conv_1x1a(x))
        p2 = nn.ReLU(self.conv_1x1b(x))
        p2 = nn.ReLU(self.conv_3x3b(p2))
        p3 = nn.ReLU(self.conv_1x1c(x))
        p3 = nn.ReLU(self.conv_5x5c(p3))
        p4 = self.pool_3x3d(x)
        
        p4 = nn.ReLU(self.conv_1x1d(p4))
        
        predicted_y = torch.cat((p1, p2, p3, p4),1)
        return predicted_y


class NormalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(NormalConv2d).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(num_features=out_channels, eps=0.0001)
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.relu(y)
        return y