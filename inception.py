import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, channels_in:int, channels_1x1:int, channels_3x3:list, channels_5x5:list, channels_pool:int):
        super(Inception).__init__()
        self.channels_1x1 = channels_1x1
        
        #Parallel Layers
        
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=channels_1x1, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=channels_3x3[0], kernel_size=1, stride=1),
            nn.Conv2d(in_channels=channels_3x3[0], out_channels=channels_3x3[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.conv_5x5_equivalent = nn.Sequential(
            nn.Conv2d(in_channels=channels_in, out_channels=channels_5x5[0], kernel_size=1, stride=1),
            nn.Conv2d(in_channels=channels_5x5[0], out_channels=channels_5x5[1], kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=channels_5x5[1], out_channels=channels_5x5[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=channels_in, out_channels=channels_pool, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        p1 = self.conv_1x1(x)
        p2 = self.conv_3x3(x)
        p3 = self.conv_5x5_equivalent(x)
        p4 = self.pool(x)
        
        predicted_y = torch.cat((p1, p2, p3, p4),1)
        return predicted_y