import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, color:bool = True, bias:bool = True):
        super(MyModel, self).__init__()
        self.in_channels = 3 if color == True else 1
        self.bias = bias
        # layers 3 conv segments (7x7, 5x5 and 3x3) into 6 layers-deep mlp

        # 7x7 conv segment 1-shot
        self.conv_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 5x5 conv segment 1-shot
        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 3x3
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(16384, 1024, bias=self.bias),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=self.bias),
            nn.ReLU(),
            nn.Linear(512, 256, bias=self.bias),
            nn.ReLU(),
            nn.Linear(256, 128, bias=self.bias),
            nn.ReLU(),
            nn.Linear(128, 64, bias=self.bias),
            nn.ReLU(),
            nn.Linear(64, 1, bias=self.bias),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_7(x)
        x = self.conv_5(x)
        x = self.conv_3(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x



class MyModel2(nn.Module):
    def __init__(self, RGB:bool = True, bias:bool = True):
        super(MyModel2).__init__()
        self.in_channels = 3 if RGB == True else 1
        self.bias = bias
        # layers 3 conv segments (7x7, 5x5 and 3x3) into 6 layers-deep mlp
       
        # 7x7 conv segment
        self.conv_7_multi = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # 5x5 conv segment
        self.conv_5_multi = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 3x3 conv segment
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(32000, 1024, bias=self.bias),
            nn.ReLU(),
            nn.Linear(1024, 512, bias=self.bias),
            nn.ReLU(),
            nn.Linear(512, 256, bias=self.bias),
            nn.ReLU(),
            nn.Linear(256, 128, bias=self.bias),
            nn.ReLU(),
            nn.Linear(128, 64, bias=self.bias),
            nn.ReLU(),
            nn.Linear(64, 1, bias=self.bias),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_7(x)
        x = self.conv_5(x)
        x = self.conv_3(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return (1 if x >0.5 else 0)