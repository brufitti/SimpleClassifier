import torch.nn as nn

from inception import Inception, NormalConv2d

# https://arxiv.org/pdf/1409.4842v1.pdf

class Classifier(nn.Module):
    def __init__(self, in_channels=3, classes=10):
        super(Classifier).__init__()
        
        # layers
        
        self.conv1       = NormalConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool1       = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2       = NormalConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv3       = NormalConv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.inception3a = Inception(in_channels=64 , channels_1x1=64 , channels_3x3=[96 ,128], channels_5x5=[16,32 ], channels_pool=32)
        self.inception3b = Inception(in_channels=256, channels_1x1=128, channels_3x3=[129,192], channels_5x5=[32,96 ], channels_pool=64)
        self.pool2       = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception4a = Inception(in_channels=480, channels_1x1=192, channels_3x3=[96 ,208], channels_5x5=[16,48 ], channels_pool=64)
        self.inception4b = Inception(in_channels=512, channels_1x1=160, channels_3x3=[112,224], channels_5x5=[24,64 ], channels_pool=64)
        self.inception4c = Inception(in_channels=512, channels_1x1=128, channels_3x3=[128,256], channels_5x5=[24,64 ], channels_pool=64)
        self.inception4d = Inception(in_channels=512, channels_1x1=112, channels_3x3=[144,288], channels_5x5=[32,64 ], channels_pool=64)
        self.inception4e = Inception(in_channels=528, channels_1x1=256, channels_3x3=[160,320], channels_5x5=[32,128], channels_pool=128)
        self.pool3       = nn.MaxPool2d(kernel_size=3, stride=2) 
        self.inception5a = Inception(in_channels=832, channels_1x1=256, channels_3x3=[160,320], channels_5x5=[32,128], channels_pool=128)
        self.inception5b = Inception(in_channels=832, channels_1x1=384, channels_3x3=[192,384], channels_5x5=[48,128], channels_pool=128)
        
        self.main_output = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(1024, classes),
            nn.Softmax(classes),
        )
        
        self.train_output1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=1024),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=classes),
            nn.Softmax(classes)
        )
        
        self.train_output2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(in_channels=528, out_channels=128, kernel_size=1, stride=1, bias=False),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=1024),
            nn.Dropout(0.7),
            nn.Linear(in_features=1024, out_features=classes),
            nn.Softmax(classes)
        )