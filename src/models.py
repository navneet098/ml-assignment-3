import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden=(512,256), dropout=0.0):
        super().__init__()
        layers=[nn.Flatten(), nn.Linear(3072,hidden[0]), nn.ReLU()]
        if dropout: layers.append(nn.Dropout(dropout))
        for a,b in zip(hidden[:-1],hidden[1:]):
            layers += [nn.Linear(a,b),nn.ReLU()]
            if dropout: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden[-1],10))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

class CNN(nn.Module):
    """Convolutional baseline with global pooling."""
    def __init__(self, channels=(64,128,256), head=256):
        super().__init__()
        blocks=[]; c=3
        for i,o in enumerate(channels):
            blocks += [nn.Conv2d(c,o,3,padding=1,bias=False),
                       nn.BatchNorm2d(o),nn.ReLU(inplace=True)]
            if i<2: blocks.append(nn.MaxPool2d(2))
            c=o
        self.features=nn.Sequential(*blocks,nn.AdaptiveAvgPool2d(1))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(c,head),
                                      nn.ReLU(),nn.Linear(head,10))
    def forward(self,x): return self.classifier(self.features(x))

class ConvAutoencoder(nn.Module):
    """Only bottleneck_dim changes across the three controlled variants."""
    def __init__(self,bottleneck_dim):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),                 # 16x16
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),
            nn.MaxPool2d(2),                 # 8x8
            nn.Conv2d(64,64,3,padding=1),nn.ReLU()
        )
        self.to_z=nn.Linear(64*8*8,bottleneck_dim)
        self.from_z=nn.Linear(bottleneck_dim,64*8*8)
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(64,64,4,stride=2,padding=1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,stride=2,padding=1),nn.ReLU(),
            nn.Conv2d(32,3,3,padding=1),nn.Sigmoid()
        )
        self.bottleneck_dim=bottleneck_dim
    def encode(self,x): return self.to_z(self.encoder(x).flatten(1))
    def forward(self,x):
        z=self.encode(x)
        h=self.from_z(z).view(-1,64,8,8)
        return self.decoder(h)
