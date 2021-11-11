import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self,  num_classes= 10, repeats:list=[3,4,6,3]):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        base_dim = 64
        self.layer2 = ResidualBlock(base_dim, base_dim, base_dim*4, repeats[0], starting=True)
        self.layer3 = ResidualBlock(base_dim*4, base_dim*2, base_dim*8, repeats[1])
        self.layer4 = ResidualBlock(base_dim*8, base_dim*4, base_dim*16, repeats[2])
        self.layer5 = ResidualBlock(base_dim*16, base_dim*8, base_dim*32, repeats[3])
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
        super(Bottleneck, self).__init__()

        if starting:
            down = False
        self.block = bottleneck_block(in_dim, mid_dim, out_dim, down=down)
        self.relu = nn.ReLU(inplace=True)

        if down:
            self.conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0), # size 줄어듬
        
        else:
            self.conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0), # size 줄어들지 않음
        
        self.changedim = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x0 = self.conn_layer(x)
        identity = self.changedim(x0)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def bottleneck_block(in_dim, mid_dim, out_dim, down=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
    else:
        layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
    layers.extend([
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(mid_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_dim),
    ])
    return nn.Sequential(*layers)

def ResidualBlock(in_dim, mid_dim, out_dim, repeats, starting=False):
        layers = []
        layers.append(Bottleneck(in_dim, mid_dim, out_dim, down=True, starting=starting))
        for _ in range(1, repeats):
            layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
        return nn.Sequential(*layers)

if __name__ == "__main__":
    resnet=ResNet50(10,[3,4,6,3])

    print(resnet)