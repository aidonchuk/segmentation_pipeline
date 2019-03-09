from torch import nn


class SCSEBlock(nn.Module):
    def __init__(self, channel, red=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, red), nn.ReLU(inplace=True),
                nn.Linear(red, channel), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
