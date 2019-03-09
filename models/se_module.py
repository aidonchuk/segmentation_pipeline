from torch.nn import AdaptiveAvgPool2d, Sequential, Linear, ReLU, Sigmoid, Module


class SCSEBlock(Module):
    def __init__(self, channel, red=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc = Sequential(
            Linear(channel, red), ReLU(inplace=True),
            Linear(red, channel), Sigmoid()
        )

    def forward(self, x):
        a, b, _, _ = x.size()
        y = self.fc(self.avg_pool(x).view(a, b)).view(a, b, 1, 1)
        return x * y
