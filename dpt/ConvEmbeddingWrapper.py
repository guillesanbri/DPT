from torch import nn


class EfficientnetEmbeddingWrapper(nn.Module):
    def __init__(self, efficientnet):
        super(EfficientnetEmbeddingWrapper, self).__init__()
        self.conv_stem = efficientnet.conv_stem
        self.bn1 = efficientnet.bn1
        self.act1 = efficientnet.act1
        self.stages = efficientnet.blocks
        self.conv_head = efficientnet.conv_head
        self.bn2 = efficientnet.bn2
        self.act2 = efficientnet.act2

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.stages(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
