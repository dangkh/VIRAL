import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer import TransformerEncoder


class CrossmodalNet(nn.Module):
    def __init__(self, inchannels) -> None:
        super(CrossmodalNet, self).__init__()

        self.vt_trans = TransformerEncoder(inchannels, num_heads= 4, layers=1)
        self.vt_self = TransformerEncoder(inchannels, num_heads= 4, layers=1)

        self.tv_trans = TransformerEncoder(inchannels, num_heads= 4, layers=1)
        self.tv_self = TransformerEncoder(inchannels, num_heads= 4, layers=1)

        
    def forward(self, x_s):
        for j in range(len(x_s)):
            x_s[j] = x_s[j].permute(1, 0, 2)

        x0, x1 = x_s[0], x_s[1]
        out0 = self.vt_trans(x0, x1, x1)
        out0 = self.vt_self(out0)

        out1 = self.tv_trans(x1, x0, x0)
        out1 = self.tv_self(out1)

        return out0, out1

if __name__ == '__main__':
    encoder = CrossmodalNet(100)
    x1 = torch.tensor(torch.rand(32, 10, 100))
    x2 = torch.tensor(torch.rand(32, 10, 100))
    print(encoder([x1, x2]).shape)