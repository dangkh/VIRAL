import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer import TransformerEncoder

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, t):
        """
        v: (B, D)  # visual embeddings
        t: (B, D)  # text embeddings
        """
        v = F.normalize(v, dim=-1)
        t = F.normalize(t, dim=-1)

        # similarity matrices
        logits_v2t = torch.matmul(v, t.T) / self.temperature  # (B, B)
        logits_t2v = torch.matmul(t, v.T) / self.temperature  # (B, B)

        labels = torch.arange(v.size(0)).to(v.device)

        loss_v2t = F.cross_entropy(logits_v2t, labels)
        loss_t2v = F.cross_entropy(logits_t2v, labels)
        loss = (loss_v2t + loss_t2v) / 2
        return loss

class CrossmodalNet(nn.Module):
    def __init__(self, inchannels) -> None:
        super(CrossmodalNet, self).__init__()

        self.vt_trans = TransformerEncoder(inchannels, num_heads= 4, layers=1)
        self.vt_self = TransformerEncoder(inchannels, num_heads= 4, layers=1)

        self.tv_trans = TransformerEncoder(inchannels, num_heads= 4, layers=1)
        self.tv_self = TransformerEncoder(inchannels, num_heads= 4, layers=1)
        self.criterion = InfoNCELoss(temperature=0.07)

        
    def forward(self, x_s):
        for j in range(len(x_s)):
            x_s[j] = x_s[j].unsqueeze(0)

        x0, x1 = x_s[0], x_s[1]
        out0 = self.vt_trans(x0, x1, x1)
        out0 = self.vt_self(out0)

        out1 = self.tv_trans(x1, x0, x0)
        out1 = self.tv_self(out1)
        out1 = out1.squeeze(0)
        out2 = out2.squeeze(0)
        out = (out0 + out1) / 2
        loss = self.criterion(out0, out1)
        return out, loss

if __name__ == '__main__':
    encoder = CrossmodalNet(64)
    x1 = torch.tensor(torch.rand(32, 64))
    x2 = torch.tensor(torch.rand(32, 64))
    out, ls = encoder([x1, x2])
    print(out[0].shape)