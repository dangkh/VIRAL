import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer import TransformerEncoder

def partial_mask(x, mask_ratio=0.3):
    """
    Randomly mask a portion of features in a modality.
    
    Args:
        x: tensor [B, D]
        mask_ratio: percentage of features to mask
    Returns:
        masked_x
    """
    B, D = x.shape
    
    # create random mask
    mask = (torch.rand(B, D, device=x.device) > mask_ratio).float()
    
    masked_x = x * mask
    
    return masked_x, mask

def mask_two_modalities(x1, x2, mask_ratio=0.3):

    x1_masked, mask1 = partial_mask(x1, mask_ratio)
    x2_masked, mask2 = partial_mask(x2, mask_ratio)

    return x1_masked, x2_masked, mask1, mask2

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
        # logits_t2v = torch.matmul(t, v.T) / self.temperature  # (B, B)

        labels = torch.arange(v.size(0)).to(v.device)

        loss_v2t = F.cross_entropy(logits_v2t, labels)
        loss_t2v = F.cross_entropy(logits_v2t.T, labels)
        loss = (loss_v2t + loss_t2v) / 2
        return loss

class CrossmodalNet(nn.Module):
    def __init__(self, inchannels) -> None:
        super(CrossmodalNet, self).__init__()

        self.vt_trans = TransformerEncoder(inchannels, num_heads= 1, layers=1)
        self.vt_self = TransformerEncoder(inchannels, num_heads= 1, layers=1)

        self.tv_trans = TransformerEncoder(inchannels, num_heads= 1, layers=1)
        self.tv_self = TransformerEncoder(inchannels, num_heads= 1, layers=1)
        self.criterion = InfoNCELoss(temperature=0.1)

        
    def forward(self, x_s):
        for j in range(len(x_s)):
            x_s[j] = x_s[j].unsqueeze(0)

        x0, x1 = x_s[0], x_s[1]
        out0 = self.vt_trans(x0, x1, x1)
        out0 = self.vt_self(out0)

        out1 = self.tv_trans(x1, x0, x0)
        out1 = self.tv_self(out1)
        out0 = out0.squeeze(0)
        out1 = out1.squeeze(0)
        out = (out0 + out1) / 2
        loss = self.criterion(out0, out1)
        return out, loss

class RedundantNet(nn.Module):
    def __init__(self, inchannels) -> None:
        super(RedundantNet, self).__init__()

        self.fusion = TransformerEncoder(inchannels, num_heads= 2, layers=1)
        self.criterion = InfoNCELoss(temperature=0.7)
        self.ln = nn.Linear(inchannels*2, inchannels)
        self.mask_ratio = 0.1

    def forward(self, xt, xv):
        t1_masked, v1_masked, _, _ = mask_two_modalities(xt, xv, self.mask_ratio)
        t2_masked, v2_masked, _, _ = mask_two_modalities(xt, xv, self.mask_ratio)
        xt = xt.unsqueeze(0)
        xv = xv.unsqueeze(0)
        t1_masked = t1_masked.unsqueeze(0)
        v1_masked = v1_masked.unsqueeze(0)
        t2_masked = t2_masked.unsqueeze(0)
        v2_masked = v2_masked.unsqueeze(0)
        zero_t = torch.zeros_like(xt) 
        zero_v = torch.zeros_like(xv) 

        zt = torch.cat((xt, zero_v), dim = -1) # Xt
        zv = torch.cat((zero_t, xv), dim = -1) # Xv
        zf = torch.cat((xt, xv), dim = -1) # X

        z1 = torch.cat((t1_masked, v1_masked), dim=-1) # X'
        z2 = torch.cat((t2_masked, v2_masked), dim=-1) # X''
        

        out_xt = self.fusion(self.ln(zt))
        out_xv = self.fusion(self.ln(zv))
        out_z = self.fusion(self.ln(zf))
        outz1 = self.fusion(self.ln(z1))
        outz2 = self.fusion(self.ln(z2))


        out_xt = out_xt.squeeze(0)
        out_xv = out_xv.squeeze(0)
        out_z = out_z.squeeze(0)
        outz1 = outz1.squeeze(0)
        outz2 = outz2.squeeze(0)

        loss = self.criterion(outz1, outz2) + 0.5 * self.criterion(out_xt, outz1)  + 0.5 * self.criterion(out_xt, outz2) + \
                0.5 * self.criterion(out_xv, outz1) + 0.5 * self.criterion(out_xv, outz2)

        return out_z, loss    

if __name__ == '__main__':
    # encoder = CrossmodalNet(64)
    # x1 = torch.tensor(torch.rand(32, 64))
    # x2 = torch.tensor(torch.rand(32, 64))
    # out, ls = encoder([x1, x2])
    # print(out.shape)

    encoder = RedundantNet(64)
    x1 = torch.tensor(torch.rand(32, 64))
    x2 = torch.tensor(torch.rand(32, 64))
    out, ls = encoder(x1, x2)
    print(out.shape)