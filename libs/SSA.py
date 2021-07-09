import torch
from torch import nn
from einops import rearrange

class SSA_Sp(nn.Module):
    def __init__(self, dim):
        super(SSA_Sp, self).__init__()

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size = 1)
        self.attend = nn.Softmax(dim = -1)

    def forward(self, x):
        # x size :(1, 125, 500, 160)
        x = x.transpose(3,1).contiguous() # (1, 160, 500, 125)
        qvk = self.to_qkv(x)# qvk size: (1, 480, 500, 125)
        # q = k = v = (1, 160, 500, 125)
        q, k, v = qvk.chunk(3, dim=1)
        # Cloud points
        q = q.transpose(2,1).contiguous() # (1, 500, 160, 125)
        qk = q.view(q.size(0), -1, q.size(3))#(1, 500x160, 125)
        q = q.view(q.size(0), q.size(1), -1) # (1, 500, 160x125)
        kq = k.transpose(3,2).contiguous().view(k.size(0), -1, k.size(3))
        ms = self.attend(torch.bmm(q, kq))# (1, 500, 500)

        # Anchors
        k = k.transpose(3,1).contiguous()# (1, 125, 500, 160)
        k = k.view(k.size(0), k.size(1), -1)# (1, 125, 500x160)
        ma = self.attend(torch.bmm(k,qk))# (1, 125, 125)

        vq = v.transpose(2,1).contiguous()
        vq = vq.view(vq.size(0), vq.size(1), -1)#(1, 500, -1)
        vk = v.transpose(3,1).contiguous()
        vk = vk.view(vk.size(0), vk.size(1), -1)#(1, 125, -1)

        ms = torch.bmm(ms, vq)
        ma = torch.bmm(ma, vk)
        ms = ms.view(ms.size(0), ms.size(1), x.size(1), x.size(3)).transpose(2, 1).contiguous()
        ma = ma.view(ma.size(0), ma.size(1), x.size(2), x.size(1)).transpose(3, 1).contiguous() # (1, 160, 500, 125)

        return (ms + ma).transpose(3, 1).contiguous()

class SSA_Temp(nn.Module):
    def __init__(self, dim, w_size):
        super(SSA_Temp, self).__init__()

        self.attend = nn.Softmax(dim = -1)
        self.to_temporal_qk = nn.Conv3d(dim, dim * 2,
                                  kernel_size=(3, 1, 1),
                                  padding=(1, 0, 0))