import torch
from torch import nn
from libs.hourglassn import Hourglass

class bi_warp(nn.Module):
    def __init__(self, dim = 3, deep = 1):
        super(bi_warp, self).__init__()
        self.hourglass = Hourglass(deep, dim * 2)
        self.conv = nn.Conv2d(dim * 2, 2, kernel_size = 1)
    def forward(self, storage, target):
        # (1, 3, 480, 640)

        h = storage.size(2)
        w = storage.size(3)
        bs = storage.size(0)
        warp = self.hourglass(torch.cat([storage, target], 1).contiguous()) #(1, 6, 480, 640)
        warp_tar, warp_self = self.conv(warp).chunk(2, dim = 1) #(1, 2, 480, 640)
        warp_tar = warp_tar.squeeze(1)
        warp_self = warp_self.squeeze(1)


        storage = storage.transpose(3, 1).contiguous() # (1, 640, 480, 3)
        storage = storage.view(storage.size(0), storage.size(1), -1) #(1, 640, 3x480)
        r_tar = torch.bmm(warp_tar, storage) # (1, 480, 3x480)
        warp_tar = warp_tar.transpose(2, 1).contiguous() #(1, 640, 480)
        r_tar = torch.bmm(warp_tar, r_tar) # (1, 640, 3x480)
        r_tar = r_tar.transpose(2, 1).contiguous() #(1, 3 x 24, 24)
        r_tar = r_tar.view(bs, 3, h, w) # (1, 3, 24, 24)

        target = target.transpose(3, 1).contiguous()
        target = target.view(target.size(0), target.size(1), -1) #(1, 24, 3x24)
        r_self = torch.bmm(warp_self, target) # (1, 24, 3 x 24)
        warp_self = warp_self.transpose(2, 1).contiguous() #(1, 640, 480)
        r_self = torch.bmm(warp_self, r_self) # (1, 640, 3x480)
        r_self = r_self.transpose(2, 1).contiguous() #(1, 3 x 24, 24)
        r_self = r_self.view(bs, 3, h, w) # (1, 3, 24, 24)
        return r_self, r_tar