import torch
from torch import nn
from libs.hourglassn import Hourglass
import torch.nn.functional as F
class bi_warp(nn.Module):
    def __init__(self, dim = 3, deep = 3):
        super(bi_warp, self).__init__()
        self.hourglass = Hourglass(deep, dim * 2)
        self.conv = nn.Conv2d(dim * 2, 2, kernel_size = 1)
        self.c1 = nn.Conv2d(2, 3, 3, 2)
        self.p1 = nn.MaxPool2d(3, 1)
        self.c2 = nn.Conv2d(3, 1, 3, 2)
        self.p2 = nn.AvgPool2d(3, 1)
        self.l1 = nn.Linear(5776, 64)
        self.l2 = nn.Linear(64, 12)
    def forward(self, storage, target):
        # (1, 3, 320, 320)

        warp = self.hourglass(torch.cat([storage, target], 1).contiguous()) #(1, 6, 480, 640)
        warp = self.conv(warp) #(1, 2, 320, 320)


        theta = self.c1(warp)
        theta = self.p1(theta)
        theta = self.c2(theta)
        theta = self.p2(theta)
        theta = theta.view(theta.size(0), -1).contiguous()
        theta = self.l1(theta)
        theta = self.l2(theta)
        theta = theta.view(theta.size(0), 2, 2, 3).contiguous()
        theta_tar, theta_self = theta.chunk(2, dim=1)  # (1, 2, 2, 3)
        theta_tar = theta_tar.squeeze(1).contiguous() #(1, 2, 3)
        theta_self = theta_self.squeeze(1).contiguous() #(1, 2, 3)

        grid_tar = F.affine_grid(theta_tar, size = target.size())
        grid_self = F.affine_grid(theta_self, size = storage.size())

        r_target = F.grid_sample(storage, grid_tar)
        r_storage = F.grid_sample(target, grid_self)

        # storage = storage.transpose(3, 1).contiguous() # (1, 640, 480, 3)
        # storage = storage.view(storage.size(0), storage.size(1), -1) #(1, 640, 3x480)
        # r_tar = torch.bmm(warp_tar, storage) # (1, 480, 3x480)
        # warp_tar = warp_tar.transpose(2, 1).contiguous() #(1, 640, 480)
        # r_tar = torch.bmm(warp_tar, r_tar) # (1, 640, 3x480)
        # r_tar = r_tar.transpose(2, 1).contiguous() #(1, 3 x 24, 24)
        # r_tar = r_tar.view(bs, 3, h, w) # (1, 3, 24, 24)
        #
        # target = target.transpose(3, 1).contiguous()
        # target = target.view(target.size(0), target.size(1), -1) #(1, 24, 3x24)
        # r_self = torch.bmm(warp_self, target) # (1, 24, 3 x 24)
        # warp_self = warp_self.transpose(2, 1).contiguous() #(1, 640, 480)
        # r_self = torch.bmm(warp_self, r_self) # (1, 640, 3x480)
        # r_self = r_self.transpose(2, 1).contiguous() #(1, 3 x 24, 24)
        # r_self = r_self.view(bs, 3, h, w) # (1, 3, 24, 24)
        return r_storage, r_target



# def homo_warping(src_fea, src_proj, ref_proj, depth_values):
#     # src_fea: [B, C, H, W]
#     # src_proj: [B, 4, 4]
#     # ref_proj: [B, 4, 4]
#     # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
#     # out: [B, C, Ndepth, H, W]
#     batch, channels = src_fea.shape[0], src_fea.shape[1]
#     num_depth = depth_values.shape[1]
#     height, width = src_fea.shape[2], src_fea.shape[3]
#
#
#     proj = torch.matmul(src_proj, torch.inverse(ref_proj[0]))
#     rot = proj[:, :3, :3]  # [B,3,3]
#     trans = proj[:, :3, 3:4]  # [B,3,1]
#     # Generate coordinates.
#     y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
#                            torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
#     y, x = y.contiguous(), x.contiguous()
#     y, x = y.view(height * width), x.view(height * width)
#     xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
#     xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
#     rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
#     rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
#                                                                                         -1)  # [B, 3, Ndepth, H*W]
#     proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
#     proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
#     ## for demon
#     # proj_xy = proj_xyz[:, :2, :, :] / torch.where(proj_xyz[:, 2:3, :, :]==0.0, 1e-6*torch.ones_like(proj_xyz[:, 2:3, :, :]), proj_xyz[:, 2:3, :, :])  # [B, 2, Ndepth, H*W]
#     proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
#     proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
#     proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
#     grid = proj_xy
#
#     warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
#                                    padding_mode='zeros')
#     warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
#
#     return warped_src_fea
#
# class homo_map(nn.Module):
#
#     def __init__(self, dim = 3, deep = 1):
#         super(homo_map, self).__init__()
#         self.hourglass = Hourglass(deep, dim * 2)
#         self.conv1 = nn.Conv2d(dim * 2, 2, kernel_size = 4, stride=4)
#         self.pool1 = nn.MaxPool2d(3, 3)
#         self.conv2 = nn.Conv2d(2, 2, kernel_size=2, stride = 2)
#         self.pool2 = nn.MaxPool2d((2,3), 1)
#     def forward(self, storage, target):
#         h = storage.size(2)
#         w = storage.size(3)
#         bs = storage.size(0)
#         warp = self.hourglass(torch.cat([storage, target], 1).contiguous()) #(1, 6, 480, 640)
#         warp = self.conv1(warp)
#         warp = self.pool1(warp)
#         warp = self.conv2(warp)
#         warp = self.pool2(warp) # (1, 2, 4, 4)
#         warp_tar, warp_self = warp.chunk(2, dim = 1)
#         warp_tar, warp_self = warp_tar.squeeze(1), warp_self.squeeze(1) # (1, 4, 4)
#         # (1, 32, 480, 640)
#         depth_range = 0.5 + 0.1 * torch.arange(32, dtype=torch.float32).reshape(1, -1, 1, 1).repeat(1, 1, h, w)
#         # (1, 3, 32, 480, 640)
#         r_t = homo_warping(storage, warp_tar, warp_self, depth_range)
#         r_s = homo_warping(target, warp_self, depth_range)
#
#         return r_s, r_t