import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataset.dataset_nocs import Dataset
from libs.network import KeyNet
from libs.loss import Loss
import torch.nn.functional as F
from libs.warpn import bi_warp
from libs.warpn import homo_map
from Homography_net import Homo_net
import pytorch_ssim
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/content/drive/MyDrive/Dataset/NOCS', help='dataset root dir')
parser.add_argument('--category', type=int, default = 5,  help='category to train')
parser.add_argument('--num_points', type=int, default = 300, help='points')
parser.add_argument('--num_cates', type=int, default = 6, help='number of categories')
parser.add_argument('--workers', type=int, default = 1, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default = 8, help='number of kp')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--img_out', type = str, default='/content/drive/MyDrive/Dataset/NOCS/Homography_results')
opt = parser.parse_args()


dataset = Dataset('train', opt.dataset_root, True, opt.num_points, opt.num_cates, 5000, opt.category, opt.w_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, opt.num_cates, 1000, opt.category, opt.w_size)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

model = Homo_net(opt)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

for epoch in range(0, opt.epoch):
    model.train()
    train_dis_avg = 0.0
    train_count = 0

    SSIM_loss = ssim_loss = pytorch_ssim.SSIM()
    optimizer.zero_grad()
    loss_sum = 0.
    num = 0
    for i, data in enumerate(dataloader, 0):
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate, bb_set = Variable(
                                                                                                                                img_fr).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  choose_fr).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  cloud_fr).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  r_fr).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  t_fr).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  img_to).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  choose_to).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  cloud_to).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  r_to).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  t_to).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  mesh).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  anchor).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  scale).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  cate).cuda(), \
                                                                                                                              Variable(
                                                                                                                                  bb_set).cuda()
        img_fr = img_fr.transpose(1, 0).contiguous()[0]
        img_to = img_to.transpose(1, 0).contiguous()[0]
        r_fr, r_to = model(img_fr, img_to) #(1, 3, 32, 480, 640)

        r_fr, r_to = r_fr.tranpose(2,0, 1).contiguous(), r_to.transpose(2, 0, 1).contiguous() #(32, 1, 3, 480, 640)

        for ind, item in enumerate(r_fr):
            if ind == 0:
                loss =  0.85 * (1 - SSIM_loss(r_fr[0], img_fr)) / 2 + 0.15 * torch.mean(r_fr[0] - img_fr) + 0.85 * (1 - SSIM_loss(r_to[0], img_to)) / 2 + 0.15 * torch.mean(r_to[0] - img_to)
            else:
                loss += 0.85 * (1 - SSIM_loss(r_fr[0], img_fr)) / 2 + 0.15 * torch.mean(r_fr[0] - img_fr) + 0.85 * (1 - SSIM_loss(r_to[0], img_to)) / 2 + 0.15 * torch.mean(r_to[0] - img_to)
        loss.backward()
        loss_sum + loss.item()
        num += 1
    print('--> Epoch:{}, loss:{}'.format(epoch, loss_sum/num))
    if epoch % 10 == 0:
        os.makedirs(os.path.join(opt.img_out, ''.join(['epoch',str(epoch)])), exist_ok=True)
        img_path = os.path.join(opt.img_out, ''.join(['epoch',str(epoch)]))
        r_fr, r_to = r_fr.tranpose(1, 2, 0).contiguous(), r_to.transpose(1, 2, 0).contiguous()
        for i in range(32):
            warped_img = r_fr[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
            img_np = warped_img[0].detach().cpu().numpy()
            img_np = img_np[:, :, ::-1]

            alpha = 0.5
            beta = 1 - alpha
            gamma = 0
            img_np = np.uint8(img_np)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_add = cv2.addWeighted(img_to.transpose(0, 2, 3, 1).contiguous().detach().cpu().numpy(), alpha, img_np, beta, gamma)
            cv2.imwrite(opt.img_out.format(i), np.hstack([img_to.transpose(0, 2, 3, 1).contiguous().detach().cpu().numpy(), img_np, img_add]))
