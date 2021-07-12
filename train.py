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
from matplotlib import pyplot as plt
import sys
from tqdm import tqdm
cate_list = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = 'NOCS', help='dataset root dir')
parser.add_argument('--resume', type=str, default = '',  help='resume model')
parser.add_argument('--category', type=int, default = 5,  help='category to train')
parser.add_argument('--num_points', type=int, default = 576, help='points')
parser.add_argument('--num_cates', type=int, default = 6, help='number of categories')
parser.add_argument('--workers', type=int, default = 5, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default = 8, help='number of kp')
parser.add_argument('--outf', type=str, default = 'models/', help='save dir')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--w_size', default=5, type=int)
opt = parser.parse_args()

model = KeyNet(num_points = opt.num_points, num_key = opt.num_kp, num_cates = opt.num_cates, opt = opt)
if opt.cuda == True:
    model.cuda()

if opt.resume != '':
    model.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume)))
# The num_points is a predefined value which is 500 and it means the maximum 3-d points needed to be considered for one object. This value is equal to the number of points in point cloud and choose list.
dataset = Dataset('train', opt.dataset_root, True, opt.num_points, opt.num_cates, 5000, opt.category, opt.w_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, opt.num_cates, 1000, opt.category, opt.w_size)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

criterion = Loss(opt.num_kp, opt.num_cates, opt=opt)

best_test = np.Inf
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

for epoch in tqdm(range(0, opt.epoch)):
    model.train()
    train_dis_avg = 0.0
    train_count = 0

    optimizer.zero_grad()

    for i, data in tqdm(enumerate(dataloader, 0)):
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = data
        '''
        Output size: img_fr:torch.Size([1, 3, 320, 160])||choose_fr:torch.Size([1, 1, 500])||cloud_fr:torch.Size([1, 500, 3])||r_fr:torch.Size([1, 3, 3])||t_fr:torch.Size([1, 3])||mesh:torch.Size([1, 3895, 3])||anchor:torch.Size([1, 125, 3])
        img_fr: The image after 2d bounding box cropping, while the original size of the image is 480 x 640.
        Choose: 1-d list
        Cloud: 2-d matrix indicate the 3d coordinates of the object.
        anchor: The 3-d coordinates of generated anchors.
        cate: The ground truth category.
        '''
        # plt.imshow(torch.transpose(img_fr.squeeze(0),0,2).numpy() * 255.)
        # plt.show()
        # plt.imshow(torch.transpose(img_to.squeeze(0),0,2).numpy() * 255.)
        # plt.show()
        # print('img_fr:{}||choose_fr:{}||cloud_fr:{}||r_fr:{}||t_fr:{}||mesh:{}||anchor:{}||scale:{}||cate:{}'.format(img_fr.size(),choose_fr.size(),cloud_fr.size(),r_fr.size(),t_fr.size(),mesh.size(),anchor.size(),scale.size(),cate.size()))

        # sys.exit()
        if opt.cuda == True:
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = Variable(img_fr).cuda(), \
                                                                                                                         Variable(choose_fr).cuda(), \
                                                                                                                         Variable(cloud_fr).cuda(), \
                                                                                                                         Variable(r_fr).cuda(), \
                                                                                                                         Variable(t_fr).cuda(), \
                                                                                                                         Variable(img_to).cuda(), \
                                                                                                                         Variable(choose_to).cuda(), \
                                                                                                                         Variable(cloud_to).cuda(), \
                                                                                                                         Variable(r_to).cuda(), \
                                                                                                                         Variable(t_to).cuda(), \
                                                                                                                         Variable(mesh).cuda(), \
                                                                                                                         Variable(anchor).cuda(), \
                                                                                                                         Variable(scale).cuda(), \
                                                                                                                         Variable(cate).cuda()
        else:
            img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = Variable(img_fr), \
                                                                                                                         Variable(choose_fr), \
                                                                                                                         Variable(cloud_fr), \
                                                                                                                         Variable(r_fr), \
                                                                                                                         Variable(t_fr), \
                                                                                                                         Variable(img_to), \
                                                                                                                         Variable(choose_to), \
                                                                                                                         Variable(cloud_to), \
                                                                                                                         Variable(r_to), \
                                                                                                                         Variable(t_to), \
                                                                                                                         Variable(mesh), \
                                                                                                                         Variable(anchor), \
                                                                                                                         Variable(scale), \
                                                                                                                         Variable(cate)
        # kp_fr: (1, 8, 3), anc_fr:(1, 125, 3), att_fr:(1, 125), reconstruct_set:(1, 4, 2, 3, 24, 24)
        Kp_fr, anc_fr, att_fr, reconstruct_set_fr, original_set_fr = model(img_fr, choose_fr, cloud_fr, anchor, scale, cate, t_fr)
        Kp_to, anc_to, att_to, reconstruct_set_to, original_set_to = model(img_to, choose_to, cloud_to, anchor, scale, cate, t_to)

        loss, _ = criterion(opt, Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate, reconstruct_set_fr, reconstruct_set_to, original_set_fr, original_set_to)
        loss.backward()

        train_dis_avg += loss.item()
        train_count += 1

        if train_count != 0 and train_count % 8 == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(train_count, float(train_dis_avg) / 8.0)
            train_dis_avg = 0.0

        if train_count != 0 and train_count % 100 == 0:
            torch.save(model.state_dict(), '{0}/model_current_{1}.pth'.format(opt.outf, cate_list[opt.category-1]))


    optimizer.zero_grad()
    model.eval()
    score = []
    for j, data in enumerate(testdataloader, 0):
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = data
        img_fr, choose_fr, cloud_fr, r_fr, t_fr, img_to, choose_to, cloud_to, r_to, t_to, mesh, anchor, scale, cate = Variable(img_fr).cuda(), \
                                                                                                                     Variable(choose_fr).cuda(), \
                                                                                                                     Variable(cloud_fr).cuda(), \
                                                                                                                     Variable(r_fr).cuda(), \
                                                                                                                     Variable(t_fr).cuda(), \
                                                                                                                     Variable(img_to).cuda(), \
                                                                                                                     Variable(choose_to).cuda(), \
                                                                                                                     Variable(cloud_to).cuda(), \
                                                                                                                     Variable(r_to).cuda(), \
                                                                                                                     Variable(t_to).cuda(), \
                                                                                                                     Variable(mesh).cuda(), \
                                                                                                                     Variable(anchor).cuda(), \
                                                                                                                     Variable(scale).cuda(), \
                                                                                                                     Variable(cate).cuda()

        Kp_fr, anc_fr, att_fr = model(img_fr, choose_fr, cloud_fr, anchor, scale, cate, t_fr)
        Kp_to, anc_to, att_to = model(img_to, choose_to, cloud_to, anchor, scale, cate, t_to)

        _, item_score = criterion(Kp_fr, Kp_to, anc_fr, anc_to, att_fr, att_to, r_fr, t_fr, r_to, t_to, mesh, scale, cate)
        
        print(item_score)
        score.append(item_score)

    test_dis = np.mean(np.array(score))
    if test_dis < best_test:
        best_test = test_dis
        torch.save(model.state_dict(), '{0}/model_{1}_{2}_{3}.pth'.format(opt.outf, epoch, test_dis, cate_list[opt.category-1]))
        print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
