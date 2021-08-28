import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from libs.pspnet import PSPNet
import torch.distributions as tdist
import copy
import sys
from libs.SSA import SSA_Sp, CrossAttention
from libs.warpn import bi_warp
import pytorch_ssim
from libs.triplet import TripletNet
from libs.triplet import SiameseNet
from libs.triplet import EmbeddingNet
from libs.triplet import ContrastiveLoss
import pytorch_ssim
psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        self.model = psp_models['resnet18'.lower()]()
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x


class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 256, 1)

        self.all_conv1 = torch.nn.Conv1d(640, 320, 1)
        self.all_conv2 = torch.nn.Conv1d(320, 160, 1)

        self.num_points = num_points

    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = torch.cat([pointfeat_1, pointfeat_2, x], dim=1).contiguous()  # 128 + 256 + 256

        x = F.leaky_relu(self.all_conv1(x))
        x = self.all_conv2(x)

        return x


class PriorityQueue(nn.Module):
    def __init__(self, w_size, strategy, triplet=None):
        super(PriorityQueue, self).__init__()
        self.w_size = w_size
        self.frame_list = []
        self.triplet = triplet
        self.strategy = strategy
        self.score = ContrastiveLoss()
        self.feat_list = []

    def get_triplet_score(self, x, y):
        x, y = self.triplet(x, y)  # (1, 128)
        euclidean_distance = self.score(x, y)
        return euclidean_distance

    def get_feat(self):
        result_feat = torch.FloatTensor(self.feat_x_set).cuda()
        result_feat = result_feat.transpose(1, 0).contiguous()  # (1, 5, 125, 500, 160)
        if len(self.frame_list) < self.w_size:
            substra = torch.zeros(result_feat.size(0), self.w_size - len(self.frame_list), result_feat.size(2),
                                  result_feat.size(3), result_feat.size(4))
            result_feat = torch.cat((result_feat, substra), 1)
        return result_feat

    def forward(self, x, x_feat, reconstruct):
        if len(self.frame_list) < self.w_size:
            self.frame_list.append(x)
            self.feat_list.append(x_feat)
        else:
            ssim_to_stor = []
            ssim_to_cur = []
            for i in self.frame_list:
                stor, tar = reconstruct(i, x)
                if self.strategy == 'ssim':
                    similarity_to_stor = pytorch_ssim.ssim(stor, i, window_size=4).data[0]
                    similarity_to_cur = pytorch_ssim.ssim(tar, i, window_size=4).data[0]
                else:
                    similarity_to_stor = self.get_triplet_score(stor, i)  # TRUE FALSE FALSE
                    similarity_to_cur = self.get_triplet_score(tar, i)

                ssim_to_stor.append(similarity_to_stor)
                ssim_to_cur.append(similarity_to_cur)
            ssim_to_cur = F.softmax(torch.FloatTensor(ssim_to_cur).cuda())
            ssim_to_stor = F.softmax(torch.FloatTensor(ssim_to_stor).cuda())
            s_cur, idx_cur = ssim_to_cur.sort(0, descending=False)
            s_stor, idx_stor = ssim_to_stor.sort(0, descending=False)
            for ind, item in enumerate(s_stor):
                which = idx_stor[ind]
                if ind > np.argwhere(idx_cur.numpy() == which)[0][0]:
                    self.frame_list.pop(idx_stor[ind])
                    self.frame_list.append(x)
                    self.feat_list.append(x_feat)


class KeyNet(nn.Module):
    def __init__(self, num_points, num_key, num_cates, opt):
        super(KeyNet, self).__init__()
        self.opt = opt
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        self.num_cates = num_cates
        self.sim = opt.sim
        self.sm = torch.nn.Softmax(dim=2)
        self.w_size = opt.w_size
        self.kp_1 = torch.nn.Conv1d(160, 90, 1)
        self.kp_2 = torch.nn.Conv1d(90, 3 * num_key, 1)

        self.att_1 = torch.nn.Conv1d(160, 90, 1)
        self.att_2 = torch.nn.Conv1d(90, 1, 1)

        self.sm2 = torch.nn.Softmax(dim=1)

        if self.sim != 'ssim':
            self.embed = EmbeddingNet()
            self.triplet = SiameseNet(self.embed)
            self.sia_loss = ContrastiveLoss()
            self.queue = PriorityQueue(opt.w_size, opt.sim, self.triplet)
        else:
            self.ssim_loss = pytorch_ssim.SSIM()
            self.queue = PriorityQueue(opt.w_size, opt.sim)
        self.num_key = num_key
        # self.ssa_sp = SSA_Sp(3)
        self.cross_attention = CrossAttention(opt.w_size)
        if opt.cuda == True:
            self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).cuda().view(1, 1, 3).repeat(1, self.num_points, 1)
        else:
            self.threezero = Variable(torch.from_numpy(np.array([0, 0, 0]).astype(np.float32))).view(1, 1, 3).repeat(1, self.num_points, 1)
        self.reconstruct = bi_warp()
    def forward(self, img_set, choose_set, x_set, anchor_set, scale_set, cate, gt_t_set , bb_set):
        # bb_set: (1, 4, 8)
        # img_set: (1, 4, 3, 480, 640)
        # When training
        choose_set = choose_set.transpose(1, 0).contiguous()
        x_set = x_set.transpose(1, 0).contiguous()
        anchor_set = anchor_set.transpose(1, 0).contiguous()
        scale_set = scale_set.transpose(1, 0).contiguous()
        gt_t_set = gt_t_set.transpose(1, 0).contiguous()
        img_set = img_set.transpose(1, 0).contiguous()
        bb_set = bb_set.transpose(1, 0).contiguous()
        feat_x_set = []
        # Storage image set
        for index, (img, choose, x, anchor, scale, gt_t, bb) in enumerate(zip(img_set, choose_set, x_set, anchor_set, scale_set, gt_t_set, bb_set)):
            # bb (1, 4)
            # img (1, 3, w, h)
            # x is cloud. size(1, 500, 3)
            num_anc = len(anchor[0])  # anchor size: (125, 3), number of anchors:125

            # s_ant = self.ssa_sp(img)
            out_img = self.cnn(img)  # img size(1,3,w<480,h<640) output size(1,32, w,h), the output w and h is identical to the original image size.
            # Spatial Attention
            # (1, 32, 480, 640)

            # img_set[index] = s_ant
            bs, di, _, _ = out_img.size()

            #choose = choose.squeeze(1).contiguous() # (bs, 500)
            emb_t = out_img
            # emb_t: (bs, 32, 480, 640)
            for idx in range(bs) :
                temp = emb_t[idx][:, int(bb[idx][0].item()):int(bb[idx][1].item()),int(bb[idx][2].item()):int(bb[idx][3].item())]
                # (32, W x h)
                temp = temp.contiguous()
                temp = temp.view(di, -1)  # size(32, h x w)
                # (32, 500)
                temp = torch.gather(temp, 1, choose[idx].repeat(di,1)).contiguous()
                if (idx == 0):
                    emb = temp.unsqueeze(0).contiguous()
                else:
                    emb = torch.cat([emb,temp.unsqueeze(0).contiguous()], dim = 0).contiguous()
            # emb size: (bs, 32, 500)

            # Assign image embedding to 125 anchors.
            emb = emb.repeat(1, 1, num_anc).contiguous()  # size(1, 32, 500 x 125)
            output_anchor = anchor.view(bs, num_anc, 3)
            # anchor size:(1, 125, 3)
            # anchor.view(1, num_anc, 1, 3) size:(1, 125, 1, 3) anchor_for_key size:(1, 125, 8, 3)
            # self.num_key is the default number of keypoints
            # Assign the 125 anchors to 8 kepoints.
            anchor_for_key = anchor.view(bs, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
            # anchor size:(1, 125, 500, 3)
            # Assign the 125 anchors to 500 points
            anchor = anchor.view(bs, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)
            # x size(1, 125, 500, 3)
            # Assign 500 cloud points to 125 anchors.
            x = x.view(bs, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
            # This step is to compute the distance between each anchor and could points.
            # x size:(1,125 x 500, 3)
            x = (x - anchor).view(bs, num_anc * self.num_points, 3).contiguous()
            # x size:(1, 3, 125 x 500)
            x = x.transpose(2, 1).contiguous()
            # emb size(1, 32, 500 x 125)
            # The feature of each color points.
            feat_x = self.feat(x, emb)  # (DenseFusion) Combine 3D information x and image embedding emb output size(1, 160, 62500=500 x 125)
            # (1, 62500, 160)
            feat_x = feat_x.transpose(2, 1).contiguous()
            # (1, 125, 500, 160)
            # Points features
            feat_x = feat_x.view(bs, num_anc, self.num_points, 160).contiguous()
            # (1, 125, 500, 3)
            loc = x.transpose(2, 1).contiguous().view(bs, num_anc, self.num_points, 3)
            # sm is softmax function
            # Times -1 is because need to use distance to compute weight, a higher weight corresponds to a closer distance. This will be used to the summation of points.
            # Norm is to compute the distance
            # (1, 125, 500)
            weight = self.sm(-1.0 * torch.norm(loc, dim=3)).contiguous()
            # (1, 125, 500, 160)
            weight = weight.view(bs, num_anc, self.num_points, 1).repeat(1, 1, 1, 160).contiguous()
            feat_x = feat_x * weight
            feat_x = torch.sum((feat_x), dim=2).contiguous().view(bs, num_anc, 160)#### 500 disappeared right here.
            feat_x_set.append(feat_x) #(4, 1, 125, 160)

        # Image set including current frame and prevous frames.
        # img_set size:(4, 1, 3, 480, 640)
        target = img_set[-1]  # ( 1(bs), 3, 480, 640)
        store_image_set = img_set[0:len(img_set) - 1] # (3, 1, 3, 480, 640)
        # Generate warping images for storage images and target image respectively and store them in reconstruct_set.
        for ii, st in enumerate(store_image_set):
            r_self, r_tar = self.reconstruct(st, target) # (bs, 3, 480, 640)


            ssim_self_fr = 0.85 * (1 - self.ssim_loss(r_self, st)) / 2 + 0.15 * torch.mean(r_self - st)
            ssim_tar_fr = 0.85 * (1 - self.ssim_loss(r_tar, target)) / 2 + 0.15 * torch.mean(r_tar - target)
            if self.opt.sim != 'ssim':
                emb_rc_tar, emb_tar = self.triplet(r_tar, target)  # (1, 128)
                emb_rc_self, emb_tar = self.triplet(r_self, target)  # (1, 128)
                emb_rc_self, emb_self = self.triplet(r_self, st)
                t_l = self.sia_loss(emb_rc_tar, emb_tar, torch.zeros([emb_rc_tar.size(0), 1], dtype=torch.float).cuda())+ self.sia_loss(emb_rc_tar, emb_rc_self, torch.ones([emb_rc_tar.size(0), 1], dtype=torch.float).cuda()) + self.sia_loss(emb_tar, emb_rc_self, torch.ones([emb_rc_tar.size(0), 1], dtype=torch.float).cuda()) + self.sia_loss(emb_self, emb_tar, torch.ones([emb_rc_tar.size(0), 1], dtype=torch.float).cuda()) + self.sia_loss(emb_rc_self, emb_self, torch.zeros([emb_rc_tar.size(0), 1], dtype=torch.float).cuda()) + self.sia_loss(emb_self, emb_rc_tar, torch.ones([emb_rc_tar.size(0), 1], dtype=torch.float).cuda())
                if ii == 0 :
                    Loss_sia = t_l
                else:
                    Loss_sia += t_l


        ssim_total = ssim_self_fr + ssim_tar_fr


        # Compute cross attention bettween current frames and stored frames.
        # feat_x_set = torch.from_numpy(np.array(feat_x_set).astype(np.float32))
        # (4, bs , 125, 160)
        feat_x_set = torch.cat([i.unsqueeze(0).contiguous() for i in feat_x_set], dim=0) #After unsqueeze(0) (1, bs, 125, 160)
        feat_x_set = feat_x_set.transpose(1, 0).contiguous()  # (bs, 5, 125, 160)
        s_f = feat_x_set[:, 0:self.opt.w_size , : , :]  # (bs, 4, 125, 160)
        t_f = feat_x_set[:, feat_x_set.size(1) - 1, :, :]  # (bs, 1, 125, 160)
        t_f = t_f.unsqueeze(1).contiguous()
        # Cross attention across frames in the memory_bank.
        w_feat = self.cross_attention(s_f, t_f)  # (bs, 4, 125, 160)
        feat_x = torch.sum(w_feat, dim=1)  # (bs , 125, 160)

        # Anchor features
        # The weighted summation of points to 125 represent anchors whose feature dimension is 160.
        # (1, 125, 160)
        # feat_x = torch.sum((feat_x), dim=2).contiguous().view(1, num_anc,160)  ###Anchor features This is the local spatial attention feature, it will be used to selected the one with highest score to generate k keypoints.
        # (1, 160, 125)
        feat_x = feat_x.transpose(2, 1).contiguous()
        # Generate 8 keypoints according to the features from feat_x.
        kp_feat = F.leaky_relu(self.kp_1(feat_x))
        kp_feat = self.kp_2(kp_feat)
        # (1, 125, 24)
        # The final keypoint features.
        kp_feat = kp_feat.transpose(2, 1).contiguous()
        # num_key: 8
        # Order 8 keypoints 3d coordinates.
        # kp_x size: (1, 125, 8, 3)
        kp_x = kp_feat.view(bs, num_anc, self.num_key, 3).contiguous()
        kp_x = (kp_x + anchor_for_key).contiguous()
        # att_1 is 1-d conv with 1 kernel size.
        # (1, 90, 125)
        att_feat = F.leaky_relu(self.att_1(feat_x))
        # The score for each anchor.
        # (1, 1, 125)
        att_feat = self.att_2(att_feat)
        att_feat = att_feat.transpose(2, 1).contiguous()
        # (1, 125)
        att_feat = att_feat.view(bs, num_anc).contiguous()
        att_x = self.sm2(att_feat).contiguous()
        # (1, 125, 3)
        scale_anc = scale.view(bs, 1, 3).repeat(1, num_anc, 1)
        # output_anchor = original anchor(1, 125, 3)
        output_anchor = (output_anchor * scale_anc).contiguous()
        # min_choose is index of the anchor that is closest to the centroid of the object.
        # (1,)
        min_choose = torch.argmin(torch.norm(output_anchor - gt_t.unsqueeze(1), dim=2), dim = 1)
        # (1, 125, 24)
        all_kp_x = kp_x.view(bs, num_anc, 3 * self.num_key).contiguous()
        # Select the anchor with the index min_choose. (1, 1, 24)
        all_kp_x = torch.gather(all_kp_x, 1, min_choose.unsqueeze(1).unsqueeze(1).repeat(1, 1 ,3 * self.num_key)).contiguous()
        # (1, 8, 3)
        all_kp_x = all_kp_x.view(bs, self.num_key, 3).contiguous()
        # (1, 8, 3)
        scale_kp = scale.view(bs, 1, 3).repeat(1, self.num_key, 1)
        all_kp_x = (all_kp_x * scale_kp).contiguous()
        # The purpose of attention module is to predict the anchor that is the closest to the centroid of object, So need to leverage gt_t to compute the index of ground truth anchor and extract the cooresponding keypoints.
        # (Ground Truth)all_kp_x: The selected kepoint coordinate that is closest to the centroid, (1, 8, 3) .This is infer from the weighted summed anchor feature to select the one that is closest to the object centroid.
        # output_anchor: the anchor coordinate, (1, 125, 3)
        # att_x: (1, 125), Attention score for each anchor point.
        if self.opt.sim != 'ssim':
            return all_kp_x, output_anchor, att_x, ssim_total, Loss_sia
        return all_kp_x, output_anchor, att_x, ssim_total

    def eval_forward(self, img, choose, ori_x, anchor, scale, space, first):
        num_anc = len(anchor[0])
        out_img = self.cnn(img)
        bs, di, _, _ = out_img.size()

        emb = out_img.view(bs, di, -1)
        choose = choose.repeat(1, di, 1)
        emb = torch.gather(emb, 2, choose)
        emb = emb.repeat(1, 1, num_anc).detach()
        # print(emb.size())
        rc_img = torch.gather(img.view(bs, 3, -1), 2, choose).contiguous()
        rc_img = rc_img.view(bs, 3, 24, 24)

        output_anchor = anchor.view(1, num_anc, 3)
        anchor_for_key = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_key, 1)
        anchor = anchor.view(1, num_anc, 1, 3).repeat(1, 1, self.num_points, 1)

        candidate_list = [-10 * space, 0.0, 10 * space]
        if space != 0.0:
            add_on = []
            for add_x in candidate_list:
                for add_y in candidate_list:
                    for add_z in candidate_list:
                        add_on.append([add_x, add_y, add_z])

            add_on = Variable(torch.from_numpy(np.array(add_on).astype(np.float32))).cuda().view(27, 1, 3)
        else:
            add_on = Variable(torch.from_numpy(np.array([0.0, 0.0, 0.0]).astype(np.float32))).cuda().view(1, 1, 3)

        all_kp_x = []
        all_att_choose = []
        scale_add_on = scale.view(1, 3)

        for tmp_add_on in add_on:
            tmp_add_on_scale = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_points, 1)
            tmp_add_on_key = (tmp_add_on / scale_add_on).view(1, 1, 3).repeat(1, self.num_key, 1)
            x = ori_x - tmp_add_on_scale

            x = x.view(1, 1, self.num_points, 3).repeat(1, num_anc, 1, 1)
            x = (x - anchor).view(1, num_anc * self.num_points, 3)

            x = x.transpose(2, 1)
            feat_x = self.feat(x, emb)
            feat_x = feat_x.transpose(2, 1)
            feat_x = feat_x.view(1, num_anc, self.num_points, 160).detach()
            # Spatial Attention
            feat_x = self.ssa_sp(feat_x)
            loc = x.transpose(2, 1).view(1, num_anc, self.num_points, 3)
            weight = self.sm(-1.0 * torch.norm(loc, dim=3))
            weight = weight.view(1, num_anc, self.num_points, 1).repeat(1, 1, 1, 160)
            feat_x = feat_x * weight
            # Cross Attention
            self.queue(rc_img, feat_x, self.reconstruct)
            feat_x_set = self.queue.get_feat()

            # Cross attention across frames in the memory_bank.
            w_feat = self.cross_attention(feat_x_set, feat_x.unsqueeze(1))
            feat_x = torch.sum(w_feat, dim=1)  # (1 , 125 , 500, 160)

            feat_x = torch.sum((feat_x), dim=2).view(1, num_anc, 160)
            feat_x = feat_x.transpose(2, 1).detach()

            kp_feat = F.leaky_relu(self.kp_1(feat_x))
            kp_feat = self.kp_2(kp_feat)
            kp_feat = kp_feat.transpose(2, 1)
            kp_x = kp_feat.view(1, num_anc, self.num_key, 3)
            kp_x = (kp_x + anchor_for_key).detach()

            att_feat = F.leaky_relu(self.att_1(feat_x))
            att_feat = self.att_2(att_feat)
            att_feat = att_feat.transpose(2, 1)
            att_feat = att_feat.view(1, num_anc)
            att_x = self.sm2(att_feat).detach()

            if not first:
                att_choose = torch.argmax(att_x.view(-1))
            else:
                att_choose = Variable(torch.from_numpy(np.array([62])).long()).cuda().view(-1)

            scale_anc = scale.view(1, 1, 3).repeat(1, num_anc, 1)
            output_anchor = (output_anchor * scale_anc)

            scale_kp = scale.view(1, 1, 3).repeat(1, self.num_key, 1)
            kp_x = kp_x.view(1, num_anc, 3 * self.num_key).detach()
            kp_x = (kp_x[:, att_choose, :].view(1, self.num_key, 3) + tmp_add_on_key).detach()

            kp_x = kp_x * scale_kp

            all_kp_x.append(copy.deepcopy(kp_x.detach()))
            all_att_choose.append(copy.deepcopy(att_choose.detach()))

        return all_kp_x, all_att_choose