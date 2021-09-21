import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
from libs.transformations import euler_matrix
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2
import _pickle as cPickle
from skimage.transform import resize
import matplotlib.pyplot as plt
class Dataset(data.Dataset):
    def __init__(self, mode, root, add_noise, num_pt, num_cates, count, cate_id, w_size, occlude = False):
        # num_cates is the total number of categories gonna be preloaded from dataset, cate_id is the category need to be trained.
        self.root = root
        self.add_noise = add_noise
        self.mode = mode
        self.num_pt = num_pt
        self.occlude = occlude
        self.num_cates = num_cates
        self.back_root = '{0}/train2017/'.format(self.root)
        self.w_size = w_size + 1
        self.cate_id = cate_id
        # Path list: obj_list[], real_obj_list[], back_list[],
        self.obj_list = {}
        self.obj_name_list = {}
        self.cate_set = [1 ,2, 3, 4 ,5, 6]
        self.real_oc_list = {}
        self.real_oc_name_set = {}
        for ca in self.cate_set:
            if ca != self.cate_id:
                real_oc_name_list = os.listdir('{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, str(ca)))
                self.real_oc_name_set[ca] = real_oc_name_list
        del self.cate_set[self.cate_id - 1]
        # Get all the occlusions.
        # print(self.real_oc_name_set)
        for key in self.real_oc_name_set.keys():
            for item in self.real_oc_name_set[key]:
                self.real_oc_list[item] = []

                input_file = open('{0}/data_list/real_{1}/{2}/{3}/list.txt'.format(self.root, self.mode, str(key), item), 'r')
                while 1:
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    self.real_oc_list[item].append('{0}/data/{1}'.format(self.root, input_line))
                input_file.close()


        if self.mode == 'train':
            for tmp_cate_id in range(1, self.num_cates+1):
                # (nxm)obj_name_list[] contains the name list of the super dir(1a9e1fb2a51ffd065b07a27512172330) of training list txt file(train/16069/0008)
                listdir = os.listdir('{0}/data_list/train/{1}/'.format(self.root, tmp_cate_id))
                self.obj_name_list[tmp_cate_id]=[]
                for i in listdir:
                    if os.path.isdir('{0}/data_list/train/{1}/{2}'.format(self.root, tmp_cate_id, i)):
                        self.obj_name_list[tmp_cate_id].append(i)
                # self.obj_name_list[tmp_cate_id] = os.listdir('{0}/data_list/train/{1}/'.format(self.root, tmp_cate_id))
                self.obj_list[tmp_cate_id] = {}

                for item in self.obj_name_list[tmp_cate_id]:
                    #print(tmp_cate_id, item)# item: 1a9e1fb2a51ffd065b07a27512172330
                    self.obj_list[tmp_cate_id][item] = []

                    input_file = open('{0}/data_list/train/{1}/{2}/list.txt'.format(self.root, tmp_cate_id, item), 'r')
                    while 1:
                        input_line = input_file.readline()# read list.txt(train/16069/0008)
                        if not input_line:
                            break
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        # (nxmxk)obj_list is the real training data from {root}/data/train/16069/0008， 0008 here is just a prefix without the 5 suffix indicate the different file like _color.png/mask.png/depth.png/meta.txt_coord.png in 16069 dir.
                        self.obj_list[tmp_cate_id][item].append('{0}/data/{1}'.format(self.root, input_line))
                    input_file.close()


        self.real_obj_list = {}
        self.real_obj_name_list = {}

        for tmp_cate_id in range(1, self.num_cates+1):
            # real_obj_name_list contains the real obj names from {}/data_list/real_train/1/ like bottle_blue_google_norm, bottle_starbuck_norm
            self.real_obj_name_list[tmp_cate_id] = []
            listdir = os.listdir('{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, tmp_cate_id))
            for i in listdir:
                if os.path.isdir('{0}/data_list/real_{1}/{2}/{3}'.format(self.root, self.mode, tmp_cate_id, i)):
                    self.real_obj_name_list[tmp_cate_id].append(i)

            # self.real_obj_name_list[tmp_cate_id] = os.listdir('{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, tmp_cate_id))
            self.real_obj_list[tmp_cate_id] = {}

            for item in self.real_obj_name_list[tmp_cate_id]:
                #print(tmp_cate_id, item) #item : bottle_blue_google_norm
                self.real_obj_list[tmp_cate_id][item] = []
                # real_train/scene_2/0000
                input_file = open('{0}/data_list/real_{1}/{2}/{3}/list.txt'.format(self.root, self.mode, tmp_cate_id, item), 'r')

                while 1:
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    # real_obj_list contains the prefix of files under the dir {}/data/real_train/scene_2/, which are all consecutive frames in video squence.
                    self.real_obj_list[tmp_cate_id][item].append('{0}/data/{1}'.format(self.root, input_line))
                input_file.close()

        self.back_list = []

        input_file = open('dataset/train2017.txt', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
        # back_list is the path list of the images in COCO dataset 2017 are about to be used in the training.
            self.back_list.append(self.back_root + input_line) # back_root is the dir of COCO dataset train2017
        input_file.close()


        self.mesh = []
        input_file = open('dataset/sphere.xyz', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            self.mesh.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        self.mesh = np.array(self.mesh) * 0.6

        self.cam_cx_1 = 322.52500
        self.cam_cy_1 = 244.11084
        self.cam_fx_1 = 591.01250
        self.cam_fy_1 = 590.16775

        self.cam_cx_2 = 319.5
        self.cam_cy_2 = 239.5
        self.cam_fx_2 = 577.5
        self.cam_fy_2 = 577.5

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trancolor = transforms.ColorJitter(0.8, 0.5, 0.5, 0.05)
        self.length = count

    def get_occlusion(self, oc_obj, oc_frame, syn_or_real):
        if syn_or_real:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        else:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        cam_scale = 1.0
        oc_target = []
        oc_input_file = open('{0}/model_scales/{1}.txt'.format(self.root, oc_obj), 'r')
        for i in range(8):
            oc_input_line = oc_input_file.readline()
            if oc_input_line[-1:] == '\n':
                oc_input_line = oc_input_line[:-1]
            oc_input_line = oc_input_line.split(' ')
            oc_target.append([float(oc_input_line[0]), float(oc_input_line[1]), float(oc_input_line[2])])
        oc_input_file.close()
        oc_target = np.array(oc_target)
        r, t, _ = self.get_pose(oc_frame, oc_obj)

        oc_target_tmp = np.dot(oc_target, r.T) + t
        oc_target_tmp[:, 0] *= -1.0
        oc_target_tmp[:, 1] *= -1.0
        oc_rmin, oc_rmax, oc_cmin, oc_cmax = get_2dbbox(oc_target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale)

        oc_img = Image.open('{0}_color.png'.format(oc_frame))
        oc_depth = np.array(self.load_depth('{0}_depth.png'.format(oc_frame)))
        oc_mask = (cv2.imread('{0}_mask.png'.format(oc_frame))[:, :, 0] == 255)  # White is True and Black is False
        oc_img = np.array(oc_img)[:, :, :3]  # (3, 640, 480)
        oc_img = np.transpose(oc_img, (2, 0, 1)) # (3, 640, 480)
        oc_img = oc_img / 255.0
        oc_img = oc_img * (~oc_mask)
        oc_depth = oc_depth * (~oc_mask)

        oc_img = oc_img[:, oc_rmin:oc_rmax, oc_cmin:oc_cmax]
        oc_depth = oc_depth[oc_rmin:oc_rmax, oc_cmin:oc_cmax]
        oc_mask = oc_mask[oc_rmin:oc_rmax, oc_cmin:oc_cmax]
        return oc_img, oc_depth, oc_mask

    def divide_scale(self, scale, pts):
        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts

    def get_anchor_box(self, ori_bbox):
        bbox = ori_bbox
        limit = np.array(search_fit(bbox))
        num_per_axis = 5
        gap_max = num_per_axis - 1

        small_range = [1, 3]

        gap_x = (limit[1] - limit[0]) / float(gap_max)
        gap_y = (limit[3] - limit[2]) / float(gap_max)
        gap_z = (limit[5] - limit[4]) / float(gap_max)

        ans = []
        scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

        for i in range(0, num_per_axis):
            for j in range(0, num_per_axis):
                for k in range(0, num_per_axis):
                    ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

        ans = np.array(ans)
        scale = np.array(scale)

        ans = self.divide_scale(scale, ans)

        return ans, scale

    def change_to_scale(self, scale, cloud_fr, cloud_to):
        cloud_fr = self.divide_scale(scale, cloud_fr)
        cloud_to = self.divide_scale(scale, cloud_to)

        return cloud_fr, cloud_to


    def enlarge_bbox(self, target):

        limit = np.array(search_fit(target))
        longest = max(limit[1]-limit[0], limit[3]-limit[2], limit[5]-limit[4])
        longest = longest * 1.3

        scale1 = longest / (limit[1]-limit[0])
        scale2 = longest / (limit[3]-limit[2])
        scale3 = longest / (limit[5]-limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def get_pose(self, choose_frame, choose_obj):
        has_pose = []
        pose = {}
        if self.mode == "train":
            input_file = open('{0}_pose.txt'.format(choose_frame.replace("data/", "data_pose/")), 'r')
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                if len(input_line) == 1:
                    idx = int(input_line[0])
                    has_pose.append(idx)
                    pose[idx] = []
                    for i in range(4):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        pose[idx].append([float(input_line[0]), float(input_line[1]), float(input_line[2]), float(input_line[3])])
            input_file.close()
        if self.mode == "val":
            with open('{0}/data/gts/real_test/results_real_test_{1}_{2}.pkl'.format(self.root, choose_frame.split("/")[-2], choose_frame.split("/")[-1]), 'rb') as f:
                nocs_data = cPickle.load(f)
            for idx in range(nocs_data['gt_RTs'].shape[0]):
                idx = idx + 1
                pose[idx] = nocs_data['gt_RTs'][idx-1]
                pose[idx][:3, :3] = pose[idx][:3, :3] / np.cbrt(np.linalg.det(pose[idx][:3, :3]))
                z_180_RT = np.zeros((4, 4), dtype=np.float32)
                z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                z_180_RT[3, 3] = 1
                pose[idx] = z_180_RT @ pose[idx]
                pose[idx][:3,3] = pose[idx][:3,3] * 1000

        input_file = open('{0}_meta.txt'.format(choose_frame), 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            if input_line[-1] == choose_obj:
                ans = pose[int(input_line[0])]
                ans_idx = int(input_line[0])
                break
        input_file.close()

        ans = np.array(ans)
        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()

        return ans_r, ans_t, ans_idx


    # choose_obj: the code of the object, choose_frame: the samples prefix.
    def get_frame(self, choose_frame, choose_obj, syn_or_real):

        if syn_or_real:
            mesh_bbox = []
            input_file = open('{0}/model_pts/{1}.txt'.format(self.root, choose_obj), 'r')
            for i in range(8):
                input_line = input_file.readline()
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                mesh_bbox.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
            mesh_bbox = np.array(mesh_bbox)

            mesh_pts = []
            input_file = open('{0}/model_pts/{1}.xyz'.format(self.root, choose_obj), 'r')
            for i in range(2800):
                input_line = input_file.readline()
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                mesh_pts.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
            mesh_pts = np.array(mesh_pts)

            mesh_bbox = self.enlarge_bbox(copy.deepcopy(mesh_bbox))


        img = Image.open('{0}_color.png'.format(choose_frame))
        depth = np.array(self.load_depth('{0}_depth.png'.format(choose_frame)))

        target_r, target_t, idx = self.get_pose(choose_frame, choose_obj)

        if syn_or_real:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1
        cam_scale = 1.0

        if syn_or_real:
            target = []
            input_file = open('{0}_bbox.txt'.format(choose_frame.replace("data/", "data_pose/")), 'r')
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                if len(input_line) == 1 and int(input_line[0]) == idx:
                    for i in range(8):
                        input_line = input_file.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
                    break
            input_file.close()
            target = np.array(target)
        else:
            target = []
            input_file = open('{0}/model_scales/{1}.txt'.format(self.root, choose_obj), 'r')
            for i in range(8):
                input_line = input_file.readline()
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                input_line = input_line.split(' ')
                target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            input_file.close()
            target = np.array(target)

        target = self.enlarge_bbox(copy.deepcopy(target))

        delta = math.pi / 10.0
        noise_trans = 0.05
        r = euler_matrix(random.uniform(-delta, delta), random.uniform(-delta, delta), random.uniform(-delta, delta))[:3, :3]
        t = np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 1000.0

        target_tmp = target - (np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 3000.0)
        target_tmp = np.dot(target_tmp, target_r.T) + target_t
        target_tmp[:, 0] *= -1.0
        target_tmp[:, 1] *= -1.0
        rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale) # These four values is the boundaries of 2d bounding box.
        limit = search_fit(target)

        if self.add_noise:
            img = self.trancolor(img)

            if random.randint(1, 20) > 3:
                back_frame = random.sample(self.back_list, 1)[0]

                back_img = np.array(self.trancolor(Image.open(back_frame).resize((640, 480), Image.ANTIALIAS)))
                back_img = np.transpose(back_img, (2, 0, 1))

                mask = (cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 0] == 255)
                img = np.transpose(np.array(img), (2, 0, 1))
                # Here use the object from /data/train/.png to be foreground and the background from /train2017 to synethize a new image.
                img = img * (~mask) + back_img * mask

                img = np.transpose(img, (1, 2, 0))

                back_cate_id = random.sample([1, 2, 3, 4, 5, 6], 1)[0]
                back_depth_choose_obj = random.sample(self.real_obj_name_list[back_cate_id], 1)[0]
                back_choose_frame = random.sample(self.real_obj_list[back_cate_id][back_depth_choose_obj], 1)[0]# The background depth is random here.
                back_depth = np.array(self.load_depth('{0}_depth.png'.format(back_choose_frame)))

                ori_back_depth = back_depth * mask # Use image from /data/real_train/scene_n/_depth.png to synthize the new depth map.
                ori_depth = depth * (~mask)

                back_delta = ori_depth.flatten()[ori_depth.flatten() != 0].mean() - ori_back_depth.flatten()[ori_back_depth.flatten() != 0].mean()
                back_depth = back_depth + back_delta

                depth = depth * (~mask) + back_depth * mask

            else:
                img = np.array(img)
        else:
            img = np.array(img)

        mask_target = (cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 2] == idx)[rmin:rmax, cmin:cmax]
        choose = (mask_target.flatten() != False).nonzero()[0]
        if len(choose) == 0:
            return 0
        # [:, rmin:rmax, cmin:cmax]
        img = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        depth = depth[rmin:rmax, cmin:cmax]# Cropping depth map.
        img = img / 255.0

        if self.occlude == True and random.randint(0,20) >= 10:

            oc_id = random.sample(self.cate_set, 1)[0]
            oc_obj = random.sample(self.real_oc_name_set[oc_id], 1)[0]
            oc_frame = random.sample(self.real_oc_list[oc_obj], 1)[0]
            oc_img, oc_depth, oc_mask = self.get_occlusion(oc_obj, oc_frame, syn_or_real)
            oc_mask = oc_mask[:, :, np.newaxis]
            oc_img = np.transpose(oc_img, (1, 2, 0))  # (h, w, 3)

            oc_depth = oc_depth[:, :, np.newaxis]  # (h, w, 1)

            ratio = 1
            h = int(oc_img.shape[0]/ratio)
            w = int(oc_img.shape[1]/ratio)
            hl = random.randint(0, h)
            hr = random.randint(0, h)
            wl = random.randint(0, w)
            wr = random.randint(0, w)
            oc_img = np.pad(oc_img, ((hl, hr), (wl, wr), (0, 0)), 'constant', constant_values=(0, 0.))
            oc_depth = np.pad(oc_depth, ((hl, hr), (wl, wr), (0, 0)), 'constant', constant_values=(0., 0.))

            np.set_printoptions(threshold=np.inf)
            oc_mask = np.pad(oc_mask, ((hl, hr), (wl, wr), (0, 0)), 'constant', constant_values=(1. , 1.))
            oc_img = resize(oc_img, (img.shape[1], img.shape[2], img.shape[0]))
            oc_mask = resize(oc_mask,(img.shape[1], img.shape[2], 1))
            oc_depth = resize(oc_depth, (img.shape[1], img.shape[2], 1), preserve_range=True)
            oc_img = np.transpose(oc_img, (2, 0, 1)) #(3, h, w)
            oc_depth = np.transpose(oc_depth, (2, 0, 1))
            oc_depth = np.squeeze(oc_depth, 0)#(h, w)
            oc_mask = np.transpose(oc_mask, (2, 0, 1))
            oc_mask = np.squeeze(oc_mask, 0)
            oc_mask = (oc_mask == 1.)
            img = img * oc_mask + oc_img

            mask = (cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 0] == 255)[rmin:rmax, cmin:cmax]
            r_depth = depth * (~mask) # Masked object's depth
            maxm = -1000
            for (indi, i) in enumerate(r_depth):
                for (indj, j) in enumerate(i):
                    if j == 0 or oc_depth[indi][indj] == 0:
                        continue
                    if j >= oc_depth[indi][indj]:
                        continue
                    maxm = max((oc_depth[indi][indj] - j), maxm)
            if maxm != -1000:
                oc_depth = oc_depth - maxm
                oc_depth = oc_depth * (~oc_mask)
            depth = depth * oc_mask + oc_depth
            # plt.subplot(2, 1, 1)
            # plt.imshow(np.transpose(img, (1, 2, 0)))
            # plt.subplot(2, 1, 2)
            # plt.imshow(np.transpose(img_ori, (1, 2, 0)) )
            # plt.show()
            #
            # plt.subplot(2, 1, 1)
            # plt.imshow(depth)
            # plt.subplot(2, 1, 2)
            # plt.imshow(depth_ori)
            # plt.show()


        # Deep copy
        img_r = img.copy()
        img_r = np.transpose(img_r, (1, 2, 0))
        img_r = resize(img_r, (320, 320, 3))
        img_r = np.transpose(img_r, (2, 0, 1))

        choose = (depth.flatten() > -1000.0).nonzero()[0] # Choose is a 1-d list whose size is depend on the number of non zero value in it, which indicate the index of non zero value in depth.
        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32) # The purpose of this step is to exclude the zero value in depth map.
        # self.xmap and self.ymap are 640 x 480 0 value map.
        # This step is create x,y maps mask.
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        ## Create point cloud from depth map after cropping. Its first dim size is equal to choose.
        pt2 = depth_masked / cam_scale # 1.0
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

        cloud = np.dot(cloud - target_t, target_r)
        cloud = np.dot(cloud, r.T) + t

        choose_temp = (cloud[:, 0] > limit[0]) * (cloud[:, 0] < limit[1]) * (cloud[:, 1] > limit[2]) * (cloud[:, 1] < limit[3]) * (cloud[:, 2] > limit[4]) * (cloud[:, 2] < limit[5])

        choose = ((depth.flatten() != 0.0) * choose_temp).nonzero()[0]
        if len(choose) == 0:
            return 0
        if len(choose) > self.num_pt: # The num_pt is a predefined value which is 500 and it means the maximum 3-d points needed to be considered for one object. This value is equal to the number of points in point cloud and choose list.
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)
        choose = np.array([choose])

        cloud = np.dot(cloud - target_t, target_r)
        cloud = np.dot(cloud, r.T) + t # This step is to randomly add noise.

        t = t / 1000.0
        cloud = cloud / 1000.0
        target = target / 1000.0
        
        if syn_or_real:
            cloud = cloud + np.random.normal(loc=0.0, scale=0.003, size=cloud.shape)

        if syn_or_real:
            return img_r, img, choose, cloud, r, t, target, mesh_pts, mesh_bbox, mask_target
        else:
            return img_r, img, choose, cloud, r, t, target, mask_target


    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr
        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale


    def __getitem__(self, index):
        syn_or_real = (random.randint(1, 20) < 15) # True(syn): 3/4, False(real): 1/4

        if self.mode == 'val':
            syn_or_real = False
        img_fr_r_set = []
        img_to_r_set = []
        img_fr_set = []
        img_to_set = []
        choose_fr_set = []
        choose_to_set = []
        cloud_fr_set = []
        cloud_to_set = []
        r_fr_set = []
        r_to_set = []
        t_fr_set = []
        t_to_set = []
        target_set = []
        scale_set = []
        anchor_set = []
        mesh_set = []

        if syn_or_real:
            # Synthetic data 3/4

            choose_obj = random.sample(self.obj_name_list[self.cate_id], 1)[0]# Select one object.

            for w in range(self.w_size):
                while 1:
                    try:

                        # self.cate_id is the category to train, default 5. Randomly sample one obj from the metioned category.
                        # choose_obj = random.sample(self.obj_name_list[self.cate_id], 1)[0] # 1a9e1fb2a51ffd065b07a27512172330 this code indicate the same obj with different pose under the same category.
                        choose_frame = random.sample(self.obj_list[self.cate_id][choose_obj], 2)# Path like data/train/06652/0003

                        img_fr_r, img_fr, choose_fr, cloud_fr, r_fr, t_fr, target_fr, mesh_pts_fr, mesh_bbox_fr, mask_target = self.get_frame(choose_frame[0], choose_obj, syn_or_real)

                        if np.max(abs(target_fr)) > 1.0:
                            continue

                        img_to_r, img_to, choose_to, cloud_to, r_to, t_to, target_to, _, _, _ = self.get_frame(choose_frame[1], choose_obj, syn_or_real)
                        if np.max(abs(target_to)) > 1.0:
                            continue

                        target, scale_factor = self.re_scale(target_fr, target_to)
                        target_mesh_fr, scale_factor_mesh_fr = self.re_scale(target_fr, mesh_bbox_fr)

                        cloud_to = cloud_to * scale_factor
                        mesh = mesh_pts_fr * scale_factor_mesh_fr
                        t_to = t_to * scale_factor


                        # img_fr = self.norm(torch.from_numpy(img_fr.astype(np.float32))).numpy()
                        # img_to = self.norm(torch.from_numpy(img_to.astype(np.float32))).numpy()
                        #
                        # img_fr_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32))).numpy()
                        # img_to_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32))).numpy()

                        img_fr = self.norm(torch.from_numpy(img_fr.astype(np.float32)))
                        img_to = self.norm(torch.from_numpy(img_to.astype(np.float32)))

                        img_fr_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32)))
                        img_to_r = self.norm(torch.from_numpy(img_to_r.astype(np.float32)))

                        img_fr_r_set.append(img_fr_r)
                        img_to_r_set.append(img_to_r)
                        img_fr_set.append(img_fr)
                        img_to_set.append(img_to)
                        choose_fr_set.append(choose_fr)
                        choose_to_set.append(choose_to)
                        r_fr_set.append(r_fr)
                        r_to_set.append(r_to)
                        t_fr_set.append(t_fr)
                        cloud_fr_set.append(cloud_fr)
                        t_to_set.append(t_to)
                        cloud_to_set.append(cloud_to)
                        target_set.append(target)
                        break
                    except:

                        continue

        else:
            # Real data from video sequence, 1/4
            choose_obj = random.sample(self.real_obj_name_list[self.cate_id], 1)[0]
            for w in range(self.w_size):
                while 1:
                    try:

                        choose_frame = random.sample(self.real_obj_list[self.cate_id][choose_obj], 2)

                        img_fr_r, img_fr, choose_fr, cloud_fr, r_fr, t_fr, target, _ = self.get_frame(choose_frame[0], choose_obj, syn_or_real)
                        img_to_r, img_to, choose_to, cloud_to, r_to, t_to, target, _ = self.get_frame(choose_frame[1], choose_obj, syn_or_real)
                        if np.max(abs(target)) > 1.0:
                            continue

                        # img_fr = self.norm(torch.from_numpy(img_fr.astype(np.float32))).numpy()
                        # img_to = self.norm(torch.from_numpy(img_to.astype(np.float32))).numpy()
                        #
                        # img_fr_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32))).numpy()
                        # img_to_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32))).numpy()

                        img_fr = self.norm(torch.from_numpy(img_fr.astype(np.float32)))
                        img_to = self.norm(torch.from_numpy(img_to.astype(np.float32)))

                        img_fr_r = self.norm(torch.from_numpy(img_fr_r.astype(np.float32)))
                        img_to_r = self.norm(torch.from_numpy(img_to_r.astype(np.float32)))

                        img_fr_r_set.append(img_fr_r)
                        img_to_r_set.append(img_to_r)
                        img_fr_set.append(img_fr)
                        img_to_set.append(img_to)
                        choose_fr_set.append(choose_fr)
                        choose_to_set.append(choose_to)
                        r_fr_set.append(r_fr)
                        r_to_set.append(r_to)
                        t_fr_set.append(t_fr)
                        cloud_fr_set.append(cloud_fr)
                        t_to_set.append(t_to)
                        cloud_to_set.append(cloud_to)
                        target_set.append(target)
                        break
                    except:

                        continue

        class_gt = np.array([self.cate_id-1])

        #anchor_box, scale = self.get_anchor_box(target)
        for i, cloud in enumerate(cloud_to_set):
            anchor_box, scale = self.get_anchor_box(target_set[i])
            anchor_set.append(anchor_box)
            scale_set.append(scale)
            mesh_set.append(self.mesh * scale)
            cloud_fr_set[i], cloud_to_set[i]=self.change_to_scale(scale, cloud_fr_set[i], cloud_to_set[i])
        # cloud_fr, cloud_to = self.change_to_scale(scale, cloud_fr, cloud_to)

        # mesh = self.mesh * scale
        # torch.FloatTensor(img_fr_r_set), \
        # torch.FloatTensor(img_fr_set), \


        return img_fr_r_set, \
               img_fr_set, \
               torch.LongTensor(np.array(choose_fr_set).astype(np.int32)), \
               torch.from_numpy(np.array(cloud_fr_set).astype(np.float32)), \
               torch.from_numpy(np.array(r_fr_set).astype(np.float32)), \
               torch.from_numpy(np.array(t_fr_set).astype(np.float32)), \
               img_to_r_set ,\
               img_to_set, \
               torch.LongTensor(np.array(choose_to_set).astype(np.int32)), \
               torch.from_numpy(np.array(cloud_to_set).astype(np.float32)), \
               torch.from_numpy(np.array(r_to_set).astype(np.float32)), \
               torch.from_numpy(np.array(t_to_set).astype(np.float32)), \
               torch.from_numpy(np.array(mesh_set).astype(np.float32)), \
               torch.from_numpy(np.array(anchor_set).astype(np.float32)), \
               torch.from_numpy(np.array(scale_set).astype(np.float32)), \
               torch.LongTensor(class_gt.astype(np.int32)), \


    def __len__(self):
        return self.length


border_list = [-1, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_2dbbox(cloud, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale):
    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000
    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 480:
        rmax = 479
    if cmax >= 640:
        cmax = 639

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
        
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt

    if ((rmax-rmin) in border_list) and ((cmax-cmin) in border_list):
        return rmin, rmax, cmin, cmax
    else:
        return 0


def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]
