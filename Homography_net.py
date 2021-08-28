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
from libs.warpn import homo_map
class Homo_net(nn.Module):
    def __init__(self,  opt):
        super(Homo_net, self).__init__()
        self.homo_warping = homo_map()
    def forward(self, img1, img2):
        r_img1, r_img2 =  self.homo_warping(img1, img2) #(1, 3, 32, 480, 640)

        return r_img1, r_img2

