import sys

# shitty thing overriding src code in current dir
if('/home/anurag/multiperson/detectron/lib' in sys.path):
    sys.path.remove('/home/anurag/multiperson/detectron/lib')

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import os
import sys
import torch
import numpy as np
import cv2
import time
import datetime
from opts import opts
# from tqdm import tqdm
from datasets.PoseTransfer_Dataset import PoseTransfer_Dataset
from models.networks import Deformable_Generator, Discriminator
from models.pose_gan import DeformablePose_GAN
from utils import pose_utils
# from tqdm import tqdm

opt = opts().parse()

dataset = PoseTransfer_Dataset(vars(opt), split='test')
loader_test = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

test_iter = loader_test.__iter__()

model = DeformablePose_GAN(opt).cuda()
epoch = model.resume(opt.checkpoints_dir)

num_iterations = dataset._pairs_file_test.shape[0]//opt.batch_size
for it in range(num_iterations):
    if(it%50==0):
        sys.stdout.flush()
        print(it / num_iterations)
    input, target, warps, masks = test_iter.__next__()
    if (opt.gen_type == 'baseline'):
        out = model.gen(input.cuda(), warps.float().cuda(), masks.cuda())
        images = pose_utils.display(input, target, out.data.cpu(), opt.use_input_pose, opt.pose_dim)
    elif (opt.gen_type == 'stacked'):
        out = model.gen(input.cuda(), )
        images = pose_utils.display_stacked(input, target, out, opt.num_stacks, opt.use_input_pose, opt.pose_dim)
    else:
        raise Exception('Invalid gen type !!')
    title = "{0}.png".format(str(it).zfill(5))
    plt.imsave(os.path.join(opt.generated_images_dir, title), images, cmap=plt.cm.gray)