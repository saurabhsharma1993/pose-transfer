import matplotlib
matplotlib.use('Agg')
import pylab as plt
import torch
import numpy as np
import cv2
import time
import datetime
from opts import opts
import tqdm
from datasets.PoseTransfer_Dataset import PoseTransfer_Dataset
from utils import pose_utils
from models.networks import Generator, Discriminator

opt = opts().parse()

disc_loader = torch.utils.data.DataLoader(
    PoseTransfer_Dataset(vars(opt), 'train'),
    batch_size=opt.batch_size,
    shuffle=True,
)

disc_iter = disc_loader.__iter__()
input, target, interpol_pose = disc_iter.next()
# print(batch.shape)
dis_images = pose_utils.display(input, target, opt.use_input_pose, opt.pose_dim)

interpol_pose = pose_utils.postProcess(interpol_pose)
result = []
for i in range(opt.num_stacks):
    pose_batch = interpol_pose[:, :, :, i * opt.pose_dim:(i + 1) * opt.pose_dim]
    pose_images = np.array([pose_utils.draw_pose_from_map(pose.numpy(), opt.pose_dim)[0] for pose in pose_batch])
    result.append(pose_images)
interpol_pose = np.concatenate(result, axis=0)
interpol_pose = pose_utils.make_grid(torch.from_numpy(interpol_pose), None, row=opt.batch_size, col=opt.num_stacks)


plt.imsave('test_interpol.png', interpol_pose , cmap=plt.cm.gray)
plt.imsave('test_image_pose.png', dis_images, cmap=plt.cm.gray)
