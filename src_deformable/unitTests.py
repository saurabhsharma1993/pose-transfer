import sys

# shitty thing overriding src code in current dir
if('/home/anurag/multiperson/detectron/lib' in sys.path):
    sys.path.remove('/home/anurag/multiperson/detectron/lib')

import matplotlib
matplotlib.use('Agg')
import pylab as plt
import torch
from torch.autograd import Variable
import skimage.transform as transform
import numpy as np
import cv2
import time
import datetime
from opts import opts
# import tqdm
from datasets.PoseTransfer_Dataset import PoseTransfer_Dataset
from utils import pose_utils
from models.networks import Deformable_Generator, Discriminator, torch_summarize, keras_to_pytorch
from models.pose_gan import DeformablePose_GAN
# from utils.pose_transform import AffineTransformLayer
import sys

# <--------------------------------------------------->
# <--------------------------------------------------->
# daata loader, used everywhere in number of unit tests

# opt = opts().parse()
#
# disc_loader = torch.utils.data.DataLoader(
#     PoseTransfer_Dataset(vars(opt), 'train'),
#     batch_size=4,
#     shuffle=False,
# )
#
# disc_iter = disc_loader.__iter__()

# <--------------------------------------------------->
# <--------------------------------------------------->
# unit tests for visualization and Affine layer

# input, target, interpol_pose = disc_iter.next()
# # print(batch.shape)
# dis_images = pose_utils.display(input, target, opt.use_input_pose, opt.pose_dim)
#
# interpol_pose = pose_utils.postProcess(interpol_pose)
# result = []
# for i in range(opt.num_stacks):
#     pose_batch = interpol_pose[:, :, :, i * opt.pose_dim:(i + 1) * opt.pose_dim]
#     pose_images = np.array([pose_utils.draw_pose_from_map(pose.numpy(), opt.pose_dim)[0] for pose in pose_batch])
#     result.append(pose_images)
# interpol_pose = np.concatenate(result, axis=0)
# interpol_pose = pose_utils.make_grid(torch.from_numpy(interpol_pose), None, row=opt.batch_size, col=opt.num_stacks)
#
#
# plt.imsave('test_interpol.png', interpol_pose , cmap=plt.cm.gray)
# plt.imsave('test_image_pose.png', dis_images, cmap=plt.cm.gray)

# input, target, warps, masks = disc_iter.next()
# warps = warps[:,:,:-1].float()
# masks = masks.float()

# <------------------------------------------------------>
# <--------------------------------------------------->
# test 1 for affinetransform layer

#
# index = 10
# for _ in range(index):
#     input, target, warps, masks = disc_iter.next()
# inp_img, inp_pose, tg_pose = pose_utils.get_imgpose(input, True, 16)
# print(inp_img[0].shape)
# warp_skip = 'mask'
# warps = warps.float()
# masks = masks.float()
# input.requires_grad = True
# image_size = input.shape[2:]

# gradcheck for affinelayer
# affine_model = AffineTransformLayer(10 if warp_skip == 'mask' else 1, image_size, warp_skip)
from torch.autograd import gradcheck
# test = gradcheck(affine_model, (input.cuda(), warps.float().cuda(), masks.cuda()), eps=1e-6, atol=1e-4)
# print(test)

# out = AffineTransformLayer(10 if warp_skip == 'mask' else 1, image_size, warp_skip)(inp_img, warps, masks)
# inp_img = pose_utils.postProcess(pose_utils._deprocess_image(inp_img))
# target = pose_utils.postProcess(pose_utils._deprocess_image(target))
# out = pose_utils.postProcess(pose_utils._deprocess_image(out.data.cpu()))
# print(torch.min(out), torch.max(out))
# print(torch.min(inp_img), torch.max(inp_img))
# img = pose_utils.make_grid(torch.cat([inp_img, out, target], dim=0), None, row=opt.batch_size, col=3, order=0)
# plt.imsave('test_warp.png', img , cmap=plt.cm.gray)
# # #
# def _deprocess_image(image):
#     return (255 * (image + 1) / 2).astype(np.uint8)

# <------------------------------------------------------>
# <--------------------------------------------------->
# test 2 for cv2 warp affine and skimage affine

# import cv2
# input, target, warps, masks = disc_iter.next()
# inp_img, inp_pose, tg_pose = pose_utils.get_imgpose(input, True, 16)
# image = np.transpose(inp_img[0].numpy(), [1,2,0])
# # print(image.shape)
# target_image = np.transpose(target[0], [1,2,0])
# # M = np.reshape(warps[0,8,:6].numpy(), [2,3])
# # M_sk = np.reshape(warps[0,8].numpy(), [3,3])
#
# M = np.reshape(warps[0,0,:6].numpy(), [2,3])
# M_sk = np.reshape(warps[0,0].numpy(), [3,3])
#
#
# warp_img = cv2.warpAffine(image,M,(224,224))
# warp_img_sk = transform.warp(image,inverse_map=M_sk)
# if(np.min(warp_img)==0 and np.max(warp_img)==0):
#     warp_img[...] = -1
# # print(warp_img.shape)
# # print(target_image.shape)
#
# # cv2 warpAffine works on multiple channel images . . method verified by observing warped images
# res = np.concatenate([_deprocess_image(image),_deprocess_image(warp_img), _deprocess_image(target_image.numpy())], axis=0)
# plt.imsave('test_warp_cv.png', res , cmap=plt.cm.gray)
# res = np.concatenate([_deprocess_image(image),_deprocess_image(warp_img_sk), _deprocess_image(target_image.numpy())], axis=0)
# plt.imsave('test_warp_sk.png', res , cmap=plt.cm.gray)

# <------------------------------------------------------>
# <--------------------------------------------------->
# check for autograd backprop through affine layer

# opt = opts().parse()
# model = DeformablePose_GAN(opt)
# with torch.autograd.profiler.profile() as prof:
#   out = model.gen(Variable(input.cuda()), Variable(warps.cuda()), Variable(masks.cuda()))
#   grad_fn = out.grad_fn
#   while(grad_fn!=None):
#       print(grad_fn)
#       grad_fn = grad_fn.next_functions[0][0]
# # print(prof)

# import torch.nn.functional as F
#
# input, target, warps, masks = disc_iter.next()
# inp_img, inp_pose, tg_pose = pose_utils.get_imgpose(input, True, 16)
# # transforms = torch.ones([1,2,3]).float()
# # transforms = torch.Tensor([[[1,0,0], [0,1,0]]]).float()
# transforms = warps[0,6,:6].view(-1,2,3).float()#/224
# transforms[:,:,2] /= 224
#
# inp_img = inp_img[0].unsqueeze(0)
# target = target[0].unsqueeze(0)
# grid = F.affine_grid(transforms,inp_img.shape)
# print(torch.max(grid), torch.min(grid), torch.mean(grid))
# warped_map = F.grid_sample(inp_img, grid)
# warped_map = pose_utils.postProcess(pose_utils._deprocess_image(warped_map))
#
# inp_img = pose_utils.postProcess(pose_utils._deprocess_image(inp_img))
# target = pose_utils.postProcess(pose_utils._deprocess_image(target))
#
# img = pose_utils.make_grid(torch.cat([inp_img, warped_map, target], dim=0), None, row=1, col=3, order=1)
#
# plt.imsave('test_warp_pytorch.png', img, cmap=plt.cm.gray)

# pose_utils.Feature_Extractor()

# <------------------------------------------------------>
# <--------------------------------------------------->
# test for pose estimator

# index = 10
# for _ in range(index):
#     input, target, warps, masks = disc_iter.next()
# inp_img, inp_pose, tg_pose = pose_utils.get_imgpose(input, True, 16)
# print(inp_img[0].shape)
# warp_skip = 'mask'
# warps = warps.float()
# masks = masks.float()
# input.requires_grad = True
# image_size = input.shape[2:]
#
# # pose_model = torch.load('pose_model.pth')
# # out_pose = pose_model(inp_img)
#
# inp_img = pose_utils.postProcess(pose_utils._deprocess_image(inp_img))
# target = pose_utils.postProcess(pose_utils._deprocess_image(target))
# out_pose = pose_utils.postProcess(pose_utils._deprocess_image(out.data.cpu()))
# img = pose_utils.make_grid(torch.cat([inp_img, out_pose, target], dim=0), None, row=opt.batch_size, col=3, order=0)
# plt.imsave('test_warp.png', img , cmap=plt.cm.gray)

# <------------------------------------------------------>
# <--------------------------------------------------->
# load images at particular index and initialzie model

# index = 2
# for _ in range(index):
#      input, target, warps, masks = disc_iter.next()
#
# model = DeformablePose_GAN(opt).cuda()
# print(torch_summarize(model))

# <------------------------------------------------------>
# <--------------------------------------------------->
# covnert mdoel from keras and save the model

# from keras.models import load_model
# import sys
# sys.path.append('../../pose-gan-clean/pose-gan-h36m-fg/')
# from conditional_gan import CGAN
# from pose_transform_dummy import AffineTransformLayer
# appending sys path to load keras model luibratries

# config = --dataset fasion --pose_dim 18 --data_Dir ../../pose-gan-clean/pose-gan/data/

# keras_gen = load_model('../../pose-gan-clean/pose-gan-h36m-fg/output/full/h36m-fg-dsc/epoch_089_generator.h5',  custom_objects={'AffineTransformLayer': AffineTransformLayer})
#
#
# keras_weights = []
# for layer in keras_gen.layers:
#     keras_weights.append(layer.get_weights())
#
# model_layers = keras_gen.layers[4:]
# del keras_gen
#
# # 48 for fasion, 40 for h36m
# model_layers_app = [model_layers[2*index] for index in range(20)]
# model_layers_pose = [model_layers[2*index+1] for index in range(20)]
# model_layers[:20] = model_layers_app
# model_layers[20:40] = model_layers_pose
#
# pytorch_gen = model.gen
# pytorch_gen, _ = keras_to_pytorch(pytorch_gen, model_layers, 0)
#
# print(torch_summarize(pytorch_gen))
# torch.save(pytorch_gen.state_dict(), "gen.pkl")

# <------------------------------------------------------>
# <--------------------------------------------------->
# load gen model and evalautte qualitatively


# pytorch_gen = model.gen
# pytorch_gen.load_state_dict(torch.load('gen.pkl'))
# pytorch_gen.eval()
#
# out = pytorch_gen(input.cuda(), warps.float().cuda(), masks.cuda())
# images = pose_utils.display(input, target, out.data.cpu(), opt.use_input_pose, opt.pose_dim)
# plt.imsave('test_defo_pytorch_model.png', images, cmap=plt.cm.gray)


# <------------------------------------------------------>
# <--------------------------------------------------->
# convert disc model


# keras_disc = load_model('../../pose-gan-clean/pose-gan-h36m-fg/output/full/h36m-fg-dsc/epoch_089_discriminator.h5',  custom_objects={'AffineTransformLayer': AffineTransformLayer})
# pytorch_disc = model.disc
# pytorch_disc, _ = keras_to_pytorch(pytorch_disc, keras_disc.layers, 0)
# torch.save(pytorch_disc.state_dict(), "disc.pkl")