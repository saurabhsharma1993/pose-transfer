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
from models.networks import Generator, Discriminator, torch_summarize, keras_to_pytorch
from models.pose_gan import Pose_GAN
import sys
sys.path.append('../../pose-gan-clean/pose-gan-h36m-fg/')
from conditional_gan import CGAN
from pose_dataset import PoseHMDataset
from opt import cmd

opt = opts().parse()

# <--------------------------------------------------->
# intialzie dataloader

# disc_loader = torch.utils.data.DataLoader(
#     PoseTransfer_Dataset(vars(opt), 'train'),
#     batch_size=opt.batch_size,
#     shuffle=False,
# )
#
# disc_iter = disc_loader.__iter__()

# <------------------------------------------------------>
# visualize interpolated pose

#
# input, target, interpol_pose = disc_iter.next()
# print(batch.shape)
# dis_images = pose_utils.display(input, target, opt.use_input_pose, opt.pose_dim)

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


# <------------------------------------------------------>
# load images at particular index and initialzie model

# index = 20
# for _ in range(index):
#      input, target, _ = disc_iter.next()
# #
# model = Pose_GAN(opt).cuda()
# print(torch_summarize(model))

# out = model.gen(input.cuda())
# images = pose_utils.display(input, target, out.data.cpu(), opt.use_input_pose, opt.pose_dim)
# plt.imsave('test_model.png', images, cmap=plt.cm.gray)

from keras.models import load_model

# <------------------------------------------------------>
# covnert mdoel from keras and save the model

# config = --dataset fasion --pose_dim 18 --data_Dir ../../pose-gan-clean/pose-gan/data/

# keras_gen = load_model('../../pose-gan-clean/pose-gan/output/baseline-fasion/epoch_089_generator.h5')
# pytorch_gen = model.gen
# pytorch_gen, _ = keras_to_pytorch(pytorch_gen, keras_gen.layers, 0)
#
# del keras_gen
# del model


# keras_disc = load_model('../../pose-gan-clean/pose-gan/output/baseline-fasion/epoch_089_discriminator.h5')
# pytorch_disc = model.disc
# pytorch_disc, _ = keras_to_pytorch(pytorch_disc, keras_disc.layers, 0)

# <------------------------------------------------------>
# load the model

# pytorch_disc = pytorch_disc.cuda()
# torch.save(pytorch_disc.state_dict(), "disc.pkl")

# pytorch_gen = model.gen
# pytorch_gen.load_state_dict(torch.load('gen.pkl'))
# pytorch_gen.eval()

# keras_weights = []
# for layer in keras_gen.layers:
#     keras_weights.append(layer.get_weights())

# out = pytorch_gen(input.cuda())
# images = pose_utils.display(input, target, out.data.cpu(), opt.use_input_pose, opt.pose_dim)
# plt.imsave('test_pytorch_model.png', images, cmap=plt.cm.gray)

# <------------------------------------------------>
# ----> keras unit test

# args = cmd.args()
# args.dataset = 'fasion'
# generator  = load_model('../../pose-gan-clean/pose-gan/output/baseline-fasion/epoch_089_generator.h5')
# dataset = PoseHMDataset(test_phase=False, **vars(args))
#
# index = 11
# for _ in range(index):
#     generator_batch = dataset.next_generator_sample()
#
# import tensorflow as tf
# sess = tf.Session()
# with sess.as_default():
#     out_batch = generator.predict_on_batch(generator_batch)
#
# image_test = dataset.display(out_batch,generator_batch)
# plt.imsave('test_keras_model.png', image_test, cmap=plt.cm.gray)


# <------------------------------------------------------>
# test forward pass for both models

# import cv2
# pytorch_out = plt.imread('test_pytorch_model.png')
# keras_out = plt.imread('test_keras_model.png')
#
# pytorch_out = pytorch_out[:,:-256]
# keras_out = keras_out[:,:-256]
#
#
# f, axarr = plt.subplots(2,1)
# axarr[0] = plt.imshow(pytorch_out)
# plt.show()
# # cv2.waitKey(0)
# axarr[1] = plt.imshow(keras_out)
# plt.show()
# print(np.max(pytorch_out), np.max(keras_out))
# print(np.min(pytorch_out), np.min(keras_out))
# print(np.mean(pytorch_out), np.mean(keras_out))
# print(np.mean(np.absolute(pytorch_out-keras_out)))
# print(np.max(pytorch_out - keras_out))

# loss = torch.nn.L1Loss(torch.from_numpy(pytorch_out), torch.from_numpy(keras_out))
# print(loss)
# print(np.allclose(pytorch_out,keras_out))

# <------------------------------------------------------>
# test losses for both models

# batch_size = 4
# out_size = 49
#
# fake_out_dis = torch.from_numpy(np.random.rand(4,49))
# fake_out_dis.requires_grad = True
# out_dis = fake_out_dis
# for it in range(out_dis.shape[0]):
#     out = out_dis[it, :]
#     # all_ones = Variable(torch.ones((out.size(0))).cuda())
#     if it == 0:
#         # ad_loss = nn.functional.binary_cross_entropy(out, all_ones)
#         ad_loss = -torch.mean(torch.log(out + 1e-7))
#     else:
#         # ad_loss += nn.functional.binary_cross_entropy(out, all_ones)
#         ad_loss += -torch.mean(torch.log(out + 1e-7)
#                                )
# ad_loss = ad_loss * opt.gan_penalty_weight / batch_size
# print(ad_loss)
#
# from keras import backend as K
# import tensorflow as tf
# sess = tf.Session()
# with sess.as_default():
#     keras_loss = -K.mean(K.log(fake_out_dis.detach().numpy() + 1e-7)).eval()
# print(keras_loss)
