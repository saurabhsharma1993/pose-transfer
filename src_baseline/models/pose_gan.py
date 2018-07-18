import torch
import torch.nn as nn
import os
import itertools
import torch.nn as nn
from torch.autograd import Variable
from utils import pose_utils
from models.networks import Generator, Stacked_Generator, Discriminator, gaussian_weights_init, print_network, xavier_weights_init

class Pose_GAN(nn.Module):
  def __init__(self, opt):
    super(Pose_GAN, self).__init__()

    # load generator and discriminator models
    # adding extra layers for larger image size
    if(opt.checkMode == 0):
      nfilters_decoder = (512, 512, 512, 256, 128, 3) if max(opt.image_size) < 256 else (512, 512, 512, 512, 256, 128, 3)
      nfilters_encoder = (64, 128, 256, 512, 512, 512) if max(opt.image_size) < 256 else (64, 128, 256, 512, 512, 512, 512)
    else:
      nfilters_decoder = (128, 3) if max(opt.image_size) < 256 else (256, 128, 3)
      nfilters_encoder = (64, 128) if max(opt.image_size) < 256 else (64, 128, 256)

    if (opt.use_input_pose):
      input_nc = 3 + 2*opt.pose_dim
    else:
      input_nc = 3 + opt.pose_dim

    self.num_stacks = opt.num_stacks
    self.batch_size = opt.batch_size
    self.pose_dim = opt.pose_dim
    if(opt.gen_type=='stacked'):
      self.gen = Stacked_Generator(input_nc, opt.num_stacks, opt.pose_dim, nfilters_encoder, nfilters_decoder, use_input_pose=opt.use_input_pose)
    elif(opt.gen_type=='baseline'):
      self.gen = Generator(input_nc, nfilters_encoder, nfilters_decoder, use_input_pose=opt.use_input_pose)
    else:
      raise Exception('Invalid gen_type')
    # discriminator also sees the output image for the target pose
    self.disc = Discriminator(input_nc + 3, use_input_pose=opt.use_input_pose, checkMode=opt.checkMode)
    print('---------- Networks initialized -------------')
    print_network(self.gen)
    print_network(self.disc)
    print('-----------------------------------------------')
    # Setup the optimizers
    lr = opt.learning_rate
    self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))

    # Network weight initialization
    self.gen.cuda()
    self.disc.cuda()
    self.disc.apply(xavier_weights_init)
    self.gen.apply(xavier_weights_init)

    # Setup the loss function for training
    self.ll_loss_criterion = torch.nn.L1Loss()

  # add code for intermediate supervision for the interpolated poses using pretrained pose-estimator
  def gen_update(self, input, target, interpol_pose, opt):
    self.gen.zero_grad()

    if(opt['gen_type']=='stacked'):
      outputs_gen = self.gen(input, interpol_pose)
      out_gen = outputs_gen[-1]
    else:
      out_gen = self.gen(input)
      outputs_gen = []

    inp_img, inp_pose, out_pose = pose_utils.get_imgpose(input, opt['use_input_pose'], opt['pose_dim'])

    inp_dis = torch.cat([inp_img, inp_pose, out_gen, out_pose], dim=1)
    out_dis = self.disc(inp_dis)

    # computing adversarial loss
    for it in range(out_dis.shape[0]):
      out = out_dis[it, :]
      all_ones = Variable(torch.ones((out.size(0))).cuda())
      if it==0:
        # ad_loss = nn.functional.binary_cross_entropy(out, all_ones)
        ad_loss = -torch.mean(torch.log(out + 1e-7))
      else:
        # ad_loss += nn.functional.binary_cross_entropy(out, all_ones)
        ad_loss += -torch.mean(torch.log(out + 1e-7)
                               )
    ll_loss = self.ll_loss_criterion(out_gen, target)
    ad_loss = ad_loss * opt['gan_penalty_weight'] / self.batch_size
    ll_loss = ll_loss * opt['l1_penalty_weight']
    total_loss = ad_loss + ll_loss
    total_loss.backward()
    self.gen_opt.step()
    self.gen_ll_loss = ll_loss.item()
    self.gen_ad_loss = ad_loss.item()
    self.gen_total_loss = total_loss.item()
    return out_gen, outputs_gen, [self.gen_total_loss, self.gen_ll_loss, self.gen_ad_loss ]

  def dis_update(self, input, target, interpol_pose, real_inp, real_target, opt):
    self.disc.zero_grad()

    if (opt['gen_type'] == 'stacked'):
      out_gen = self.gen(input, interpol_pose)
      out_gen = out_gen[-1]
    else:
      out_gen = self.gen(input)

    inp_img, inp_pose, out_pose = pose_utils.get_imgpose(input, opt['use_input_pose'], opt['pose_dim'])

    fake_disc_inp = torch.cat([inp_img, inp_pose, out_gen, out_pose], dim=1)
    r_inp_img, r_inp_pose, r_out_pose = pose_utils.get_imgpose(real_inp, opt['use_input_pose'], opt['pose_dim'])
    real_disc_inp = torch.cat([r_inp_img, r_inp_pose, real_target, r_out_pose], dim=1)
    data_dis = torch.cat((real_disc_inp, fake_disc_inp), 0)
    res_dis = self.disc(data_dis)

    for it in range(res_dis.shape[0]):
      out = res_dis[it,:]
      if(it<opt['batch_size']):
        out_true_n = out.size(0)
        # real inputs should be 1
        # all1 = Variable(torch.ones((out_true_n)).cuda())
        if it == 0:
          # ad_true_loss = nn.functional.binary_cross_entropy(out, all1)
          ad_true_loss = -torch.mean(torch.log(out + 1e-7))
        else:
          # ad_true_loss += nn.functional.binary_cross_entropy(out, all1)
          ad_true_loss += -torch.mean(torch.log(out + 1e-7))
      else:
        out_fake_n = out.size(0)
        # fake inputs should be 0, appear after batch_size iters
        # all0 = Variable(torch.zeros((out_fake_n)).cuda())
        if it == opt['batch_size']:
          ad_fake_loss = -torch.mean(torch.log(1- out + 1e-7))
        else:
          ad_fake_loss += -torch.mean(torch.log(1 - out + 1e-7))

    ad_true_loss = ad_true_loss*opt['gan_penalty_weight']/self.batch_size
    ad_fake_loss = ad_fake_loss*opt['gan_penalty_weight']/self.batch_size
    ad_loss = ad_true_loss + ad_fake_loss
    loss = ad_loss
    loss.backward()
    self.disc_opt.step()
    self.dis_total_loss = loss.item()
    self.dis_true_loss = ad_true_loss.item()
    self.dis_fake_loss = ad_fake_loss.item()
    return [self.dis_total_loss , self.dis_true_loss , self.dis_fake_loss ]

  def resume(self, save_dir):
    last_model_name = pose_utils.get_model_list(save_dir,"gen")
    if last_model_name is None:
      return 1
    self.gen.load_state_dict(torch.load(last_model_name))
    epoch = int(last_model_name[-7:-4])
    print('Resume gen from epoch %d' % epoch)
    last_model_name = pose_utils.get_model_list(save_dir, "dis")
    if last_model_name is None:
      return 1
    epoch = int(last_model_name[-7:-4])
    self.disc.load_state_dict(torch.load(last_model_name))
    print('Resume disc from epoch %d' % epoch)
    return epoch

  def save(self, save_dir, epoch):
    gen_filename = os.path.join(save_dir, 'gen_{0:03d}.pkl'.format(epoch))
    disc_filename = os.path.join(save_dir, 'disc_{0:03d}.pkl'.format(epoch))
    torch.save(self.gen.state_dict(), gen_filename)
    torch.save(self.disc.state_dict(), disc_filename)

  def normalize_image(self, x):
    return x[:,0:3,:,:]

