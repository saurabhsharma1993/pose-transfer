import torch
import torch.nn as nn
import os
import itertools
import torch.nn as nn
from torch.autograd import Variable
from utils import pose_utils
from models.networks import Deformable_Generator, Stacked_Generator, Discriminator, gaussian_weights_init, print_network, xavier_weights_init
from torchvision.models import vgg19

class DeformablePose_GAN(nn.Module):
  def __init__(self, opt):
    super(DeformablePose_GAN, self).__init__()

    # load generator and discriminator models
    # adding extra layers for larger image size
    nfilters_decoder = (512, 512, 512, 256, 128, 3) if max(opt.image_size) < 256 else (512, 512, 512, 512, 256, 128, 3)
    nfilters_encoder = (64, 128, 256, 512, 512, 512) if max(opt.image_size) < 256 else (64, 128, 256, 512, 512, 512, 512)

    if (opt.use_input_pose):
      input_nc = 3 + 2*opt.pose_dim
    else:
      input_nc = 3 + opt.pose_dim

    self.batch_size = opt.batch_size
    self.num_stacks = opt.num_stacks
    self.pose_dim = opt.pose_dim
    if(opt.gen_type=='stacked'):
      self.gen = Stacked_Generator(input_nc, opt.num_stacks, opt.image_size, opt.pose_dim, nfilters_encoder, nfilters_decoder, opt.warp_skip, use_input_pose=opt.use_input_pose)
      # hack to get better results
      pretrained_gen_path = '../exp/' + 'full_' + opt.dataset + '/models/gen_090.pkl'
      self.gen.generator.load_state_dict(torch.load(pretrained_gen_path))
      print("Loaded generator from pretrained model ")
    elif(opt.gen_type=='baseline'):
      self.gen = Deformable_Generator(input_nc, self.pose_dim, opt.image_size, nfilters_encoder, nfilters_decoder, opt.warp_skip, use_input_pose=opt.use_input_pose)
    else:
      raise Exception('Invalid gen_type')
    # discriminator also sees the output image for the target pose
    self.disc = Discriminator(input_nc + 3, use_input_pose=opt.use_input_pose)
    pretrained_disc_path = '../exp/' + 'full_' + opt.dataset + '/models/disc_090.pkl'
    print("Loaded discriminator from pretrained model ")
    self.disc.load_state_dict(torch.load(pretrained_disc_path))

    print('---------- Networks initialized -------------')
    # print_network(self.gen)
    # print_network(self.disc)
    print('-----------------------------------------------')
    # Setup the optimizers
    lr = opt.learning_rate
    self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))

    self.content_loss_layer = opt.content_loss_layer
    self.nn_loss_area_size = opt.nn_loss_area_size
    if self.content_loss_layer != 'none':
      self.content_model = vgg19(pretrained=True)
      # Setup the loss function for training
    # Network weight initialization
    self.gen.cuda()
    self.disc.cuda()
    self._nn_loss_area_size = opt.nn_loss_area_size
    # applying xavier_uniform, equivalent to glorot unigorm, as in Keras Defo GAN
    # skipping as models are pretrained
    # self.disc.apply(xavier_weights_init)
    # self.gen.apply(xavier_weights_init)
    self.ll_loss_criterion = torch.nn.L1Loss()

  # add code for intermediate supervision for the interpolated poses using pretrained pose-estimator
  def gen_update(self, input, target, other_inputs, opt):
    self.gen.zero_grad()

    if(opt['gen_type']=='stacked'):
      interpol_pose = other_inputs['interpol_pose']
      interpol_warps = other_inputs['interpol_warps']
      interpol_masks = other_inputs['interpol_masks']
      outputs_gen = self.gen(input, interpol_pose, interpol_warps, interpol_masks)
      out_gen = outputs_gen[-1]
    else:
      warps = other_inputs['warps']
      masks = other_inputs['masks']
      out_gen = self.gen(input,  warps, masks)
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
        ad_loss += -torch.mean(torch.log(out + 1e-7))

    if self.content_loss_layer != 'none':
      content_out_gen = pose_utils.Feature_Extractor(self.content_model, input=out_gen, layer_name=self.content_loss_layer )
      content_target = pose_utils.Feature_Extractor(self.content_model, input=target, layer_name=self.content_loss_layer )
      ll_loss = self.nn_loss(content_out_gen, content_target, self.nn_loss_area_size, self.nn_loss_area_size)
    else:
      ll_loss = self.ll_loss_criterion(out_gen, target)

    ad_loss = ad_loss*opt['gan_penalty_weight']/self.batch_size
    ll_loss = ll_loss*opt['l1_penalty_weight']
    total_loss = ad_loss + ll_loss
    total_loss.backward()
    self.gen_opt.step()
    self.gen_ll_loss = ll_loss.item()
    self.gen_ad_loss = ad_loss.item()
    self.gen_total_loss = total_loss.item()
    return out_gen, outputs_gen, [self.gen_total_loss, self.gen_ll_loss, self.gen_ad_loss]

  def dis_update(self, input, target, other_inputs, real_inp, real_target, opt):
    self.disc.zero_grad()

    if (opt['gen_type'] == 'stacked'):
      interpol_pose = other_inputs['interpol_pose']
      interpol_warps = other_inputs['interpol_warps']
      interpol_masks = other_inputs['interpol_masks']
      out_gen = self.gen(input, interpol_pose, interpol_warps, interpol_masks)
      out_gen = out_gen[-1]
    else:
      warps = other_inputs['warps']
      masks = other_inputs['masks']
      out_gen = self.gen(input, warps, masks)

    inp_img, inp_pose, out_pose = pose_utils.get_imgpose(input, opt['use_input_pose'], opt['pose_dim'])

    fake_disc_inp = torch.cat([inp_img, inp_pose, out_gen, out_pose], dim=1)
    r_inp_img, r_inp_pose, r_out_pose = pose_utils.get_imgpose(real_inp, opt['use_input_pose'], opt['pose_dim'])
    real_disc_inp = torch.cat([r_inp_img, r_inp_pose, real_target, r_out_pose], dim=1)
    data_dis = torch.cat((real_disc_inp, fake_disc_inp), 0)
    res_dis = self.disc(data_dis)

    # print(res_dis.shape)
    for it in range(res_dis.shape[0]):
      out = res_dis[it, :]
      if (it < opt['batch_size']):
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
          # ad_true_loss = -torch.mean(torch.log(out + 1e-7))= nn.functional.binary_cross_entropy(out, all0)
          ad_fake_loss = -torch.mean(torch.log(1 - out + 1e-7))
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
    return [self.dis_total_loss, self.dis_true_loss, self.dis_fake_loss]

  def nn_loss(self, predicted, ground_truth, nh=3, nw=3):
    v_pad = nh // 2
    h_pad = nw // 2
    val_pad = nn.ConstantPad2d((v_pad, v_pad, h_pad, h_pad), -10000)(ground_truth)

    reference_tensors = []
    for i_begin in range(0, nh):
      i_end = i_begin - nh + 1
      i_end = None if i_end == 0 else i_end
      for j_begin in range(0, nw):
        j_end = j_begin - nw + 1
        j_end = None if j_end == 0 else j_end
        sub_tensor = val_pad[:, :, i_begin:i_end, j_begin:j_end]
        reference_tensors.append(sub_tensor.unsqueeze(-1))
    reference = torch.cat(reference_tensors, dim=-1)
    ground_truth = ground_truth.unsqueeze(dim=-1)

    predicted = predicted.unsqueeze(-1)
    abs = torch.abs(reference - predicted)
    # sum along channels
    norms = torch.sum(abs, dim=1)
    # min over neighbourhood
    loss,_ = torch.min(norms, dim=-1)
    # loss = torch.sum(loss)/self.batch_size
    loss = torch.mean(loss)

    return loss

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

