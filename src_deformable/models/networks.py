# from keras.models import Model, Input
# from keras.layers import Flatten, Concatenate, Activation, Dropout, Dense
# from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D, Cropping2D
# from keras_contrib.layers.normalization import InstanceNormalization
# from keras.layers.advanced_activations import LeakyReLU
# import keras.backend as K
# from keras.backend import tf as ktf
from utils import pose_utils
from keras.optimizers import Adam
from utils.pose_transform import AffineTransformLayer

import torch
import torch.nn as nn
from torch.nn import init


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# glorot uniform weigths with zero biases
def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight.data)
        if(m.bias is not None):
            init.constant_(m.bias.data, 0.0)

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Cropping2D(nn.Module):
    def __init__(self, crop_size):
        super(Cropping2D, self).__init__()
        self.crop_size = crop_size
    def forward(self, input):
        return input[:,:,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]

# pytorch block module based on below commented keras code
class Block(nn.Module):
    def __init__(self,  input_nc, output_nc, down=True, bn=True, dropout=False, leaky=True):
        super(Block, self).__init__()
        self.net = self.build_net( input_nc, output_nc, down, bn, dropout, leaky)

    def build_net(self, input_nc, output_nc, down=True, bn=True, dropout=False, leaky=True):
        model = []
        if leaky:
            model.append(nn.LeakyReLU(0.2))
        else:
            model.append(nn.ReLU())
        if down:
            model.append(nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False))
        else:
            model.append(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, bias=False))
            model.append(Cropping2D(1))
        if bn:
            model.append(nn.InstanceNorm2d(output_nc, eps=1e-5))
        if dropout:
            model.append(nn.Dropout2d())
        return nn.Sequential(*model)

    def forward(self, input):
        return self.net(input)

# def block(out, nkernels, down=True, bn=True, dropout=False, leaky=True):
#     if leaky:
#         out = LeakyReLU(0.2)(out)
#     else:
#         out = Activation('relu')(out)
#     if down:
#         out = ZeroPadding2D((1, 1))(out)
#         out = Conv2D(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
#     else:
#         out = Conv2DTranspose(nkernels, kernel_size=(4, 4), strides=(2, 2), use_bias=False)(out)
#         out = Cropping2D((1, 1))(out)
#     if bn:
#         out = InstanceNormalization()(out)
#     if dropout:
#         out = Dropout(0.5)(out)
#     return out

# note - we must concatenate inputs and pass to the encoder
# returns a list of outputs, for the skip connections


# unet encoder in pytorch . . written this way to enhance modularity ( separate encoders for app and pose, as in defo-gan )
class encoder(nn.Module):
    def __init__(self, input_nc, nfilters_enc):
        super(encoder, self).__init__()
        self.input_nc = input_nc
        self.nfilters_enc = nfilters_enc
        self.net = self.build_net(input_nc, nfilters_enc)

    def build_net(self, input_nc, nfilters_enc):
        model = []
        for i, nf in enumerate(nfilters_enc):
            if i == 0:
                model.append(nn.Conv2d(input_nc, nf, kernel_size=3, padding=1, bias=True))
            elif i == len(nfilters_enc) - 1:
                model.append(Block(nfilters_enc[i-1], nf, bn=False))
            else:
                model.append(Block(nfilters_enc[i - 1], nf))
        return nn.ModuleList(model)

    def forward(self, input):
        outputs = []
        for i,module in enumerate(self.net):
            if(i==0):
                out = module(input)
                outputs.append(out)
            else:
                out = module(out)
                outputs.append(out)
        return outputs


# this should backpropagate the gradient through the skips, as skips are a pytorch variable, and shall be optimized in teh combined generator
# else use the bulky unet generator
class decoder(nn.Module):
    def __init__(self, nfilters_dec, nfilters_enc, num_skips = 1):
        super(decoder, self).__init__()
        # number of skip connections
        self.num_skips = num_skips
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.net = self.build_net(nfilters_dec)

    # def set_skips(self, skips):
    #     self.skips = skips

    # added logic for computing number of input channels due to skip connections
    # alternative design of unet may be recursive, as in cyclegan
    def build_net(self, nfilters_dec):
        model_dec = []
        for i, nf in enumerate(nfilters_dec):
            if i==0:
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[-1], nf, down=False, leaky=False, dropout=True))
            elif 0 < i < 3:
                # due to skip connections
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, down=False, leaky=False, dropout=True))
            elif i==len(nfilters_dec)-1:
                model_dec.append(nn.ReLU())
                model_dec.append(nn.Conv2d((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, kernel_size=3, padding=1, bias=True))
            else:
                # due to skip connections
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, down=False, leaky=False))
        model_dec.append(nn.Tanh())

        return nn.ModuleList(model_dec)

    def forward(self, skips):
        for i in range(len(self.nfilters_dec)):
            if (i == 0):
                out = self.net[0](skips[-(i+1)])
            elif i<len(self.nfilters_dec)-1:
                out = torch.cat([out, skips[-(i+1)]], 1)
                out = self.net[i](out)
            else:
                # final processing
                out = torch.cat([out, skips[-(i+1)]], 1)
                out = self.net[i](out)
                out = self.net[i+1](out)
        # applying non linearity
        out = self.net[-1](out)
        return out

#
# class UNet(nn.Module):
#     def __init__(self, input_nc, nfilters_enc=(64, 128, 256, 512, 512, 512), nfilters_dec =(512, 512, 512, 256, 128, 3)):
#         super(UNet, self).__init__()
#         self.input_nc = input_nc
#         self.nfilters_enc = nfilters_enc
#         self.nfilters_dec = nfilters_dec
#         self.encoder, self.decoder = self.build_net(input_nc, nfilters_enc, nfilters_dec)
#
#     def build_net(self, input_nc, nfilters_enc, nfilters_dec):
#         model_enc, model_dec = [], []
#
#         for i, nf in enumerate(nfilters_enc):
#             if i == 0:
#                 model_enc.append(nn.Conv2D(input_nc, nf, kernel_size=3, padding=1))
#             elif i == len(nfilters_enc) - 1:
#                 model_enc.append(Block(nfilters_enc[i-1], nf, bn=False))
#             else:
#                 model_enc.append(Block(nfilters_enc[i - 1], nf))
#
#         for i, nf in enumerate(nfilters_dec):
#             if i==0:
#                 model_dec.append(nn.ReLU())
#                 model_dec.append(nn.Conv2d(nfilters_enc[-1], nf, kernel_size=3, padding=1, bias=True))
#             elif 0 < i < 3:
#                 model_dec.append(Block(nfilters_dec[i - 1], nf, down=False, leaky=False, dropout=True))
#             else:
#                 model_dec.append(Block(nfilters_dec[i - 1], nf, down=False, leaky=False))
#         model_dec.append(nn.Tanh())
#
#         # add affinetransform layers for warping skip connections later
#         return nn.ModuleList(model_enc), nn.ModuleList(model_dec)
#
#     def forward(self, input):
#         skips = []
#         for i,module in enumerate(self.encoder):
#             if(i==0):
#                 out = module(input)
#                 skips.append(out)
#             else:
#                 out = module(out)
#                 skips.append(out)
#
#         for i in range(self.nfilters_dec):
#             if(i==0):
#                 out = self.decoder[0](out)
#                 out = self.decoder[1](out)
#             else:
#                 out = torch.cat([out,skips[-i]], 1)
#                 out = self.decoder[i+1](out)
#
#         out = self.decoder[-1](out)
#         return out

#
# def encoder(inps, nfilters=(64, 128, 256, 512, 512, 512)):
#     layers = []
#     if len(inps) != 1:
#         out = Concatenate(axis=-1)(inps)
#     else:
#         out = inps[0]
#     for i, nf in enumerate(nfilters):
#         if i == 0:
#             out = Conv2D(nf, kernel_size=(3, 3), padding='same')(out)
#         elif i == len(nfilters) - 1:
#             out = block(out, nf, bn=False)
#         else:
#             out = block(out, nf)
#         layers.append(out)
#     return layers

# def decoder(skips, nfilters=(512, 512, 512, 256, 128, 3)):
#     out = None
#     for i, (skip, nf) in enumerate(zip(skips, nfilters)):
#         if 0 < i < 3:
#             out = Concatenate(axis=-1)([out, skip])
#             out = block(out, nf, down=False, leaky=False, dropout=True)
#         elif i == 0:
#             out = block(skip, nf, down=False, leaky=False, dropout=True)
#         elif i == len(nfilters) - 1:
#             out = Concatenate(axis=-1)([out, skip])
#             out = Activation('relu')(out)
#             out = Conv2D(nf, kernel_size=(3, 3), use_bias=True, padding='same')(out)
#         else:
#             out = Concatenate(axis=-1)([out, skip])
#             out = block(out, nf, down=False, leaky=False)
#     out = Activation('tanh')(out)
#     return out


# def concatenate_skips(skips_app, skips_pose, warp, image_size, warp_agg, warp_skip):
#     skips = []
#     for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
#         if i < 4:
#             out = AffineTransformLayer(10 if warp_skip == 'mask' else 1, warp_agg, image_size)([sk_app] + warp)
#             out = Concatenate(axis=-1)([out, sk_pose])
#         else:
#             out = Concatenate(axis=-1)([sk_app, sk_pose])
#         skips.append(out)
#     return skips


# def concatenate_skips(skips_app, skips_pose, warp, image_size, warp_agg, warp_skip):
#     skips = []
#     for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
#         if i < 4:
#             out = AffineTransformLayer(10 if warp_skip == 'mask' else 1, warp_agg, image_size)([sk_app] + warp)
#             out = Concatenate(axis=-1)([out, sk_pose])
#         else:
#             out = Concatenate(axis=-1)([sk_app, sk_pose])
#         skips.append(out)
#     return skips





class Deformable_Generator(nn.Module):
    def __init__(self, input_nc, pose_dim, image_size, nfilters_enc, nfilters_dec, warp_skip, use_input_pose=True):
        super(Deformable_Generator, self).__init__()
        self.input_nc = input_nc
        # number of skip connections
        self.num_skips = 1 if warp_skip=='None' else 2
        self.warp_skip = warp_skip
        self.pose_dim = pose_dim
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.image_size = image_size
        self.use_input_pose = use_input_pose
        self.encoder_app = encoder(input_nc-self.pose_dim, nfilters_enc)
        self.encoder_pose = encoder(self.pose_dim, nfilters_enc)
        self.decoder = decoder(nfilters_dec, nfilters_enc, self.num_skips)
        # change to incorporate separate encoders when using dsc

    def forward(self, input, warps, masks):
        inp_app, inp_pose, tg_pose = pose_utils.get_imgpose(input, self.use_input_pose, self.pose_dim)
        inp_app = torch.cat([inp_app, inp_pose], dim=1)
        skips_app = self.encoder_app(inp_app)
        skips_pose = self.encoder_pose(inp_pose)
        # define concatenate func
        skips = self.concatenate_skips(skips_app, skips_pose, warps, masks)
        out = self.decoder(skips)
        return out

    def concatenate_skips(self, skips_app, skips_pose, warps, masks):
        skips = []
        for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
            if i < 4:
                out = AffineTransformLayer(10 if self.warp_skip == 'mask' else 1, self.image_size, self.warp_skip)(sk_app, warps, masks)
                out = torch.cat([out, sk_pose], dim=1)
            else:
                out = torch.cat([sk_app, sk_pose], dim=1)
            skips.append(out)
        return skips

class Stacked_Generator(nn.Module):
    def __init__(self, input_nc, num_stacks, pose_dim, nfilters_enc, nfilters_dec, num_skips = 1, warp_skip=False, use_input_pose=True):
        super(Stacked_Generator, self).__init__()
        self.input_nc = input_nc
        # number of skip connections
        self.num_skips = num_skips
        self.num_stacks = num_stacks
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.use_input_pose = use_input_pose
        self.pose_dim = pose_dim

        # gens = []
        # # maintains a stack of generators
        # for i in range(num_stacks):
        #     gens.append(Generator(input_nc, nfilters_enc, nfilters_dec, num_skips, warp_skip, use_input_pose))
        # self.stacked_gen = nn.ModuleList(gens)

        self.generator = Deformable_Generator(input_nc, nfilters_enc, nfilters_dec, num_skips, warp_skip, use_input_pose)


    # interpol psoe called target pose here
    def forward(self, input, target_pose):
        # extract initial input and init pose
        init_input, init_pose, _ = pose_utils.get_imgpose(input,self.use_input_pose,self.pose_dim)
        outputs = []
        # at every stage feed output from previous stage, input pose(if use input pose) as target pose for previous stage and new target pose from the list
        for i in range(self.num_stacks):
            if(i==0):
                if (self.use_input_pose):
                    inp = torch.cat([init_input, init_pose, target_pose[:, i * self.pose_dim:(i + 1) * self.pose_dim]], dim=1)
                else:
                    inp = torch.cat([init_input, target_pose[:,i*self.pose_dim:(i + 1)*self.pose_dim]], dim=1)
                # out = self.stacked_gen[i](inp)
                out = self.generator(inp)
            else:
                if(self.use_input_pose):
                    stage_inp = torch.cat([out,target_pose[:,(i-1)*self.pose_dim:i*self.pose_dim], target_pose[:,i*self.pose_dim:(i+1)*self.pose_dim]], dim=1)
                else:
                    stage_inp = torch.cat([out,target_pose[:,i*self.pose_dim:(i + 1)*self.pose_dim]], dim=1)
                # out = self.stacked_gen[i](stage_inp)
                out = self.generator(stage_inp)
            outputs.append(out)
        return outputs
#
# def make_generator(image_size, use_input_pose, warp_skip, disc_type, warp_agg):
#     # input is 128 x 64 x nc
#     use_warp_skip = warp_skip != 'none'
#     input_img = Input(list(image_size) + [3])
#     output_pose = Input(list(image_size) + [18])
#     output_img = Input(list(image_size) + [3])
#
#     nfilters_decoder = (512, 512, 512, 256, 128, 3) if max(image_size) == 128 else (512, 512, 512, 512, 256, 128, 3)
#     nfilters_encoder = (64, 128, 256, 512, 512, 512) if max(image_size) == 128 else (64, 128, 256, 512, 512, 512, 512)
#
#     if warp_skip == 'full':
#         warp = [Input((1, 8))]
#     elif warp_skip == 'mask':
#         warp = [Input((10, 8)), Input((10, image_size[0], image_size[1]))]
#     else:
#         warp = []
#
#     if use_input_pose:
#         input_pose = [Input(list(image_size) + [18])]
#     else:
#         input_pose = []
#
#     if use_warp_skip:
#         enc_app_layers = encoder([input_img] + input_pose, nfilters_encoder)
#         enc_tg_layers = encoder([output_pose], nfilters_encoder)
#         enc_layers = concatenate_skips(enc_app_layers, enc_tg_layers, warp, image_size, warp_agg, warp_skip)
#     else:
#         enc_layers = encoder([input_img] + input_pose + [output_pose], nfilters_encoder)
#
#     out = decoder(enc_layers[::-1], nfilters_decoder)
#
#     warp_in_disc = [] if disc_type != 'warp' else warp
#
#     return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + warp,
#                  outputs=[input_img] + input_pose + [out, output_pose] + warp_in_disc)


# take care of passing appropriate number of input channels . . the keras codebase doesn't care about this
class Discriminator(nn.Module):
    def __init__(self, input_nc, warp_skip=False, use_input_pose=True):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc
        self.use_input_pose = use_input_pose
        self.warp_skip = warp_skip
        self.net = self.build_net()

    def build_net(self):
        # only writing code for simple kind of discriminator
        model = []
        model.append(nn.Conv2d(self.input_nc, 64, kernel_size=4, stride=2))
        model.append(Block(64,128))
        model.append(Block(128, 256))
        model.append(Block(256, 512))
        model.append(Block(512, 1, bn=False))
        model.append(nn.Sigmoid())
        # notice the flatten . . this is for outputting a vector i.e output of discriminator can be of a variable size , needn't be a scalar
        # author was carrying a bias it should be a scalar
        model.append(Flatten())
        return nn.Sequential(*model)

    def forward(self, input):
        out = self.net(input)
        return out


# def make_discriminator(image_size, use_input_pose, warp_skip, disc_type, warp_agg):
#     input_img = Input(list(image_size) + [3])
#     output_pose = Input(list(image_size) + [18])
#     input_pose = Input(list(image_size) + [18])
#     output_img = Input(list(image_size) + [3])
#
#     if warp_skip == 'full':
#         warp = [Input((10, 8))]
#     elif warp_skip == 'mask':
#         warp = [Input((10, 8)), Input((10, image_size[0], image_size[1]))]
#     else:
#         warp = []
#
#     if use_input_pose:
#         input_pose = [input_pose]
#     else:
#         input_pose = []
#
#     if disc_type == 'call':
#         out = Concatenate(axis=-1)([input_img] + input_pose + [output_img, output_pose])
#         out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
#         out = block(out, 128)
#         out = block(out, 256)
#         out = block(out, 512)
#         out = block(out, 1, bn=False)
#         out = Activation('sigmoid')(out)
#         out = Flatten()(out)
#         return Model(inputs=[input_img] + input_pose + [output_img, output_pose], outputs=[out])
#     elif disc_type == 'sim':
#         out = Concatenate(axis=-1)([output_img, output_pose])
#         out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
#         out = block(out, 128)
#         out = block(out, 256)
#         out = block(out, 512)
#         m_share = Model(inputs=[output_img, output_pose], outputs=[out])
#         output_feat = m_share([output_img, output_pose])
#         input_feat = m_share([input_img] + input_pose)
#
#         out = Concatenate(axis=-1)([output_feat, input_feat])
#         out = LeakyReLU(0.2)(out)
#         out = Flatten()(out)
#         out = Dense(1)(out)
#         out = Activation('sigmoid')(out)
#
#         return Model(inputs=[input_img] + input_pose + [output_img, output_pose], outputs=[out])
#     else:
#         out_inp = Concatenate(axis=-1)([input_img] + input_pose)
#         out_inp = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out_inp)
#
#         out_inp = AffineTransformLayer(10, warp_agg, image_size)([out_inp] + warp)
#
#         out = Concatenate(axis=-1)([output_img, output_pose])
#         out = Conv2D(64, kernel_size=(4, 4), strides=(2, 2))(out)
#
#         out = Concatenate(axis=-1)([out, out_inp])
#
#         out = block(out, 128)
#         out = block(out, 256)
#         out = block(out, 512)
#         out = block(out, 1, bn=False)
#         out = Activation('sigmoid')(out)
#         out = Flatten()(out)
#         return Model(inputs=[input_img] + input_pose + [output_img, output_pose] + warp, outputs=[out])