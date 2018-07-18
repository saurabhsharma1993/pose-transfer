from utils import pose_utils
from utils.pose_transform import AffineTransformLayer
from torch.nn.modules.module import _addindent
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            Block,
            encoder,
            decoder,
            Generator,
            Discriminator,
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr

def keras_to_pytorch(model, layers, index=0):
    """Loads weights of pytorch model from Keras model, relying on sequential arrangement of model layers"""
    print("Setting weights for model ", model.__class__.__name__)
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        classname = module.__class__.__name__
        print("Setting weights for sub module ", classname)
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            Block,
            encoder,
            decoder,
            Generator,
            Discriminator,
        ]:
            # edit accordingly
            module, index = keras_to_pytorch(module, layers, index)

        elif( classname.find('Conv2d') !=-1 or classname.find('ConvTranspose2d') !=-1 ):
            while(True):
                weights = layers[index].get_weights()
                if(len(weights)==0):
                    index +=1
                elif(len(weights)==1):
                    module.weight.data = torch.from_numpy(np.transpose(weights[0],[3,2,0,1])).float()
                    print(index, layers[index].__class__)
                    index +=1
                    break
                elif(len(weights) == 2):
                    module.weight.data = torch.from_numpy(np.transpose(weights[0],[3,2,0,1])).float()
                    module.bias.data = torch.from_numpy(weights[1]).float()
                    print(index, layers[index].__class__)
                    index += 1
                    break
                else:
                    raise Exception('Unexpected keras layer at {0:d}'.format(index))

        elif (classname.find('InstanceNorm3d') != -1 ):
            while (True):
                weights = layers[index].get_weights()
                if (len(weights) == 0):
                    index += 1
                elif (len(weights) == 2):
                    module.weight.data = torch.from_numpy(weights[0]).float()
                    module.bias.data = torch.from_numpy(weights[1]).float()
                    print(index, layers[index].__class__)
                    index += 1
                    break

    return model, index

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

# glorot uniform weigths with zero biases
def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight.data)
        if(m.bias is not None):
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
            model.append(nn.InstanceNorm3d(1, eps=1e-3, affine=True, track_running_stats=False))
        if dropout:
            model.append(nn.Dropout2d())
        return nn.ModuleList(model)

    def forward(self, input):
        for module in self.net:
            if("Instance" in module.__class__.__name__):
                input = input.unsqueeze(1)
                input = module(input)
                input = input.squeeze()
            else:
                input = module(input)
        return input

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
                model.append(nn.Conv2d(input_nc, nf, kernel_size=3, padding=1))
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

class decoder(nn.Module):
    def __init__(self, nfilters_dec, nfilters_enc, num_skips = 1):
        super(decoder, self).__init__()
        # number of skip connections
        self.num_skips = num_skips
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.net = self.build_net(nfilters_dec)

    # alternative design of unet may be recursive, as in cyclegan
    def build_net(self, nfilters_dec):
        model_dec = []
        for i, nf in enumerate(nfilters_dec):
            if i==0:
                model_dec.append(Block(self.nfilters_enc[-1], nf, down=False, leaky=False, dropout=True))
            elif i==len(nfilters_dec)-1:
                model_dec.append(nn.ReLU())
                model_dec.append(nn.Conv2d((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, kernel_size=3, padding=1, bias=True))
            elif 0 < i < 3:
                # due to skip connections
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, down=False, leaky=False, dropout=True))
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

class Generator(nn.Module):
    def __init__(self, input_nc, nfilters_enc, nfilters_dec, num_skips = 1, warp_skip=False, use_input_pose=True):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        # number of skip connections
        self.num_skips = num_skips
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.encoder = encoder(input_nc, nfilters_enc)
        self.decoder = decoder(nfilters_dec, nfilters_enc, num_skips)
        # change to incorporate separate encoders when using dsc

    def forward(self, input):
        skips = self.encoder(input)
        out = self.decoder(skips)
        return out

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

        self.generator = Generator(input_nc, nfilters_enc, nfilters_dec, num_skips, warp_skip, use_input_pose)


    # interpol psoe called target pose here
    def forward(self, input, target_pose):
        # extract initial input and init pose
        init_input, init_pose, _ = pose_utils.get_imgpose(input,self.use_input_pose,self.pose_dim)
        outputs = []
        # at every stage feed input pose(if use input pose) as target pose for previous stage and new target pose from the list
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

class Discriminator(nn.Module):
    def __init__(self, input_nc, warp_skip=False, use_input_pose=True, checkMode = 0):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc
        self.use_input_pose = use_input_pose
        self.warp_skip = warp_skip
        self.checkMode = checkMode
        self.net = self.build_net()

    def build_net(self):
        # only writing code for simple kind of discriminator
        model = []
        model.append(nn.Conv2d(self.input_nc, 64, kernel_size=4, stride=2))
        model.append(Block(64,128))
        if(self.checkMode == 0):
            model.append(Block(128, 256))
            model.append(Block(256, 512))
            model.append(Block(512, 1, bn=False))
        else:
            model.append(Block(128, 1, bn=False))
        model.append(nn.Sigmoid())
        # notice the flatten . . this is for outputting a vector i.e output of discriminator can be of a variable size , needn't be a scalar
        # author was carrying a bias it should be a scalar
        model.append(Flatten())
        return nn.Sequential(*model)

    def forward(self, input):
        out = self.net(input)
        return out
