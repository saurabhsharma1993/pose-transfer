import pylab as plt
import numpy as np
from skimage.io import imread
from skimage.transform import warp_coords
import skimage.measure
import skimage.transform
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from .pose_utils import LABELS, LABELS_PAF, MISSING_VALUE
import itertools

# write in pyTorch after testing baseline model

# no learnable parameters in the layer .. simply
# applies the given projective transform to the image
# and returns image of same size
# doesn't work because of numpy operations applied to torch tensors converted to numpy
class AffineLayer_Numpy(nn.Module):
    def __init__(self):
        super(AffineLayer_Numpy, self).__init__()

    def forward(self, input, transforms):
        num_transforms = transforms.shape[1]
        output = -1*torch.ones([num_transforms] + list(input.shape), requires_grad=True).cuda()

        def compute_mapping(x,y,transform):
            k = x*transform[6] + y*transform[7] + 1
            map_x = (x*transform[0] + y*transform[1] + transform[2])/k
            map_y = (x*transform[3] + y*transform[3] + transform[4])/k
            return round(map_x), round(map_y)

        batch_size = input.shape[0]

        for index in range(batch_size):
            image = input[index]
            for n in range(num_transforms):
                transform = transforms[index, n]
                # for _,(x,y) in enumerate(itertools.product(np.arange(image.shape[-2]),np.arange(image.shape[-1]))):
                #     inp_x, inp_y = compute_mapping(x,y,transform)
                #     # if valid input mapping
                #     if(inp_x>=0 and inp_x<image.shape[-2] and inp_y>=0 and inp_y<image.shape[-1]):
                #         output[n,index,:,x,y] = image[:,inp_x,inp_y]


                import cv2
                M = np.reshape(transform.cpu().numpy()[:6], [2, 3])
                inp_img = np.transpose(image.detach().data.cpu().numpy(),[1,2,0])
                rows,cols,ch = inp_img.shape
                warped_map = cv2.warpAffine(inp_img,M,(cols,rows))
                warped_map = np.transpose(warped_map, [2,0,1])
                if (np.min(warped_map) == 0 and np.max(warped_map) == 0):
                    warped_map[...] = -1
                output[n,index] = torch.from_numpy(warped_map)

        # batch x transform x channel x h x w
        output = output.permute(1,0,2,3,4)
        # output[output==0] = -1
        return Variable(output)


class AffineLayer(nn.Module):
    def __init__(self):
        super(AffineLayer, self).__init__()

    def forward(self, input, transforms):
        num_transforms = transforms.shape[1]
        # output = -1*torch.ones([num_transforms] + list(input.shape), requires_grad=True).cuda()
        N,C,H,W = input.shape
        input = input.unsqueeze(-1)
        input = input.repeat(1,num_transforms,1,1,1)
        input = input.view(N*num_transforms, C, H, W)

        transforms = transforms[:,:,:6].view(-1,2,3)
        # dividing bias of affine transform by size of image in respective dimesnsion
        # this is due to semantic of F.affine_grid, which outputs a flow field in [-1,1]
        # for which the biases must be normalized as well

        transforms = self.normalize_transforms(transforms, H, W)

        # note that the transforms are the inverse transforms, from output to input
        # similar to the tensorflow tf.contrib.image.transform api
        grid = F.affine_grid(transforms, input.shape)

        warped_map = F.grid_sample(input,grid)

        warped_map = warped_map.view(-1,num_transforms,C,H,W)
        # batch x transform x channel x h x w

        # warped_map[warped_map==0] = -1

        return warped_map

    def normalize_transforms(self, transforms, H,W):

        transforms[:,0,0] = transforms[:,0,0]
        transforms[:,0,1] = transforms[:,0,1]*W/H
        transforms[:,0,2] = transforms[:,0,2]*2/H + transforms[:,0,0] + transforms[:,0,1] - 1

        transforms[:,1,0] = transforms[:,1,0]*H/W
        transforms[:,1,1] = transforms[:,1,1]
        transforms[:,1,2] = transforms[:,1,2]*2/W + transforms[:,1,0] + transforms[:,1,1] - 1

        return transforms

class AffineTransformLayer(nn.Module):
    def __init__(self, number_of_transforms, init_image_size, warp_skip):
        super(AffineTransformLayer, self).__init__()
        self.number_of_transforms = number_of_transforms
        self.init_image_size = init_image_size
        self.affine_layer = AffineLayer()
        self.warp_skip = warp_skip
        # change to incorporate separate encoders when using dsc

    def forward(self, input, warps, masks):
        # height and width
        self.image_size = input.shape[2:]
        # self.scale = torch.nn.Upsample(size=self.image_size)
        self.affine_mul = torch.Tensor([1, 1, self.init_image_size[0] / self.image_size[0],
                           1, 1, self.init_image_size[1] / self.image_size[1],
                           1, 1]).cuda()
        #scaling transform according to image size
        warps = warps/self.affine_mul
        affine_transform = self.affine_layer(input,warps)
        # scaling masks
        if(self.warp_skip=='mask'):
        #     # if(self.init_image_size!=self.image_size):
        #     #     masks = self.scale(masks)
            import cv2
            masks = torch.from_numpy(np.array([cv2.resize(np.transpose(mask,[1,2,0]), self.image_size) for mask in masks.data.cpu().numpy()])).cuda()
            masks = masks.permute(0,3,1,2)
            # batch x transform x 1 x height x width
            masks = torch.unsqueeze(masks,dim=2).float()
            affine_transform = affine_transform*masks
        res,_ = torch.max(affine_transform, dim=1)
        # res[res==0] = -1

        return res
# class AffineTransformLayer(Layer):
#     def __init__(self, number_of_transforms, aggregation_fn, init_image_size, **kwargs):
#         assert aggregation_fn in ['none', 'max', 'avg']
#         self.aggregation_fn = aggregation_fn
#         self.number_of_transforms = number_of_transforms
#         self.init_image_size = init_image_size
#         super(AffineTransformLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.image_size = list(input_shape[0][1:])
#         self.affine_mul = [1, 1, self.init_image_size[0] / self.image_size[0],
#                            1, 1, self.init_image_size[1] / self.image_size[1],
#                            1, 1]
#         self.affine_mul = np.array(self.affine_mul).reshape((1, 1, 8))
#
#     def call(self, inputs):
#         expanded_tensor = ktf.expand_dims(inputs[0], -1)
#         multiples = [1, self.number_of_transforms, 1, 1, 1]
#         tiled_tensor = ktf.tile(expanded_tensor, multiples=multiples)
#         repeated_tensor = ktf.reshape(tiled_tensor, ktf.shape(inputs[0]) * np.array([self.number_of_transforms, 1, 1, 1]))
#
#         affine_transforms = inputs[1] / self.affine_mul
#
#         affine_transforms = ktf.reshape(affine_transforms, (-1, 8))
#         tranformed = tf_affine_transform(repeated_tensor, affine_transforms)
#         res = ktf.reshape(tranformed, [-1, self.number_of_transforms] + self.image_size)
#         res = ktf.transpose(res, [0, 2, 3, 1, 4])
#
#         #Use masks
#         if len(inputs) == 3:
#             mask = ktf.transpose(inputs[2], [0, 2, 3, 1])
#             mask = ktf.image.resize_images(mask, self.image_size[:2], method=ktf.image.ResizeMethod.NEAREST_NEIGHBOR)
#             res = res * ktf.expand_dims(mask, axis=-1)
#
#
#         if self.aggregation_fn == 'none':
#             res = ktf.reshape(res, [-1] + self.image_size[:2] + [self.image_size[2] * self.number_of_transforms])
#         elif self.aggregation_fn == 'max':
#             res = ktf.reduce_max(res, reduction_indices=[-2])
#         elif self.aggregation_fn == 'avg':
#             counts = ktf.reduce_sum(mask, reduction_indices=[-1])
#             counts = ktf.expand_dims(counts, axis=-1)
#             res = ktf.reduce_sum(res, reduction_indices=[-2])
#             res /= counts
#             res = ktf.where(ktf.is_nan(res), ktf.zeros_like(res), res)
#         return res
#
#     def compute_output_shape(self, input_shape):
#         if self.aggregation_fn == 'none':
#             return tuple([input_shape[0][0]] + self.image_size[:2] + [self.image_size[2] * self.number_of_transforms])
#         else:
#             return input_shape[0]
#
#     def get_config(self):
#         config = {"number_of_transforms": self.number_of_transforms,
#                   "aggregation_fn": self.aggregation_fn}
#         base_config = super(AffineTransformLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

def give_name_to_keypoints(array, pose_dim):
    res = {}
    if(pose_dim==16):
        for i, name in enumerate(LABELS):
            if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
                res[name] = array[i][::-1]
    else:
        for i, name in enumerate(LABELS_PAF):
            if array[i][0] != MISSING_VALUE and array[i][1] != MISSING_VALUE:
                res[name] = array[i][::-1]
    return res


def check_valid(kp_array):
    kp = give_name_to_keypoints(kp_array)
    return check_keypoints_present(kp, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])


def check_keypoints_present(kp, kp_names):
    result = True
    for name in kp_names:
        result = result and (name in kp)
    return result


def compute_st_distance(kp):
    st_distance1 = np.sum((kp['Rhip'] - kp['Rsho']) ** 2)
    st_distance2 = np.sum((kp['Lhip'] - kp['Lsho']) ** 2)
    return np.sqrt((st_distance1 + st_distance2)/2.0)


def mask_from_kp_array(kp_array, border_inc, img_size):
    min = np.min(kp_array, axis=0)
    max = np.max(kp_array, axis=0)
    min -= int(border_inc)
    max += int(border_inc)

    min = np.maximum(min, 0)
    max = np.minimum(max, img_size[::-1])

    mask = np.zeros(img_size)
    mask[min[1]:max[1], min[0]:max[0]] = 1
    return mask


def get_array_of_points(kp, names):
    return np.array([kp[name] for name in names])


def pose_masks(array2, img_size, pose_dim):
    kp2 = give_name_to_keypoints(array2, pose_dim)
    masks = []
    st2 = compute_st_distance(kp2)
    empty_mask = np.zeros(img_size)

    body_mask = np.ones(img_size)# mask_from_kp_array(get_array_of_points(kp2, ['Rhip', 'Lhip', 'Lsho', 'Rsho']), 0.1 * st2, img_size)
    masks.append(body_mask)

    head_candidate_names = {'Leye', 'Reye', 'Lear', 'Rear', 'nose'}
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp2:
            head_kp_names.add(cn)


    if len(head_kp_names)!=0:
        center_of_mass = np.mean(get_array_of_points(kp2, list(head_kp_names)), axis=0, keepdims=True)
        center_of_mass = center_of_mass.astype(int)
        head_mask = mask_from_kp_array(center_of_mass, 0.40 * st2, img_size)
        masks.append(head_mask)
    else:
        masks.append(empty_mask)

    def mask_joint(fr, to, inc_to):
        if not check_keypoints_present(kp2, [fr, to]):
            return empty_mask
        return skimage.measure.grid_points_in_poly(img_size, estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)[:, ::-1])

    masks.append(mask_joint('Rhip', 'Rkne', 0.1))
    masks.append(mask_joint('Lhip', 'Lkne', 0.1))

    masks.append(mask_joint('Rkne', 'Rank', 0.5))
    masks.append(mask_joint('Lkne', 'Lank', 0.5))

    masks.append(mask_joint('Rsho', 'Relb', 0.1))
    masks.append(mask_joint('Lsho', 'Lelb', 0.1))

    masks.append(mask_joint('Relb', 'Rwri', 0.5))
    masks.append(mask_joint('Lelb', 'Lwri', 0.5))

    return np.array(masks)


def estimate_polygon(fr, to, st, inc_to, inc_from, p_to, p_from):
    fr = fr + (fr - to) * inc_from
    to = to + (to - fr) * inc_to

    norm_vec = fr - to
    norm_vec = np.array([-norm_vec[1], norm_vec[0]])
    norm = np.linalg.norm(norm_vec)
    if norm == 0:
        return np.array([
            fr + 1,
            fr - 1,
            to - 1,
            to + 1,
        ])
    norm_vec = norm_vec / norm
    vetexes = np.array([
        fr + st * p_from * norm_vec,
        fr - st * p_from * norm_vec,
        to - st * p_to * norm_vec,
        to + st * p_to * norm_vec
    ])

    return vetexes

# note that the transforms are the inverse transforms, from output to input
# this is how tf and pytorch apis expect the affine warp matrices
def affine_transforms(array1, array2, pose_dim):
    kp1 = give_name_to_keypoints(array1, pose_dim)
    kp2 = give_name_to_keypoints(array2, pose_dim)

    st1 = compute_st_distance(kp1)
    st2 = compute_st_distance(kp2)


    no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])

    transforms = []
    def to_transforms(tr):
        from numpy.linalg import LinAlgError
        try:
            np.linalg.inv(tr)
            transforms.append(tr)
        except LinAlgError:
            transforms.append(no_point_tr)

    body_poly_1 = get_array_of_points(kp1, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])
    body_poly_2 = get_array_of_points(kp2, ['Rhip', 'Lhip', 'Lsho', 'Rsho'])
    tr = skimage.transform.estimate_transform('affine', src=body_poly_2, dst=body_poly_1)
    # tr = skimage.transform.estimate_transform('affine', src=body_poly_1, dst=body_poly_2)


    to_transforms(tr.params)

    head_candidate_names = {'Leye', 'Reye', 'Lear', 'Rear', 'nose'}
    head_kp_names = set()
    for cn in head_candidate_names:
        if cn in kp1 and cn in kp2:
            head_kp_names.add(cn)
    if len(head_kp_names) != 0:
        #if len(head_kp_names) < 3:
        head_kp_names.add('Lsho')
        head_kp_names.add('Rsho')
        head_poly_1 = get_array_of_points(kp1, list(head_kp_names))
        head_poly_2 = get_array_of_points(kp2, list(head_kp_names))
        tr = skimage.transform.estimate_transform('affine', src=head_poly_2, dst=head_poly_1)
        # tr = skimage.transform.estimate_transform('affine', src=head_poly_1, dst=head_poly_2)
        to_transforms(tr.params)
    else:
        to_transforms(no_point_tr)

    def estimate_join(fr, to, inc_to):
        if not check_keypoints_present(kp2, [fr, to]):
            return no_point_tr
        poly_2 = estimate_polygon(kp2[fr], kp2[to], st2, inc_to, 0.1, 0.2, 0.2)
        if check_keypoints_present(kp1, [fr, to]):
            poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
        else:
            if fr[0]=='R':
                fr = fr.replace('R', 'L')
                to = to.replace('R', 'L')
            else:
                fr = fr.replace('L', 'R')
                to = to.replace('L', 'R')
            if check_keypoints_present(kp1, [fr, to]):
                poly_1 = estimate_polygon(kp1[fr], kp1[to], st1, inc_to, 0.1, 0.2, 0.2)
            else:
                return no_point_tr
        return skimage.transform.estimate_transform('affine', dst=poly_1, src=poly_2).params
        # return skimage.transform.estimate_transform('affine', dst=poly_2, src=poly_1).params

    to_transforms(estimate_join('Rhip', 'Rkne', 0.1))
    to_transforms(estimate_join('Lhip', 'Lkne', 0.1))

    to_transforms(estimate_join('Rkne', 'Rank', 0.3))
    to_transforms(estimate_join('Lkne', 'Lank', 0.3))

    to_transforms(estimate_join('Rsho', 'Relb', 0.1))
    to_transforms(estimate_join('Lsho', 'Lelb', 0.1))

    to_transforms(estimate_join('Relb', 'Rwri', 0.3))
    to_transforms(estimate_join('Lelb', 'Lwri', 0.3))

    return np.array(transforms).reshape((-1, 9))[..., :-1]
    # return np.array(transforms).reshape((-1, 9))


def estimate_uniform_transform(array1, array2, pose_dim):
    kp1 = give_name_to_keypoints(array1, pose_dim)
    kp2 = give_name_to_keypoints(array2, pose_dim)

    no_point_tr = np.array([[1, 0, 1000], [0, 1, 1000], [0, 0, 1]])

    def check_invertible(tr):
        from numpy.linalg import LinAlgError
        try:
            np.linalg.inv(tr)
            return True
        except LinAlgError:
            return False

    keypoint_names = {'Rhip', 'Lhip', 'Lsho', 'Rsho'}
    candidate_names = {'Rkne', 'Lkne'}

    for cn in candidate_names:
        if cn in kp1 and cn in kp2:
            keypoint_names.add(cn)

    poly_1 = get_array_of_points(kp1, list(keypoint_names))
    poly_2 = get_array_of_points(kp2, list(keypoint_names))

    tr = skimage.transform.estimate_transform('affine', src=poly_2, dst=poly_1)
    # tr = skimage.transform.estimate_transform('affine', src=poly_1, dst=poly_2)

    tr = tr.params

    if check_invertible(tr):
        # return tr.reshape((-1, 9))[..., :-1]
        return tr.reshape((-1, 9))
    else:
        return no_point_tr.reshape((-1, 9))[..., :-1]
        # return no_point_tr.reshape((-1, 9))


