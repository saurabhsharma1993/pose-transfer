# rewrite pose utils and conditional gan and train file to due to 16 dimensional pose vector
# refactor code to take input of dimensionality of pose
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.draw import circle, line_aa, polygon
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import skimage.measure, skimage.transform
import sys
import torch
import os
from torchvision.models import vgg19

LIMB_SEQ = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5],
         [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15],
         [6, 8], [8, 9]]


COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


LABELS = [ 'Rank' , 'Rknee' , 'Rhip' , 'Lhip' , 'Lknee' , 'Lank' , 'pelv' ,  'spine' , 'neck' , 'head' , 'Rwri' , 'Relb' , 'Rsho' , 'Lsho' , 'Lelb' , 'Lelb'  ]


LIMB_SEQ_PAF = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,16], [5,17]]


LABELS_PAF = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']



MISSING_VALUE = -1


def get_model_list(dirname, key):
  if os.path.exists(dirname) is False:
    return None
  gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                os.path.isfile(os.path.join(dirname, f)) and key in f and "pkl" in f]
  if gen_models is None or gen_models==[]:
    return None
  gen_models.sort()
  last_model_name = gen_models[-1]
  return last_model_name

def map_to_cord(pose_map, pose_dim, threshold=0.1):
    all_peaks = [[] for i in range(pose_dim)]
    pose_map = pose_map[..., :pose_dim]

    y, x, z = np.where(np.logical_and(pose_map == pose_map.max(axis = (0, 1)),
                                     pose_map > threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(pose_dim):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)


def cords_to_map(cords, img_size, sigma=6):
    result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
    return result


def compute_interpol_pose(inp_pos,tg_pos,index,num_stacks,pose_dim):
    assert index <= num_stacks
    if(pose_dim==16):
        interpol_pose = inp_pos + (tg_pos-inp_pos)*index/num_stacks
    # bad logic to circumvent missing annot . . synthesize and vanish missing annot after half sequence is completed
    elif(pose_dim==18):
        interpol_pose = np.zeros([pose_dim,2], dtype='float32')
        for i in range(pose_dim):
            # inp pose has missing annot and tg pose has it
            if ((inp_pos[i,0] == MISSING_VALUE or inp_pos[i,1]== MISSING_VALUE) and
                    (tg_pos[i,0] != MISSING_VALUE and tg_pos[i,1]!= MISSING_VALUE)):
                if(index<=num_stacks//2):
                    interpol_pose[i] = MISSING_VALUE
                else:
                    interpol_pose[i] = tg_pos[i]
            # tg pose has missing annot and inp pose has it
            elif ((tg_pos[i, 0] == MISSING_VALUE or tg_pos[i, 1] == MISSING_VALUE) and (
                    inp_pos[i, 0] != MISSING_VALUE and inp_pos[i, 1] != MISSING_VALUE)):
                if (index <= num_stacks // 2):
                    interpol_pose[i] = inp_pos[i]
                else:
                    interpol_pose[i] = MISSING_VALUE
            # annot missing in both poses
            elif ((tg_pos[i, 0] == MISSING_VALUE or tg_pos[i, 1] == MISSING_VALUE) and (
                    inp_pos[i, 0] == MISSING_VALUE or inp_pos[i, 1] == MISSING_VALUE)):
                interpol_pose[i] = MISSING_VALUE
            # normal interpol when annot are present in both cases
            else:
                interpol_pose[i] = inp_pos[i] + (tg_pos[i]-inp_pos[i])*index/num_stacks
    return interpol_pose

def draw_pose_from_cords(pose_joints, pose_dim, img_size, radius=2, draw_joints=True, img = np.zeros([224,224,3])):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    # interesting case of numpy array assignment semantics here . . the below ass is a view, hence the overliad poses error that destroyed the precious time of an incomporable specimen on this overpopulated information craving filth obsessed world
    # colors = img
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        if(pose_dim==16):
            for f, t in LIMB_SEQ:
                from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
                to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
                if from_missing or to_missing:
                    continue
                yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
                colors[yy, xx] = np.expand_dims(val, 1) * 255
                mask[yy, xx] = True
        else:
            for f, t in LIMB_SEQ_PAF:
                from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
                to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
                if from_missing or to_missing:
                    continue
                yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
                colors[yy, xx] = np.expand_dims(val, 1) * 255
                mask[yy, xx] = True

    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask


def draw_pose_from_map(pose_map, pose_dim, threshold=0.1, **kwargs):
    cords = map_to_cord(pose_map, pose_dim, threshold=threshold)
    return draw_pose_from_cords(cords, pose_dim, pose_map.shape[:2], **kwargs)


def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def mean_inputation(X):
    X = X.copy()
    for i in range(X.shape[1]):
        for j in range(X.shape[2]):
            val = np.mean(X[:, i, j][X[:, i, j] != -1])
            X[:, i, j][X[:, i, j] == -1] = val
    return X

def draw_legend():
    handles = [mpatches.Patch(color=np.array(color) / 255.0, label=name) for color, name in zip(COLORS, LABELS)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def produce_ma_mask(kp_array, img_size, point_radius=4):
    from skimage.morphology import dilation, erosion, square
    mask = np.zeros(shape=img_size, dtype=bool)
    limbs = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10],
              [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17],
               [1,16], [16,18], [2,17], [2,18], [9,12], [12,6], [9,3], [17,18]]
    limbs = np.array(limbs) - 1
    for f, t in limbs:
        from_missing = kp_array[f][0] == MISSING_VALUE or kp_array[f][1] == MISSING_VALUE
        to_missing = kp_array[t][0] == MISSING_VALUE or kp_array[t][1] == MISSING_VALUE
        if from_missing or to_missing:
            continue

        norm_vec = kp_array[f] - kp_array[t]
        norm_vec = np.array([-norm_vec[1], norm_vec[0]])
        norm_vec = point_radius * norm_vec / np.linalg.norm(norm_vec)


        vetexes = np.array([
            kp_array[f] + norm_vec,
            kp_array[f] - norm_vec,
            kp_array[t] - norm_vec,
            kp_array[t] + norm_vec
        ])
        yy, xx = polygon(vetexes[:, 0], vetexes[:, 1], shape=img_size)
        mask[yy, xx] = True

    for i, joint in enumerate(kp_array):
        if kp_array[i][0] == MISSING_VALUE or kp_array[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=point_radius, shape=img_size)
        mask[yy, xx] = True

    mask = dilation(mask, square(5))
    mask = erosion(mask, square(5))
    return mask


def _preprocess_image(image):
    return (image / 255 - 0.5) * 2

def _deprocess_image(image):
    return (255 * (image + 1) / 2).byte()

# channels last for display by pyplot/cv2
def postProcess(image):
    return image.permute(0,2,3,1)


def get_imgpose(input, use_input_pose, pose_dim):
    inp_img = input[:, :3]
    inp_pose = input[:, 3:3 + pose_dim] if use_input_pose else None
    tg_pose_index = 3 + pose_dim if use_input_pose else 6
    tg_pose = input[:, tg_pose_index:]

    return inp_img, inp_pose, tg_pose

def display(input_batch, target_batch, output_batch,_use_input_pose, pose_dim): # output_batch for result image
    row = input_batch.shape[0]
    col = 1

    inp_img, inp_pose, tg_pose = get_imgpose(input_batch, _use_input_pose, pose_dim)
    inp_img = postProcess(_deprocess_image(inp_img))
    tg_pose = postProcess(tg_pose)
    tg_img = postProcess(_deprocess_image(target_batch))

    res_img = postProcess(_deprocess_image(output_batch))

    # changing original code due to extra named argument
    inp_img = make_grid(inp_img, None, row=row, col=col)

    pose_images = np.array([draw_pose_from_map(pose.numpy(), pose_dim)[0] for pose in tg_pose])
    tg_pose = make_grid(torch.from_numpy(pose_images), None, row=row, col=col)

    tg_img = make_grid(tg_img, None, row=row, col=col)
    res_img = make_grid(res_img, None, row=row, col=col)

    return np.concatenate(np.array([inp_img, tg_pose, tg_img, res_img]), axis=1) #res_img]) , axis=1)


def display_stacked(input_batch, interpol_batch, target_batch, output_batch, num_stacks, _use_input_pose, pose_dim):
    row = input_batch.shape[0]
    col = 1

    inp_img, inp_pose, tg_pose = get_imgpose(input_batch, _use_input_pose, pose_dim)
    inp_img = postProcess(_deprocess_image(inp_img))
    interpol_pose = postProcess(interpol_batch)
    tg_img = postProcess(_deprocess_image(target_batch))

    res_img = [postProcess(_deprocess_image(output.data.cpu())) for output in output_batch]

    # changing original code due to extra named argument
    inp_img = make_grid(inp_img, None, row=row, col=col)

    result = []
    for i in range(num_stacks):
        pose_batch = interpol_pose[:,:,:,i*pose_dim:(i+1)*pose_dim]
        pose_images = np.array([draw_pose_from_map(pose.numpy(), pose_dim)[0] for pose in pose_batch])
        result.append(pose_images)
    interpol_pose = np.concatenate(result, axis=0)
    # ith interpol pose for jth sample = interpol_pose[(i-1)*batch_size+j]
    interpol_pose = make_grid(torch.from_numpy(interpol_pose), None, row=row, col=num_stacks)

    tg_img = make_grid(tg_img, None, row=row, col=col)

    res_img = np.concatenate(res_img,axis=0)
    # ith res_img for jth sample = res_img[(i-1)*batch_size+j]
    res_img = make_grid(torch.from_numpy(res_img), None, row=row, col=num_stacks)

    # print(inp_img.shape, interpol_pose.shape, tg_img.shape, res_img.shape)
    return np.concatenate([inp_img, interpol_pose, tg_img, res_img] , axis=1) #res_img]) , axis=1)



def make_grid(output_batch, input_batch = None, row=8, col=8, order=0):
    batch = output_batch.numpy()
    height, width = batch.shape[1], batch.shape[2]
    total_width, total_height = width * col, height * row
    result_image = np.empty((total_height, total_width, batch.shape[3]), dtype=batch.dtype)
    batch_index = 0
    # fill rows first then columns
    if(order==0):
        for i in range(col):
            for j in range(row):
                result_image[(j * height):((j+1)*height), (i * width):((i+1)*width)] = batch[batch_index]
                batch_index += 1
    else:
        for i in range(row):
            for j in range(col):
                result_image[(i * height):((i + 1) * height), (j * width):((j + 1) * width)] = batch[batch_index]
                batch_index += 1
    return result_image


def get_layer_ind(layer_name):
    block, conv = layer_name.split('_')
    block = int(block[-1])
    conv = int(conv[-1])
    blocks = ['0', '5', '10', '19', '28']
    return int(blocks[block-1]) + conv-1


def Feature_Extractor(model, input=None, layer_name=None):
    model = model.cuda()
    layer = get_layer_ind(layer_name)
    # for name, module in model.named_children():
    def preprocess_for_vgg(x):
        N,C,H,W = x.shape
        x = x.view(N,H,W,C)
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        x = (x - mean)/std
        x = x.view(N,C,H,W)
        return x

    input = preprocess_for_vgg(input)
    for it,module in enumerate(model.features.children()):
        # print(name)
        if(it<=layer):
            input = module(input)
    return input