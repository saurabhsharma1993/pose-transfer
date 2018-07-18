import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from utils import pose_utils
from utils import pose_transform
from skimage.io import imread
import torch.utils.data as data
import pandas as pd
import os


class PoseTransfer_Dataset(data.Dataset):
    def __init__(self, opt, split):
        self.split = split

        self.gen_type = opt['gen_type']
        self.num_stacks = opt['num_stacks']

        self.pose_dim = opt['pose_dim']
        self._batch_size = 1 if (self.split=='test' or self.split=='val') else opt['batch_size']
        self._image_size = opt['image_size']

        self._images_dir_train = opt['images_dir_train']
        self._images_dir_test = opt['images_dir_test']

        # self._pairs_file_train = pd.read_csv(opt['pairs_file_train'])
        # self._pairs_file_test = pd.read_csv(opt['pairs_file_test'])

        self._pairs_file_train = pd.read_csv(opt['pairs_file_train_interpol'])
        self._pairs_file_test = pd.read_csv(opt['pairs_file_test_interpol'])

        # self.pairs_file_test_interpol = pd.read_csv(opt['pairs_file_test_interpol'])
        # self.pairs_file_train_interpol = pd.read_csv(opt['pairs_file_train_interpol'])


        # self._pairs_file_test_iterative = pd.read_csv(opt['pairs_file_test_iterative'])
        # self._pairs_file_train_iterative = pd.read_csv(opt['pairs_file_train_iterative'])

        self._annotations_file_test = pd.read_csv(opt['annotations_file_train'], sep=':')
        self._annotations_file_train = pd.read_csv(opt['annotations_file_test'], sep=':')

        self._annotations_file = pd.concat([self._annotations_file_test, self._annotations_file_train],
                                           axis=0, ignore_index=True)

        self._annotations_file = self._annotations_file.set_index('name')

        if (split == 'train'):
            self.length = len(self._pairs_file_train )
        else:
            self.length = len(self._pairs_file_test )

        self._use_input_pose = opt['use_input_pose']

        self._warp_skip = opt['warp_skip']

        # self._disc_type = opt['disc_type']

        self._tmp_pose = opt['tmp_pose_dir']
        # self.frame_diff = opt['frame_diff']

        # use if enough space available
        # if not os.path.exists(self._tmp_pose):
        #     os.makedirs(self._tmp_pose)

        print ("Statistics for loaded dataset : {}".format(opt['dataset']))
        print ("Number of images: %s" % len(self._annotations_file))
        print ("Number of pairs train: %s" % len(self._pairs_file_train))
        print ("Number of pairs test: %s" % len(self._pairs_file_test))

        # print("Number of pairs train: %s" % len(self.pairs_file_train_interpol))
        # print("Number of pairs test: %s" % len(self.pairs_file_test_interpol))

        # print ("Number of pairs test iterative: %s" % len(self._pairs_file_train_iterative))
        # print ("Number of pairs test iterative: %s" % len(self._pairs_file_test_iterative))

    # pair is now a pandas series
    def compute_pose_map(self, pair, direction):
        assert direction in ['to', 'from']
        pose_map = np.empty(list(self._image_size) + [self.pose_dim])
        row = self._annotations_file.loc[pair[direction]]
        # file_name = self._tmp_pose + pair[direction] + '.npy'
        # if os.path.exists(file_name):
        #     pose = np.load(file_name)
        # else:
        kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
        pose_map = pose_utils.cords_to_map(kp_array, self._image_size)
        # np.save(file_name, pose)
        # channels must be first
        return np.transpose(pose_map, [2,0,1] )

    def compute_cord_warp(self, pair):
        if self._warp_skip == 'full':
            warp = [np.empty([1, 8]), 1]
        else:
            warp = [np.empty([10, 8]),
                    np.empty([10] + list(self._image_size))]
        fr = self._annotations_file.loc[pair['from']]
        to = self._annotations_file.loc[pair['to']]
        kp_array1 = pose_utils.load_pose_cords_from_strings(fr['keypoints_y'],
                                                            fr['keypoints_x'])
        kp_array2 = pose_utils.load_pose_cords_from_strings(to['keypoints_y'],
                                                            to['keypoints_x'])
        if self._warp_skip == 'mask':
            warp[0] = pose_transform.affine_transforms(kp_array1, kp_array2, self.pose_dim)
            warp[1] = pose_transform.pose_masks(kp_array2, self._image_size, self.pose_dim)
        else:
            warp[0] = pose_transform.estimate_uniform_transform(kp_array1, kp_array2, self.pose_dim)
        return warp

    # returns a warp and mask for each interpolated pose pairs
    def compute_interpol_cord_warp(self, inp_map, interpol_pose):
        interpol_pose = [inp_map] + interpol_pose
        num_interpol = len(interpol_pose)
        interpol_warps, interpol_masks = [], []
        # possibly causing nan error
        kp_array1 = pose_utils.map_to_cord(np.transpose(inp_map, [1,2,0]), self.pose_dim)
        for pose in interpol_pose:
            if self._warp_skip == 'full':
                warp = [np.empty([1, 8]), 1]
            else:
                warp = [np.empty([10, 8]),
                        np.empty([10] + list(self._image_size))]
            kp_array2 = pose_utils.map_to_cord(np.transpose(pose, [1,2,0]), self.pose_dim)
            if self._warp_skip == 'mask':
                warp[0] = pose_transform.affine_transforms(kp_array1, kp_array2, self.pose_dim)
                warp[1] = pose_transform.pose_masks(kp_array2, self._image_size, self.pose_dim)
            else:
                warp[0] = pose_transform.estimate_uniform_transform(kp_array1, kp_array2, self.pose_dim)
            interpol_warps.append(warp[0])
            interpol_masks.append(warp[1])
            kp_array1 = kp_array2
        return np.array(interpol_warps), np.array(interpol_masks)


    def load_image(self, pair, direction='from'):
        assert direction in ['to', 'from']
        if os.path.exists(os.path.join(self._images_dir_train, pair[direction])):
            image = imread(os.path.join(self._images_dir_train, pair[direction]))
        elif os.path.exists(os.path.join(self._images_dir_test, pair[direction])):
            image = imread(os.path.join(self._images_dir_test, pair[direction]))
        else:
            # blank image if no file present on disk . . hacky way
            image = np.zeros([self._image_size[0],self._image_size[1], 3])
        return np.transpose(pose_utils._preprocess_image(image), [2,0,1])

    def compute_interpol_map(self, inp_map, tg_map):

    # map to cord expects channels in last dim ( old keras code )
        inp_pos = pose_utils.map_to_cord(np.transpose(inp_map, [1,2,0]), self.pose_dim)
        tg_pos = pose_utils.map_to_cord(np.transpose(tg_map, [1,2,0]), self.pose_dim)
        pose_maps = []
        # compute interpol poses equal to num_stacks, with final pose being equal to the target psoe
        for i in range(1,self.num_stacks+1):
            interpol_pose = pose_utils.compute_interpol_pose(inp_pos,tg_pos,i, self.num_stacks, self.pose_dim)
            interpol_pose_map = pose_utils.cords_to_map(interpol_pose, self._image_size)
            pose_maps.append(np.transpose(interpol_pose_map, [2,0,1]))
        return pose_maps


    # returns a dictionary with the generator sample and the discriminator sample
    # training loop will update generator and discriminator accordingly
    def __getitem__(self, index):
        if self.split == 'val' or self.split == 'test':
            pair = self._pairs_file_test.iloc[index]
        else:
            pair = self._pairs_file_train.iloc[index]
        result = [self.load_image(pair, 'from')]
        input_pose_map = self.compute_pose_map(pair, 'from')
        if self._use_input_pose:
            result.append(input_pose_map)
        target_pose_map = self.compute_pose_map(pair, 'to')
        result.append( target_pose_map )
        target = self.load_image(pair, 'to')

        interpol_pose_map = self.compute_interpol_map(input_pose_map,target_pose_map)

        interpol_warps, interpol_masks = self.compute_interpol_cord_warp(input_pose_map, interpol_pose_map)
        interpol_pose_map = torch.from_numpy(np.concatenate(interpol_pose_map, axis=0)).float()

        warps, masks = self.compute_cord_warp(pair)

        input = torch.from_numpy(np.concatenate(result, axis=0)).float()
        target = torch.from_numpy(target).float()

        if(self.gen_type=='baseline'):
            return input, target, warps, masks
        elif(self.gen_type=='stacked'):
            return input, target, interpol_pose_map, interpol_warps, interpol_masks

    def __len__(self):
        return self.length


# class PoseTransfer_Dataset(data.Dataset):
#     def __init__(self, opt, split):
#         self.split = split
#
#         self.pose_dim = opt['pose_dim']
#         self.num_stacks = opt['num_stacks']
#         self._batch_size = 1 if (self.split=='test' or self.split=='val') else opt['batch_size']
#         self._image_size = opt['image_size']
#
#         self._images_dir_train = opt['images_dir_train']
#         self._images_dir_test = opt['images_dir_test']
#
#         # self._pairs_file_train = pd.read_csv(opt['pairs_file_train'])
#         # self._pairs_file_test = pd.read_csv(opt['pairs_file_test'])
#
#         self.pairs_file_test_interpol = pd.read_csv(opt['pairs_file_test_interpol'])
#         self.pairs_file_train_interpol = pd.read_csv(opt['pairs_file_train_interpol'])
#
#         self._annotations_file_test = pd.read_csv(opt['annotations_file_train'], sep=':')
#         self._annotations_file_train = pd.read_csv(opt['annotations_file_test'], sep=':')
#
#         self._annotations_file = pd.concat([self._annotations_file_test, self._annotations_file_train],
#                                            axis=0, ignore_index=True)
#
#         self._annotations_file = self._annotations_file.set_index('name')
#
#         if (split == 'train'):
#             self.length = len(self.pairs_file_train_interpol)
#         else:
#             self.length = len(self.pairs_file_test_interpol)
#
#         self._use_input_pose = opt['use_input_pose']
#
#         self._warp_skip = opt['warp_skip']
#
#         # self._disc_type = opt['disc_type']
#
#         self._tmp_pose = opt['tmp_pose_dir']
#         # self.frame_diff = opt['frame_diff']
#
#         if not os.path.exists(self._tmp_pose):
#             os.makedirs(self._tmp_pose)
#
#         print ("Statistics for loaded dataset : Human 3.6")
#         print ("Number of images: %s" % len(self._annotations_file))
#         # print ("Number of pairs train: %s" % len(self._pairs_file_train_iterative))
#         # print ("Number of pairs test: %s" % len(self._pairs_file_test_iterative))
#
#         print ("Number of pairs train interpol: %s" % len(self.pairs_file_train_interpol))
#         print ("Number of pairs test interpol: %s" % len(self.pairs_file_test_interpol))
#
#     # pair is now a pandas series
#     def compute_pose_map(self, pair, direction):
#         assert direction in ['to', 'from']
#         pose_map = np.empty(list(self._image_size) + [self.pose_dim])
#         row = self._annotations_file.loc[pair[direction]]
#         # file_name = self._tmp_pose + pair[direction] + '.npy'
#         # if os.path.exists(file_name):
#         #     pose = np.load(file_name)
#         # else:
#         kp_array = pose_utils.load_pose_cords_from_strings(row['keypoints_y'], row['keypoints_x'])
#         pose_map = pose_utils.cords_to_map(kp_array, self._image_size)
#         # np.save(file_name, pose)
#
#         # channels must be first for pytorch ( cord to map yields channles in last dim - old keras code )
#         return np.transpose(pose_map, [2,0,1] )
#
#     def compute_interpol_map(self, inp_map, tg_map):
#         # map to cord expects channels in last dim ( old keras code )
#         inp_pos = pose_utils.map_to_cord(np.transpose(inp_map, [1,2,0]), self.pose_dim)
#         tg_pos = pose_utils.map_to_cord(np.transpose(tg_map, [1,2,0]), self.pose_dim)
#         pose_maps = []
#         # compute interpol poses equal to num_stacks, with final pose being equal to the target psoe
#         for i in range(1,self.num_stacks+1):
#             interpol_pose = pose_utils.compute_interpol_pose(inp_pos,tg_pos,i, self.num_stacks, self.pose_dim)
#             interpol_pose_map = pose_utils.cords_to_map(interpol_pose, self._image_size)
#             pose_maps.append(np.transpose(interpol_pose_map, [2,0,1]))
#         return pose_maps
#
#     def compute_cord_warp(self, pair):
#         if self._warp_skip == 'full':
#             warp = [np.empty([1, 8])]
#         else:
#             warp = [np.empty([10, 8]),
#                      np.empty([10] + list(self._image_size))]
#         fr = self._annotations_file.loc[pair['from']]
#         to = self._annotations_file.loc[pair['to']]
#         kp_array1 = pose_utils.load_pose_cords_from_strings(fr['keypoints_y'],
#                                                             fr['keypoints_x'])
#         kp_array2 = pose_utils.load_pose_cords_from_strings(to['keypoints_y'],
#                                                             to['keypoints_x'])
#         if self._warp_skip == 'mask':
#             warp[0] = pose_transform.affine_transforms(kp_array1, kp_array2, self.pose_dim)
#             warp[1] = pose_transform.pose_masks(kp_array2, self._image_size, self.pose_dim)
#         else:
#             warp[0] = pose_transform.estimate_uniform_transform(kp_array1, kp_array2)
#         return warp
#
#     def load_image(self, pair, direction='from'):
#         assert direction in ['to', 'from']
#         if os.path.exists(os.path.join(self._images_dir_train, pair[direction])):
#             image = imread(os.path.join(self._images_dir_train, pair[direction]))
#         elif os.path.exists(os.path.join(self._images_dir_test, pair[direction])):
#             image = imread(os.path.join(self._images_dir_test, pair[direction]))
#         else:
#             # blank image if no file present on disk . . hacky way
#             image = np.zeros([3, self._image_size,self._image_size])
#         return np.transpose(pose_utils._preprocess_image(image), [2,0,1])
#
#     # returns a dictionary with the generator sample and the discriminator sample
#     # training loop will update generator and discriminator accordingly
#     def __getitem__(self, index):
#         if self.split == 'val' or self.split == 'test':
#             pair = self.pairs_file_test_interpol.iloc[index]
#         else:
#             pair = self.pairs_file_train_interpol.iloc[index]
#         result = [self.load_image(pair, 'from')]
#         input_pose_map = self.compute_pose_map(pair, 'from')
#         if self._use_input_pose:
#             result.append(input_pose_map)
#         target_pose_map = self.compute_pose_map(pair, 'to')
#         result.append(target_pose_map)
#         interpol_pose_map = self.compute_interpol_map(input_pose_map,target_pose_map)
#         target = self.load_image(pair, 'to')
#
#         input = torch.from_numpy(np.concatenate(result, axis=0)).float()
#         target = torch.from_numpy(target).float()
#         interpol_pose_map = torch.from_numpy(np.concatenate(interpol_pose_map, axis=0)).float()
#         # if(torch.cuda.is_available()):
#         #     input = input.cuda()
#
#         # TODO
#         # if self._warp_skip != 'none' and (not for_discriminator or self._disc_type == 'warp'):
#         #     result += self.compute_cord_warp(pair)
#         return input, target, interpol_pose_map
#
#     def __len__(self):
#         return self.length

