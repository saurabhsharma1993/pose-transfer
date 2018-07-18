import scipy.io as sio
import numpy as np
import h5py
import os
import cv2
import xml.etree.ElementTree as ElementTree

subject_list = [1, 5, 6, 7, 8, 9, 11]
action_list = np.arange(2, 17)
subaction_list = np.arange(1, 3)
camera_list = np.arange(1, 5)

root_dir = '/home/project/datasets/Human3.6/'
xml_path = "/home/project/datasets/Human3.6/metadata.xml"
SAVE_PATH = root_dir + '/images_fg/'

annot_path = '/home/project/Pose-Estimation/images/'
annot_name = 'matlab_meta.mat'

xml_file = ElementTree.parse(xml_path)
xml_mapping = xml_file.find('mapping')
xml_cameras = xml_file.find('dbcameras')

for subject in subject_list:
    for action in action_list:
        for subaction in subaction_list:
            for camera in camera_list:

                if (subject < 11):
                    print("Pass suubject :{0:d}".format(subject))
                    continue

                vid_folder = root_dir + "S" + str(subject) + "/Videos/"
                bg_vid_folder = root_dir + "/S" + str(subject) + "/ground_truth_bs/"
                save_folder_name = 's_{:02d}_act_{:02d}_subact_{:02d}_ca_{:02d}'.format(subject, action, subaction,
                                                                                        camera)

                if (os.path.exists(os.path.join(SAVE_PATH, save_folder_name)) != True):
                    os.makedirs(os.path.join(SAVE_PATH, save_folder_name))

                # fetch relevant video . . code converted from matlab code in h36m release code
                vid_name = xml_mapping[int(action * 2 + subaction - 2)][int(subject + 1)].text + '.' + str(
                    xml_cameras[0][int(camera - 1)].text) + '.mp4'

                print("Processing file S{0:d}: {1} ".format(subject, vid_name))

                try:
                    vid_file = os.path.join(vid_folder, vid_name)
                    bg_vid_file = os.path.join(bg_vid_folder, vid_name)

                    vidcap = cv2.VideoCapture(vid_file)
                    bg_vidcap = cv2.VideoCapture(bg_vid_file)
                except:
                    print("Pass vid : {}".format(vid_name))
                    continue

                # process annot file for fetching bounding box annotaions
                annot_file = annot_path + save_folder_name + '/' + annot_name
                try:
                    data = sio.loadmat(annot_file)
                except:
                    continue
                bboxx = data['bbox'].transpose(1, 0)

                index, num_proc = 0, 0
                while (True):
                    success_img, image = vidcap.read()
                    success_bg_img, bg_image = bg_vidcap.read()

                    index += 1

                    if (success_img != success_bg_img):
                        print('Invalid background frame for {0} : {1:06d}'.format(vid_file, index))
                        break

                    success = success_bg_img and success_img

                    if (success):

                        if ((index - 1) % 5 != 0):
                            continue

                        num_proc += 1

                        bg_image[bg_image > 100] = 1
                        fg_image = np.multiply(image, bg_image)

                        # bounding box computation . . converted from matlab code directly
                        bb = bboxx[index - 1, :].astype(np.int32)
                        bb[bb < 0] = 0
                        bb[2] = min(bb[2], image.shape[1]);
                        bb[3] = min(bb[3], image.shape[0]);
                        bb = np.round(bb)

                        if (bb[3] - bb[1] > bb[2] - bb[0]):
                            PAD = ((bb[3] - bb[1]) - (bb[2] - bb[0])) / 2;
                            bb[2] = bb[2] + PAD;
                            bb[0] = bb[0] - PAD;
                        else:
                            PAD = ((bb[2] - bb[0]) - (bb[3] - bb[1])) / 2;
                            bb[3] = bb[3] + PAD;
                            bb[1] = bb[1] - PAD;

                        bb[bb < 0] = 0
                        bb[2] = min(bb[2], image.shape[1]);
                        bb[3] = min(bb[3], image.shape[0]);
                        bb = np.round(bb)

                        fg_image = cv2.resize(fg_image[bb[1]:bb[3], bb[0]:bb[2], :], (224, 224))
                        file_name = '{}/{}/{}_{:06d}.jpg'.format(SAVE_PATH, save_folder_name, save_folder_name, index)
                        cv2.imwrite(file_name, fg_image)
                        # cv2.imshow(file_name, fg_image)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    else:
                        break

                print("Finished . Processed {0:d} frames".format(num_proc))



