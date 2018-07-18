# Pose-Transfer

PyTorch Implementation of Deformable GAN https://arxiv.org/abs/1801.00055
Also check out the original repo in Keras, by AliaksandrSiarohin - https://github.com/AliaksandrSiarohin/pose-gan

## Baseline Model ( run in src_baseline )

### Fasion :

CUDA_VISIBLE_DEVICES=1 nohup python main.py --l1_penalty_weight 10 --batch_size 4 --number_of_epochs 90 --gen_type baseline --expID baseline_fasion --pose_dim 18 --dataset fasion 

### Human 3.6 :

CUDA_VISIBLE_DEVICES=0 nohup python main.py --l1_penalty_weight 10 --batch_size 4 --number_of_epochs 90 --gen_type baseline --expID baseline_h36m 

## Deformable Model ( run in src_deformable )

### Fasion :

CUDA_VISIBLE_DEVICES=1 nohup python main.py --warp_skip mask --l1_penalty_weight 0.01 --batch_size 2 --number_of_epochs 90 --gen_type baseline --expID full_fasion --pose_dim 18 --dataset fasion --nn_loss_area_size 5 --batch_size 2 --content_loss_layer block1_conv2

### Human 3.6 :

CUDA_VISIBLE_DEVICES=1 nohup python main.py --warp_skip mask --l1_penalty_weight 0.01 --batch_size 2 --number_of_epochs 90 --gen_type baseline --expID full_fasion --dataset fasion --nn_loss_area_size 5 --batch_size 2 --content_loss_layer block1_conv2

Specify data directory by passing data_Dir option.

## ToDo :

-> Add code for loading annotations and train/test pairs for Fasion and Human 3.6 dataset.
