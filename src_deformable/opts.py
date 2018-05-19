import argparse
import os

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        """
            Define args that is used in project
        """
        parser = argparse.ArgumentParser(description="Pose guided image generation usign deformable skip layers")
        # adding arguments for expID and data directories lacking in originial codebase
        self.parser.add_argument('--expID', default='default', help='Experiment ID')
        self.parser.add_argument("--data_Dir", default='../../pose-gan-clean/pose-gan-h36m-fg/data/', help="Directory with annotations and data")

        self.parser.add_argument("--output_dir", default='output/displayed_samples', help="Directory with generated sample images")
        self.parser.add_argument("--batch_size", default=4, type=int, help='Size of the batch')
        self.parser.add_argument("--log_file", default="output/full/fasion/log", help="log file")
        self.parser.add_argument("--training_ratio", default=1, type=int,
                            help="The training ratio is the number of discriminator updates per generator update.")

        self.parser.add_argument("--resume", default=0, type=int, help="resume from checkpoint")

        self.parser.add_argument("--learning_rate", default=2e-4, type=float, help='learning rate of optimizer')
        self.parser.add_argument("--l1_penalty_weight", default=100, type=float, help='Weight of l1 loss')
        self.parser.add_argument('--gan_penalty_weight', default=1, type=float, help='Weight of GAN loss')
        self.parser.add_argument('--tv_penalty_weight', default=0, type=float, help='Weight of total variation loss')
        self.parser.add_argument('--lstruct_penalty_weight', default=0, type=float, help="Weight of lstruct")

        self.parser.add_argument("--number_of_epochs", default=500, type=int, help="Number of training epochs")

        self.parser.add_argument("--content_loss_layer", default='none', help='Name of content layer (vgg19)'
                                                                         ' e.g. block4_conv1 or none')
        # change pose_utils and pose_transfroms to honor this
        self.parser.add_argument("--pose_dim", default=16, type=int, help="Dimensionality of pose vector")
        self.parser.add_argument("--iters_per_epoch", default=1000, type=int, help="Dimensionality of pose vector")
        self.parser.add_argument("--checkpoints_dir", default="output/checkpoints", help="Folder with checkpoints")
        self.parser.add_argument("--checkpoint_ratio", default=5, type=int, help="Number of epochs between consecutive checkpoints")
        self.parser.add_argument("--generator_checkpoint", default=None, help="Previosly saved model of generator")
        self.parser.add_argument("--discriminator_checkpoint", default=None, help="Previosly saved model of discriminator")
        self.parser.add_argument("--nn_loss_area_size", default=1, type=int, help="Use nearest neighbour loss")
        self.parser.add_argument('--dataset', default='h36m', choices=['market', 'fasion', 'fasion128', 'fasion128128', 'h36m'],
                            help='Market or fasion or h36m')

        self.parser.add_argument("--frame_diff", default=10, type=int,  help='Number of frames for iterative testing . . increment by 1 to get 10 frames displacement from initial frame and adding another stack')
        self.parser.add_argument("--num_stacks", default=4, type=int, help='Number of stacks for interpol arch')

        self.parser.add_argument('--compute_h36m_paf_split', default=0, type=int,
                            help='which split to compute annot for')

        self.parser.add_argument("--display_ratio", default=50, type=int,  help='Number of epochs between ploting')
        self.parser.add_argument("--start_epoch", default=0, type=int, help='Start epoch for starting from checkpoint')
        self.parser.add_argument("--pose_estimator", default='pose_estimator.h5',
                                help='Pretrained model for cao pose estimator')

        self.parser.add_argument("--images_for_test", default=12000, type=int, help="Number of images for testing")

        self.parser.add_argument("--use_input_pose", default=True, type=int, help='Feed to generator input pose')
        self.parser.add_argument("--warp_skip", default='mask', choices=['none', 'full', 'mask'],
                            help="Type of warping skip layers to use.")
        self.parser.add_argument("--warp_agg", default='max', choices=['max', 'avg'],
                            help="Type of aggregation.")

        self.parser.add_argument("--disc_type", default='call', choices=['call', 'sim', 'warp'],
                            help="Type of discriminator call - concat all, sim - siamease, sharewarp - warp.")
        self.parser.add_argument("--gen_type", default='baseline', choices=['baseline', 'stacked'],
                                 help="Type of generator")

        self.parser.add_argument("--generated_images_dir", default='output/generated_images',
                            help='Folder with generated images from training dataset')

        self.parser.add_argument('--load_generated_images', default=0, type=int,
                            help='Load images from generated_images_dir or generate')

        self.parser.add_argument('--use_dropout_test', default=0, type=int,
                            help='To use dropout when generate images')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        self.opt.saveDir = os.path.join("../exp/", self.opt.expID)
        self.opt.output_dir = os.path.join("../exp/", self.opt.expID, "results")
        self.opt.checkpoints_dir = os.path.join("../exp/", self.opt.expID, 'models')

        args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                    if not name.startswith('_'))

        if self.opt.dataset == 'fasion':
            self.opt.image_size = (256, 256)
        elif self.opt.dataset == 'h36m':
            self.opt.image_size = (224, 224)
        elif self.opt.dataset == 'fasion128128':
            self.opt.image_size = (128, 128)
        else:
            self.opt.image_size = (128, 64)

        self.opt.generated_images_dir = 'output/generated_images/' + self.opt.dataset + "-restricted"
        
        self.opt.images_dir_train = self.opt.data_Dir + self.opt.dataset + '-dataset/train'
        self.opt.images_dir_test = self.opt.data_Dir + self.opt.dataset + '-dataset/test'

        # for shg ( 16 joints ) human 3.6 annotations
        self.opt.annotations_file_train = self.opt.data_Dir + self.opt.dataset + '-annotation-train.csv'
        self.opt.annotations_file_test = self.opt.data_Dir + self.opt.dataset + '-annotation-test.csv'

        # for paf ( 18 joints ) human 3.6 annotations
        self.opt.annotations_file_train_paf = self.opt.data_Dir + self.opt.dataset + '-annotation-paf-train' + str(
            self.opt.compute_h36m_paf_split) + '.csv'
        self.opt.annotations_file_test_paf = self.opt.data_Dir + self.opt.dataset + '-annotation-paf-test' + str(
            self.opt.compute_h36m_paf_split) + '.csv'

        self.opt.pairs_file_train = self.opt.data_Dir + self.opt.dataset + '-pairs-train.csv'
        self.opt.pairs_file_test = self.opt.data_Dir + self.opt.dataset + '-pairs-test.csv'
        self.opt.pairs_file_train_iterative = self.opt.data_Dir + self.opt.dataset + '-pairs-train-iterative.csv'
        self.opt.pairs_file_test_iterative = self.opt.data_Dir + self.opt.dataset + '-pairs-test-iterative.csv'

        self.opt.pairs_file_train_interpol = self.opt.data_Dir + self.opt.dataset + '-pairs-train-interpol.csv'
        self.opt.pairs_file_test_interpol = self.opt.data_Dir + self.opt.dataset + '-pairs-test-interpol.csv'

        self.opt.tmp_pose_dir = 'tmp/' + self.opt.dataset + '/'

        if not os.path.exists(self.opt.saveDir):
            os.makedirs(self.opt.saveDir)

        if not os.path.exists(self.opt.output_dir):
            os.makedirs(self.opt.output_dir)
            os.makedirs(os.path.join(self.opt.output_dir, 'train'))
            os.makedirs(os.path.join(self.opt.output_dir, 'test'))

        if not os.path.exists(self.opt.checkpoints_dir):
            os.makedirs(self.opt.checkpoints_dir)

        file_name = os.path.join(self.opt.saveDir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(vars(self.opt).items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')
        return self.opt
