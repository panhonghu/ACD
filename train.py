from __future__ import print_function
import argparse
import logging
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from dataset import get_train_loader, get_test_loader
from eval_metrics import eval_sysu, eval_regdb
from utils import *
import pdb
import scipy.io as scio
from models import ConditionDiffusion
import warnings
warnings.filterwarnings("ignore")


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu')
parser.add_argument('--data_path', default='../data/sysu', help='dataset path')
parser.add_argument('--sample_method', default='identity_random', help='method to sample data')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test_only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--model_prefix', type=str, default='ConditionDiffusion', help='prefix for saved model')
parser.add_argument('--constrain_feat', action='store_true', help='whether to use z loss')
parser.add_argument('--constrain_identity', action='store_true', help='whether to use identity loss')
parser.add_argument('--constrain_modality', action='store_true', help='whether to use modal loss')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--img_w', default=72, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=144, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=4, type=int, metavar='B', help='training batch size')
parser.add_argument('--test_batch', default=8, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw, adp')
parser.add_argument('--norm_type', type=str, default='instance', choices=('instance', 'batch', 'group'))
parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in an image.')
parser.add_argument('--num_channels_d', default=64, type=int, help='Number of filters in the discriminator.')
parser.add_argument('--kernel_size_d', default=4, type=int, help='Size of the discriminator\'s kernels.')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=8, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--total_epoch', default=70, type=int, metavar='t', help='random seed')
parser.add_argument('--diffusion_only_epoch', default=50, type=int, metavar='t', help='random seed')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--augc', default=0 , type=int, metavar='aug', help='use channel aug or not')
parser.add_argument('--rande', default= 0 , type=float, metavar='ra', help='use random erasing or not')
parser.add_argument('--kl', default= 0 , type=float, metavar='kl', help='use kl loss and the weight')
parser.add_argument('--alpha', default=1 , type=int, metavar='alpha', help='magnification for the hard mining')
parser.add_argument('--gamma', default=1 , type=int, metavar='gamma', help='gamma for the hard mining')
parser.add_argument('--square', default= 1 , type=int, metavar='square', help='gamma for the hard mining')
parser.add_argument('--is_training', default=True, type=bool, help='training or testing')
parser.add_argument('--initializer', default='normal', type=str, help='method to initialize flow models')
parser.add_argument('--weight_norm_l2', type=float, default=5e-5, help='L2 regularization factor for weight norm')
parser.add_argument('--lr_G', default=0.0001 , type=float, help='learning rate')
parser.add_argument('--lr_S', default=0.0001 , type=float, help='learning rate')
parser.add_argument('--beta_1', default=0.5 , type=float, help='beta_1 for adam')
parser.add_argument('--beta_2', default=0.999 , type=float, help='beta_2 for adam')
parser.add_argument('--lr_policy', type=str, default='linear', help='Learning rate schedule policy', choices=('linear', 'plateau', 'step'))
parser.add_argument('--lr_step_epochs', type=int, default=100, help='Number of epochs between each divide-by-10 step (step policy only).')
parser.add_argument('--lr_warmup_epochs', type=int, default=100, help='Number of epochs before we start decaying the learning rate (linear only).')
parser.add_argument('--lr_decay_epochs', type=int, default=100, help='Number of epochs to decay the learning rate linearly to 0 (linear only).')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--log_file', type=str, default='log.txt')
#### diffusion model parameters
parser.add_argument('--beta_schedule', default='linear', type=str, help='')
parser.add_argument('--beta_start', default=0.0001, type=float, help='')
parser.add_argument('--beta_end', default=0.02, type=float, help='')
parser.add_argument('--model_ema', action='store_false', help='')
parser.add_argument('--type', default='simple', type=str, help='')
parser.add_argument('--var_type', default='fixedlarge', type=str, help='')
parser.add_argument('--in_channels', type=int, default=4, help='')
parser.add_argument('--out_ch', type=int, default=3, help='')
parser.add_argument('--ch_mult', type=list, default=[1, 2, 2, 2], help='')
parser.add_argument('--ch', type=int, default=128, help='')
parser.add_argument('--num_res_blocks', type=int, default=2, help='')
parser.add_argument('--attn_resolutions', type=list, default=[36, ], help='')
parser.add_argument('--dropout', default=0.1, type=float, help='')
parser.add_argument('--ema_rate', default=0.9999, type=float, help='')
parser.add_argument('--ema', action='store_false', help='')
parser.add_argument('--resamp_with_conv', action='store_false', help='')
parser.add_argument("--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)",)
parser.add_argument("--eta", type=float, default=0.0, help="eta used to control the variances of sigma",)
parser.add_argument('--num_diffusion_timesteps', default=1000, type=int, help='')
parser.add_argument("--timesteps", type=int, default=20, help="number of steps involved")
parser.add_argument('--g_steps', default=2, type=int, help='')
parser.add_argument('--d_steps', default=1, type=int, help='')
parser.add_argument('--middle', action='store_true', help='whether to generate middle-modality images')


### training args and logger settings
args = parser.parse_args()
set_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if os.path.exists(args.log_path):
    pass
else:
    os.mkdir(args.log_path)
logger = get_logger(os.path.join(args.log_path, args.log_file))
logger.info("==========\nArgs:{}\n==========".format(args))

## dataset
train_loader = get_train_loader(dataset=args.dataset,
                                root=args.data_path,
                                sample_method=args.sample_method,
                                batch_size=args.batch_size*args.num_pos,
                                p_size=args.batch_size,
                                k_size=args.num_pos,
                                random_flip=False,
                                random_crop=False,
                                random_erase=False,
                                color_jitter=False,
                                padding=10,
                                image_size=(args.img_h, args.img_w),
                                num_workers=args.workers)

####3 initialize flow model
DiffusionModel = ConditionDiffusion(args)
logger.info("--->>> construct model successfully!")
# # checkpoint = torch.load(os.path.join(args.model_path, "pretrained_ConditionDiffusion.pth"))
# # DiffusionModel.generator.model.load_state_dict(checkpoint['generator'])
# checkpoint = torch.load(os.path.join(args.model_path, "Diffusion-middle-g_1_d_2-epoch_14_models.pth"))
# DiffusionModel.generator.model.load_state_dict(checkpoint['generator'])
# DiffusionModel.d_rgb.load_state_dict(checkpoint['d_rgb'])
# DiffusionModel.d_ir.load_state_dict(checkpoint['d_ir'])
# logger.info("--->>> load pretrained model successfully!")

# print('len --> ', len(train_loader))
for epoch in range(15, args.total_epoch+1):
    ## odd index for infrared, even index for visible
    for idx, (img, img_HF, label, cam, path, item) in enumerate(train_loader):
        bs = img.shape[0]
        ir_idx = list(range(0, bs, 2))
        rgb_idx = list(range(1, bs, 2))
        img_rgb = img[rgb_idx]
        img_rgb_HF = img_HF[rgb_idx]
        img_ir = img[ir_idx]
        img_ir_HF = img_HF[ir_idx]
        target_rgb = label[rgb_idx]
        target_ir = label[ir_idx]
        path_rgb = [path[x] for x in rgb_idx]
        path_ir = [path[x] for x in ir_idx]
        # print('idx -> ', idx)
        DiffusionModel.set_inputs(img_rgb, img_rgb_HF, img_ir, img_ir_HF, target_rgb, target_ir)
        if args.middle:
            loss_g, loss_d = DiffusionModel.train_iter_middle(epoch)
        else:
            loss_g, loss_d = DiffusionModel.train_iter(epoch)

    logger.info('----->>>>> training model of epoch -> %d' % epoch)
    logger.info('loss_g -> %f' % loss_g)
    logger.info('loss_d -> %f' % loss_d)

    ## same models
    if epoch%args.save_epoch==0:
        if os.path.exists(args.model_path):
            pass
        else:
            os.mkdir(args.model_path)
        model_name = args.model_prefix + '-epoch_' + str(epoch) + '_models.pth'
        torch.save({'generator': DiffusionModel.generator.model.state_dict(), \
                    'd_rgb': DiffusionModel.d_rgb.state_dict(), \
                    'd_ir': DiffusionModel.d_ir.state_dict(), \
                   }, os.path.join(args.model_path, model_name))
        logger.info("--->>> save models of epoch %d successfully!" % (epoch))


