from __future__ import print_function
import argparse
import logging
import sys
import time, cv2
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
parser.add_argument('--constrain_feat', action='store_true', help='whether to use z loss')
parser.add_argument('--constrain_identity', action='store_true', help='whether to use identity loss')
parser.add_argument('--constrain_modality', action='store_true', help='whether to use modal loss')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--img_w', default=72, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=144, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch_size', default=2, type=int, metavar='B', help='training batch size')
parser.add_argument('--test_batch', default=2, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw, adp')
parser.add_argument('--norm_type', type=str, default='instance', choices=('instance', 'batch', 'group'))
parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in an image.')
parser.add_argument('--num_channels_d', default=64, type=int, help='Number of filters in the discriminator.')
parser.add_argument('--kernel_size_d', default=4, type=int, help='Size of the discriminator\'s kernels.')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=8, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--total_epoch', default=100, type=int, metavar='t', help='random seed')
parser.add_argument('--diffusion_epoch', default=50, type=int, metavar='t', help='random seed')
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

parser.add_argument('--model_prefix', type=str, default='Diffusion-middle', help='prefix for saved model')
parser.add_argument('--num_diffusion_timesteps', default=1000, type=int, help='')
parser.add_argument("--timesteps", type=int, default=20, help="number of steps involved")
parser.add_argument('--epoch', default=50, type=int, help='loading epoch')


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
                                batch_size=args.batch_size*args.num_pos,
                                image_size=(args.img_h, args.img_w),
                                num_workers=args.workers)
# visible images  infrared images
gallery_loader, query_loader = get_test_loader(dataset=args.dataset,
                                               root=args.data_path,
                                               batch_size=16,
                                               image_size=(args.img_h, args.img_w),
                                               num_workers=args.workers)
logger.info("--->>> length of train loader   %d " % (len(train_loader)))
logger.info("--->>> length of gallery loader %d " % (len(gallery_loader)))
logger.info("--->>> length of query loader   %d " % (len(query_loader)))


####3 initialize diffusion model
DiffusionModel = ConditionDiffusion(args)
epoch = args.epoch
path_models = args.model_prefix + '-epoch_' + str(epoch) + '_models.pth'
checkpoint = torch.load(os.path.join(args.model_path, path_models))
DiffusionModel.generator.model.load_state_dict(checkpoint['generator'])
DiffusionModel.eval()
DiffusionModel.generator.model.eval()
logger.info("--->>> load models of epoch %d successfully!" % (epoch))
save_dir = os.path.join('./save_images', "Diffusion-middle")


#### 4 save training images
for idx, (img, img_HF, label, cam, path, item) in enumerate(train_loader):
    bs = img.shape[0]
    for i in range(bs):
        img_i = img[i].unsqueeze(0)
        img_HF_i = img_HF[i].unsqueeze(0)
        path_i = path[i]   # ../data/sysu/cam1/0140/0010.jpg
        path_i_item = path_i.split('/')
        if path_i_item[3] in ['cam1', 'cam2', 'cam4', 'cam5']:  # rgb->mid
            # for the true visible images
            rgb_dir = os.path.join(save_dir, 'true', path_i_item[-3], path_i_item[-2])
            if os.path.exists(rgb_dir):
                pass
            else:
                os.makedirs(rgb_dir)
            rgb_path = os.path.join(rgb_dir, path_i_item[-1])
            # for the generated infrared images
            rgb2mid_dir = os.path.join(save_dir, 'middle', path_i_item[-3], path_i_item[-2])
            if os.path.exists(rgb2mid_dir):
                pass
            else:
                os.makedirs(rgb2mid_dir)
            rgb2mid_path = os.path.join(rgb2mid_dir, path_i_item[-1])
            # network propagation
            rgb_i = img_i
            rgb_HF_i = img_HF_i
            target_i = label[i]
            DiffusionModel.set_inputs(rgb_i, rgb_HF_i, rgb_i, rgb_HF_i, target_i, target_i)
            rgb2mid = DiffusionModel.generator.sample(x0=DiffusionModel.ir, con_x=DiffusionModel.rgb_HF, modality="mid")
            rgb_i = rgb_i.detach().cpu().squeeze(0)
            rgb2mid_i = rgb2mid.detach().cpu().squeeze(0)
            rgb_i = torch.cat((rgb_i[2,:,:].unsqueeze(0), \
                               rgb_i[1,:,:].unsqueeze(0), \
                               rgb_i[0,:,:].unsqueeze(0)), dim=0)
            rgb_i = rgb_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            rgb2mid_i = torch.cat((rgb2mid_i[2,:,:].unsqueeze(0), \
                                   rgb2mid_i[1,:,:].unsqueeze(0), \
                                   rgb2mid_i[0,:,:].unsqueeze(0)), dim=0)
            rgb2mid_i = rgb2mid_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            cv2.imwrite(rgb_path, rgb_i)
            cv2.imwrite(rgb2mid_path, rgb2mid_i)

        elif path_i_item[3] in ['cam3', 'cam6']:  # ir->mid
            # for the true visible images
            ir_dir = os.path.join(save_dir, 'true', path_i_item[-3], path_i_item[-2])
            if os.path.exists(ir_dir):
                pass
            else:
                os.makedirs(ir_dir)
            ir_path = os.path.join(ir_dir, path_i_item[-1])
            # for the generated infrared images
            ir2mid_dir = os.path.join(save_dir, 'middle', path_i_item[-3], path_i_item[-2])
            if os.path.exists(ir2mid_dir):
                pass
            else:
                os.makedirs(ir2mid_dir)
            ir2mid_path = os.path.join(ir2mid_dir, path_i_item[-1])
            # network propagation
            ir_i = img_i
            ir_HF_i = img_HF_i
            target_i = label[i]
            DiffusionModel.set_inputs(ir_i, ir_HF_i, ir_i, ir_HF_i, target_i, target_i)
            ir2mid = DiffusionModel.generator.sample(x0=DiffusionModel.ir, con_x=DiffusionModel.ir_HF, modality="mid")
            ir2mid_i = ir2mid.detach().cpu()
            ir_i = ir_i.detach().cpu().squeeze(0)
            ir2mid_i = ir2mid_i.detach().cpu().squeeze(0)
            ir_i = torch.cat((ir_i[2,:,:].unsqueeze(0), \
                              ir_i[1,:,:].unsqueeze(0), \
                              ir_i[0,:,:].unsqueeze(0)), dim=0)
            ir_i = ir_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            ir2mid_i = torch.cat((ir2mid_i[2,:,:].unsqueeze(0), \
                                  ir2mid_i[1,:,:].unsqueeze(0), \
                                  ir2mid_i[0,:,:].unsqueeze(0)), dim=0)
            ir2mid_i = ir2mid_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            cv2.imwrite(ir_path, ir_i)
            cv2.imwrite(ir2mid_path, ir2mid_i)

    if idx%10==0:
        print(idx)


#### 5 save gallery images  rgb->mid
for idx, (img, img_HF, label, cam, path, item) in enumerate(gallery_loader):
    print('gallery loader idx ', idx)
    path_rgb = path
    img_HF = img_HF.cuda()
    img_rgb = img.cuda()
    DiffusionModel.set_inputs(img_rgb, img_HF, img_rgb, img_HF, label, label)
    # generate ir
    rgb2mid = DiffusionModel.generator.sample(x0=DiffusionModel.ir, con_x=DiffusionModel.rgb_HF, modality="mid")
    rgb2mid = rgb2mid.detach().cpu()
    img_rgb = img_rgb.detach().cpu()

    # save images
    num_images_one_iter = len(path_rgb)
    for i in range(num_images_one_iter):
        # for the true visible images
        rgb_name = path_rgb[i].split('/')        # '../data/sysu/cam5/0340/0008.jpg'
        rgb_dir = os.path.join(save_dir, 'true', rgb_name[-3], rgb_name[-2])
        if os.path.exists(rgb_dir):
            pass
        else:
            os.makedirs(rgb_dir)
        rgb_path = os.path.join(rgb_dir, rgb_name[-1])
        # for the generated infrared images
        rgb2mid_dir = os.path.join(save_dir, 'middle', rgb_name[-3], rgb_name[-2])
        if os.path.exists(rgb2mid_dir):
            pass
        else:
            os.makedirs(rgb2mid_dir)
        rgb2mid_path = os.path.join(rgb2mid_dir, rgb_name[-1])

        rgb_i = torch.cat((img_rgb[i][2,:,:].unsqueeze(0), \
                           img_rgb[i][1,:,:].unsqueeze(0), \
                           img_rgb[i][0,:,:].unsqueeze(0)), dim=0)
        rgb_i = rgb_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        rgb2mid_i = torch.cat((rgb2mid[i][2,:,:].unsqueeze(0), \
                               rgb2mid[i][1,:,:].unsqueeze(0), \
                               rgb2mid[i][0,:,:].unsqueeze(0)), dim=0)
        rgb2mid_i = rgb2mid_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        cv2.imwrite(rgb_path, rgb_i)
        cv2.imwrite(rgb2mid_path, rgb2mid_i)


#### 6 save query images  ir->mid
for idx, (img, img_HF, label, cam, path, item) in enumerate(query_loader):
    print('query loader idx ', idx)
    path_ir = path
    img_HF = img_HF.cuda()
    img_ir = img.cuda()
    DiffusionModel.set_inputs(img_ir, img_HF, img_ir, img_HF, label, label)
    # generate ir
    ir2mid = DiffusionModel.generator.sample(x0=DiffusionModel.ir, con_x=DiffusionModel.ir_HF, modality="mid")
    ir2mid = ir2mid.detach().cpu()
    img_ir = img_ir.detach().cpu()
    # save images
    num_images_one_iter = len(path_ir)
    for i in range(num_images_one_iter):
        # for the true visible images
        ir_name = path_ir[i].split('/')        # '../data/sysu/cam6/0340/0008.jpg'
        ir_dir = os.path.join(save_dir, 'true', ir_name[-3], ir_name[-2])
        if os.path.exists(ir_dir):
            pass
        else:
            os.makedirs(ir_dir)
        ir_path = os.path.join(ir_dir, ir_name[-1])
        # for the generated infrared images
        ir2mid_dir = os.path.join(save_dir, 'middle', ir_name[-3], ir_name[-2])
        if os.path.exists(ir2mid_dir):
            pass
        else:
            os.makedirs(ir2mid_dir)
        ir2mid_path = os.path.join(ir2mid_dir, ir_name[-1])

        ir_i = torch.cat((img_ir[i][2,:,:].unsqueeze(0), \
                          img_ir[i][1,:,:].unsqueeze(0), \
                          img_ir[i][0,:,:].unsqueeze(0)), dim=0)
        ir_i = ir_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        ir2mid_i = torch.cat((ir2mid[i][2,:,:].unsqueeze(0), \
                              ir2mid[i][1,:,:].unsqueeze(0), \
                              ir2mid[i][0,:,:].unsqueeze(0)), dim=0)
        ir2mid_i = ir2mid_i.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        cv2.imwrite(ir_path, ir_i)
        cv2.imwrite(ir2mid_path, ir2mid_i)


