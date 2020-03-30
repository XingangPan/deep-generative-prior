import argparse
import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import dataset

from .biggan_utils import CenterCropLongEdge


# Arguments for DGP
def add_dgp_parser(parser):
    parser.add_argument(
        '--dist', action='store_true', default=False,
        help='Train with distributed implementation (default: %(default)s)')
    parser.add_argument(
        '--port', type=str, default='12345',
        help='Port id for distributed training (default: %(default)s)')
    parser.add_argument(
        '--exp_path', type=str, default='',
        help='Experiment path (default: %(default)s)')
    parser.add_argument(
        '--root_dir', type=str, default='',
        help='Root path of dataset (default: %(default)s)')
    parser.add_argument(
        '--list_file', type=str, default='',
        help='List file of the dataset (default: %(default)s)')
    parser.add_argument(
        '--resolution', type=int, default=256,
        help='Resolution to resize the input image (default: %(default)s)')
    parser.add_argument(
        '--dgp_mode', type=str, default='reconstruct',
        help='DGP mode (default: %(default)s)')
    parser.add_argument(
        '--random_G', action='store_true', default=False,
        help='Use randomly initialized generator? (default: %(default)s)')
    parser.add_argument(
        '--update_G', action='store_true', default=False,
        help='Finetune Generator? (default: %(default)s)')
    parser.add_argument(
        '--update_embed', action='store_true', default=False,
        help='Finetune class embedding? (default: %(default)s)')
    parser.add_argument(
        '--save_G', action='store_true', default=False,
        help='Save fine-tuned generator and latent vector? (default: %(default)s)')
    parser.add_argument(
        '--ftr_type', type=str, default='Discriminator',
        choices=['Discriminator', 'VGG'],
        help='Feature loss type, choose from Discriminator and VGG (default: %(default)s)')
    parser.add_argument(
        '--ftr_num', type=int, default=[3], nargs='+',
        help='Number of features to computer feature loss (default: %(default)s)')
    parser.add_argument(
        '--ft_num', type=int, default=[2], nargs='+',
        help='Number of parameter groups to finetune (default: %(default)s)')
    parser.add_argument(
        '--print_interval', type=int, default=100, nargs='+',
        help='Number of iterations to print training loss (default: %(default)s)')
    parser.add_argument(
        '--save_interval', type=int, default=None, nargs='+',
        help='Number of iterations to save image')
    parser.add_argument(
        '--lr_ratio', type=float, default=[1.0, 1.0, 1.0, 1.0], nargs='+',
        help='Decreasing ratio for learning rate in blocks (default: %(default)s)')
    parser.add_argument(
        '--w_D_loss', type=float, default=[0.1], nargs='+',
        help='Discriminator feature loss weight (default: %(default)s)')
    parser.add_argument(
        '--w_nll', type=float, default=0.001,
        help='Weight for the negative log-likelihood loss (default: %(default)s)')
    parser.add_argument(
        '--w_mse', type=float, default=[0.1], nargs='+',
        help='MSE loss weight (default: %(default)s)')
    parser.add_argument(
        '--select_num', type=int, default=500,
        help='Number of image pool to select from (default: %(default)s)')
    parser.add_argument(
        '--sample_std', type=float, default=1.0,
        help='Std of the gaussian distribution used for sampling (default: %(default)s)')
    parser.add_argument(
        '--iterations', type=int, default=[200, 200, 200, 200], nargs='+',
        help='Training iterations for all stages')
    parser.add_argument(
        '--G_lrs', type=float, default=[1e-6, 2e-5, 1e-5, 1e-6], nargs='+',
        help='Learning rate steps of Generator')
    parser.add_argument(
        '--z_lrs', type=float, default=[1e-1, 1e-3, 1e-5, 1e-6], nargs='+',
        help='Learning rate steps of latent code z')
    parser.add_argument(
        '--warm_up', type=int, default=0,
        help='Number of warmup iterations (default: %(default)s)')
    parser.add_argument(
        '--use_in', type=str2bool, default=[False, False, False, False], nargs='+',
        help='Whether to use instance normalization in generator')
    parser.add_argument(
        '--stop_mse', type=float, default=0.0,
        help='MSE threshold for stopping training (default: %(default)s)')
    parser.add_argument(
        '--stop_ftr', type=float, default=0.0,
        help='Feature loss threshold for stopping training (default: %(default)s)')
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_img(img_path, resolution):
    img = dataset.default_loader(img_path)
    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        CenterCropLongEdge(),
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    img = transform(img)
    return img.unsqueeze(0)


def save_img(image, path):
    image = np.uint8(255 * (image.cpu().detach().numpy() + 1) / 2.)
    image = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image)
    image.save(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def size_splits(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(
        tensor.narrow(int(dim), int(start), int(length))
        for start, length in zip(splits, split_sizes))


class LRScheduler(object):

    def __init__(self, optimizer, warm_up):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group=1000, ratio=1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i >= num_group:
                param_group['lr'] = 0
            else:
                param_group['lr'] = learning_rate * ratio**i


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


def map_func(storage, location):
    return storage.cuda()


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def bicubic_torch(img, size):
    img_pil = np_to_pil(torch_to_np(img))
    img_bicubic_pil = img_pil.resize(size, Image.BICUBIC)
    img_bicubic_pth = np_to_torch(pil_to_np(img_bicubic_pil))
    return img_bicubic_pth
