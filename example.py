import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from collections import OrderedDict

import torch
import torchvision.utils as vutils

import utils
from models import DGP

sys.path.append("./")


# Arguments for demo
def add_example_parser(parser):
    parser.add_argument(
        '--image_path', type=str, default='',
        help='Path of the image to be processed (default: %(default)s)')
    parser.add_argument(
        '--class', type=int, default=-1,
        help='class index of the image (default: %(default)s)')
    parser.add_argument(
        '--image_path2', type=str, default='',
        help='Path of the 2nd image to be processed, used in "morphing" mode (default: %(default)s)')
    parser.add_argument(
        '--class2', type=int, default=-1,
        help='class index of the 2nd image, used in "morphing" mode (default: %(default)s)')
    return parser


# prepare arguments and save in config
parser = utils.prepare_parser()
parser = utils.add_dgp_parser(parser)
parser = add_example_parser(parser)
config = vars(parser.parse_args())
utils.dgp_update_config(config)

# set random seed
utils.seed_rng(config['seed'])

if not os.path.exists('{}/images'.format(config['exp_path'])):
    os.makedirs('{}/images'.format(config['exp_path']))
if not os.path.exists('{}/images_sheet'.format(config['exp_path'])):
    os.makedirs('{}/images_sheet'.format(config['exp_path']))

# initialize DGP model
dgp = DGP(config)

# prepare the target image
img = utils.get_img(config['image_path'], config['resolution']).cuda()
category = torch.Tensor([config['class']]).long().cuda()
dgp.set_target(img, category, config['image_path'])

# prepare initial latent vector
dgp.select_z(select_y=True if config['class'] < 0 else False)
# start reconstruction
loss_dict = dgp.run()

if config['dgp_mode'] == 'category_transfer':
    save_imgs = img.clone().cpu()
    for i in range(151, 294):  # dog & cat
    # for i in range(7, 25):  # bird
        with torch.no_grad():
            x = dgp.G(dgp.z, dgp.G.shared(dgp.y.fill_(i)))
            utils.save_img(
                x[0],
                '%s/images/%s_class%d.jpg' % (config['exp_path'], dgp.img_name, i))
            save_imgs = torch.cat((save_imgs, x.cpu()), dim=0)
    vutils.save_image(
        save_imgs,
        '%s/images_sheet/%s_categories.jpg' % (config['exp_path'], dgp.img_name),
        nrow=int(save_imgs.size(0)**0.5),
        normalize=True)

elif config['dgp_mode'] == 'morphing':
    dgp2 = DGP(config)
    dgp_interp = DGP(config)

    img2 = utils.get_img(config['image_path2'], config['resolution']).cuda()
    category2 = torch.Tensor([config['class2']]).long().cuda()

    dgp2.set_target(img2, category2, config['image_path2'])
    dgp2.select_z(select_y=True if config['class2'] < 0 else False)
    loss_dict = dgp2.run()

    weight1 = dgp.G.state_dict()
    weight2 = dgp2.G.state_dict()
    weight_interp = OrderedDict()
    save_imgs = []
    with torch.no_grad():
        for i in range(11):
            alpha = i / 10
            # interpolate between both latent vector and generator weight
            z_interp = alpha * dgp.z + (1 - alpha) * dgp2.z
            y_interp = alpha * dgp.G.shared(dgp.y) + (1 - alpha) * dgp2.G.shared(dgp2.y)
            for k, w1 in weight1.items():
                w2 = weight2[k]
                weight_interp[k] = alpha * w1 + (1 - alpha) * w2
            dgp_interp.G.load_state_dict(weight_interp)
            x_interp = dgp_interp.G(z_interp, y_interp)
            save_imgs.append(x_interp.cpu())
            # save images
            save_path = '%s/images/%s_%s' % (config['exp_path'], dgp.img_name, dgp2.img_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            utils.save_img(x_interp[0], '%s/%03d.jpg' % (save_path, i + 1))
        save_imgs = torch.cat(save_imgs, 0)
    vutils.save_image(
        save_imgs,
        '%s/images_sheet/morphing_%s_%s.jpg' % (config['exp_path'], dgp.img_name, dgp2.img_name),
        nrow=int(save_imgs.size(0)**0.5),
        normalize=True)
