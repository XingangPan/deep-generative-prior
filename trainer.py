import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import models
import utils
from dataset import ImageDataset


class Trainer(object):

    def __init__(self, config):
        self.rank, self.world_size = 0, 1
        if config['dist']:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.mode = config['dgp_mode']
        assert self.mode in [
            'reconstruct', 'colorization', 'SR', 'hybrid', 'inpainting',
            'morphing', 'defence', 'jitter'
        ]

        if self.rank == 0:
            # mkdir path
            if not os.path.exists('{}/images'.format(config['exp_path'])):
                os.makedirs('{}/images'.format(config['exp_path']))
            if not os.path.exists('{}/images_sheet'.format(
                    config['exp_path'])):
                os.makedirs('{}/images_sheet'.format(config['exp_path']))
            if not os.path.exists('{}/logs'.format(config['exp_path'])):
                os.makedirs('{}/logs'.format(config['exp_path']))

            # prepare logger
            if not config['no_tb']:
                try:
                    from tensorboardX import SummaryWriter
                except ImportError:
                    raise Exception("Please switch off \"tensorboard\" "
                                    "in your config file if you do not "
                                    "want to use it, otherwise install it.")
                self.tb_logger = SummaryWriter('{}'.format(config['exp_path']))
            else:
                self.tb_logger = None

            self.logger = utils.create_logger(
                'global_logger',
                '{}/logs/log_train.txt'.format(config['exp_path']))

        self.model = models.DGP(config)
        if self.mode == 'morphing':
            self.model2 = models.DGP(config)
            self.model_interp = models.DGP(config)

        # Data loader
        train_dataset = ImageDataset(
            config['root_dir'],
            config['list_file'],
            image_size=config['resolution'],
            normalize=True)
        sampler = utils.DistributedSampler(
            train_dataset) if config['dist'] else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            sampler=sampler,
            num_workers=1,
            pin_memory=False)
        self.config = config

    def run(self):
        # train
        if self.mode == 'morphing':
            self.train_morphing()
        else:
            self.train()

    def train(self):
        btime_rec = utils.AverageMeter()
        dtime_rec = utils.AverageMeter()
        recorder = {}
        end = time.time()
        for i, (image, category, img_path) in enumerate(self.train_loader):
            # measure data loading time
            dtime_rec.update(time.time() - end)

            torch.cuda.empty_cache()

            image = image.cuda()
            category = category.cuda()
            img_path = img_path[0]

            self.model.reset_G()
            self.model.set_target(image, category, img_path)
            # when category is unkonwn (category=-1), it would be selected from samples
            self.model.select_z(select_y=True if category.item() < 0 else False)
            loss_dict = self.model.run(save_interval=self.config['save_interval'])

            # average loss if distributed
            if self.config['dist']:
                for k, v in loss_dict.items():
                    reduced = v.data.clone() / self.world_size
                    dist.all_reduce_multigpu([reduced])
                    loss_dict[k] = reduced

            if len(recorder) == 0:
                for k in loss_dict.keys():
                    recorder[k] = utils.AverageMeter()
            for k in loss_dict.keys():
                recorder[k].update(loss_dict[k].item())

            btime_rec.update(time.time() - end)
            end = time.time()

            # logging
            loss_str = ""
            if self.rank == 0:
                for k in recorder.keys():
                    if self.tb_logger is not None:
                        self.tb_logger.add_scalar('train_{}'.format(k),
                                                  recorder[k].avg, i + 1)
                    loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})  '.format(
                        k, loss=recorder[k])

                self.logger.info(
                    'Iter: [{0}/{1}]  '.format(i + 1, len(self.train_loader)) +
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '.
                    format(batch_time=btime_rec) +
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})  '.format(
                        data_time=dtime_rec) + 'Image {}  '.format(img_path) +
                    loss_str)

    def train_morphing(self):
        btime_rec = utils.AverageMeter()
        dtime_rec = utils.AverageMeter()
        recorder = {}
        last_category = -1
        end = time.time()
        for i, (image, category, img_path) in enumerate(self.train_loader):
            # measure data loading time
            dtime_rec.update(time.time() - end)

            assert image.shape[0] > 0
            image = image.cuda()
            category = category.cuda()
            img_path = img_path[0]

            self.model.reset_G()
            self.model.set_target(image, category, img_path)
            self.model.select_z()
            loss_dict = self.model.run(
                save_interval=self.config['save_interval'])

            # apply image morphing within the same category
            if category == last_category:
                self.morphing()
            torch.cuda.empty_cache()

            with torch.no_grad():
                self.model2.G.load_state_dict(self.model.G.state_dict())
                self.model2.z.copy_(self.model.z)
                self.model2.img_name = self.model.img_name
                self.model2.target = self.model.target
                self.model2.category = self.model.category

            if category == last_category:
                # average loss if distributed
                if self.config['dist']:
                    for k, v in loss_dict.items():
                        reduced = v.data.clone() / self.world_size
                        dist.all_reduce_multigpu([reduced])
                        loss_dict[k] = reduced

                if len(recorder) < len(loss_dict):
                    for k in loss_dict.keys():
                        recorder[k] = utils.AverageMeter()
                for k in loss_dict.keys():
                    recorder[k].update(loss_dict[k].item())

                btime_rec.update(time.time() - end)
                end = time.time()

                # logging
                loss_str = ""
                if self.rank == 0:
                    for k in recorder.keys():
                        if self.tb_logger is not None:
                            self.tb_logger.add_scalar('train_{}'.format(k),
                                                      recorder[k].avg, i + 1)
                        loss_str += '{}: {loss.val:.4g} ({loss.avg:.4g})  '.format(
                            k, loss=recorder[k])

                    self.logger.info(
                        'Iter: [{0}/{1}]  '.format(i, len(self.train_loader)) +
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '.
                        format(batch_time=btime_rec) +
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})  '.
                        format(data_time=dtime_rec) +
                        'Image {}  '.format(img_path) + loss_str)

            last_category = category

    def morphing(self):
        weight1 = self.model.G.state_dict()
        weight2 = self.model2.G.state_dict()
        weight_interp = OrderedDict()
        imgs = []
        with torch.no_grad():
            for i in range(11):
                alpha = i / 10
                # interpolate between both latent vector and generator weight
                z_interp = alpha * self.model.z + (1 - alpha) * self.model2.z
                for k, w1 in weight1.items():
                    w2 = weight2[k]
                    weight_interp[k] = alpha * w1 + (1 - alpha) * w2
                self.model_interp.G.load_state_dict(weight_interp)
                x_interp = self.model_interp.G(
                    z_interp, self.model_interp.G.shared(self.model.y))
                imgs.append(x_interp.cpu())
                # save image
                save_path = '%s/images/%s_%s' % (self.config['exp_path'],
                                                 self.model.img_name,
                                                 self.model2.img_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                utils.save_img(x_interp[0], '%s/%03d.jpg' % (save_path, i + 1))
            imgs = torch.cat(imgs, 0)
        vutils.save_image(
            imgs,
            '%s/images_sheet/morphing_class%d_%s_%s.jpg' %
            (self.config['exp_path'], self.model.category, self.model.img_name,
             self.model2.img_name),
            nrow=int(imgs.size(0)**0.5),
            normalize=True)
        del weight_interp
