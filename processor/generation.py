#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
import pdb

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class GEN_Processor(Processor):
    """
        Processor for Graph data generation
    """

    def load_model(self):
        self.generator = self.io.load_model(self.arg.generator,
                                        **(self.arg.gen_args))
        self.discriminator = self.io.load_model(self.arg.discriminator,
                                        **(self.arg.dis_args))
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)


        self.l1_loss = nn.L1Loss()
        self.sml1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer_gen = optim.SGD(
                self.generator.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

            self.optimizer_dis = optim.SGD(
                self.discriminator.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)

        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group_gen, param_group_dis in zip(self.optimizer_gen.param_groups, self.optimizer_dis.param_groups):
                param_group_gen['lr'] = lr
                param_group_dis['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.generator.train()
        self.discriminator.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        g_loss_value = []
        d_loss_value = []

        for data_list, label in loader:
            # get data
            data = data_list[0]
            data_mask = data_list[1]

            data = data.float().to(self.dev)
            data_mask = data_mask.float().to(self.dev)
            label = label.long().to(self.dev)

            # generator forward
            x_stage1, x_stage2 = self.generator(data, data_mask)
            gen_recon_loss = self.l1_loss(data, x_stage1) + self.l1_loss(data, x_stage2)

            # discriminator forward
            x_com = x_stage2*data_mask + data*(1-data_mask)
            input_pos_neg = torch.cat((data, x_com), 0)
            dis_pos_neg = self.discriminator(input_pos_neg)
            dis_pos, dis_neg = torch.split(dis_pos_neg, 2, 0)

            #  gan loss
            # https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/gan_ops.py
            # https://github.com/ozanciga/gans-with-pytorch/blob/master/wgan/wgan.py
            label_size = dis_pos.size()
            label_size[1:] = 1

            neg = np.zeros(label_size, np.int)
            pos = neg + 1

            pdb.set_trace()



    # def train(self):
    #     self.model.train()
    #     self.adjust_lr()
    #     loader = self.data_loader['train']
    #     loss_value = []
    #
    #     for data_list, label in loader:
    #
    #         # get data
    #         data = data_list[0]
    #         data_mask = data_list[1]
    #
    #         data = data.float().to(self.dev)
    #         data_mask = data_mask.int().to(self.dev)
    #         label = label.long().to(self.dev)
    #
    #         # forward
    #         output = self.model(data, data_mask)
    #         loss = self.loss(output, label)
    #
    #         # backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #
    #         # statistics
    #         self.iter_info['loss'] = loss.data.item()
    #         self.iter_info['lr'] = '{:.6f}'.format(self.lr)
    #         loss_value.append(self.iter_info['loss'])
    #         self.show_iter_info()
    #         self.meta_info['iter'] += 1
    #
    #     self.epoch_info['mean_loss']= np.mean(loss_value)
    #     self.show_epoch_info()
    #     self.io.print_timer()



    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data_list, label in loader:
            
            # get data
            data = data_list[0]
            data_mask = data_list[1]

            data = data.float().to(self.dev)
            data_mask = data_mask.int().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                output = self.model(data, data_mask)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.show_epoch_info()

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
