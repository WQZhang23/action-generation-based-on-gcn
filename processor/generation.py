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
import time

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

    # definition of the gan loss
    # https://github.com/JiahuiYu/neuralgym/blob/master/neuralgym/ops/gan_ops.py
    # https://github.com/ozanciga/gans-with-pytorch
    def gan_hinge_loss(self, pos, neg):
        self.relu = nn.ReLU()
        hinge_pos = self.relu(1-pos).mean()
        hinge_neg = self.relu(1+neg).mean()
        d_loss = .5*hinge_pos + .5*hinge_neg
        g_loss = -neg.mean()
        return g_loss, d_loss
        
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
            self.optimizer_gen = optim.Adam(
                self.generator.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

            self.optimizer_dis = optim.Adam(
                self.discriminator.parameters(),
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

            #dis_pos, dis_neg = torch.split(dis_pos_neg, 2, 0)

            dis_pos = dis_pos_neg[:,0]
            dis_neg = dis_pos_neg[:,1]

            g_gan_loss, d_gan_loss = self.gan_hinge_loss(dis_pos, dis_neg)
            g_loss_sum = self.arg.recon_loss_weight*gen_recon_loss + g_gan_loss
            d_loss_sum = d_gan_loss

            # backward_dis
            self.optimizer_dis.zero_grad()
            d_loss_sum.backward(retain_graph=True)
            self.optimizer_dis.step()

            # backward_gen
            self.optimizer_gen.zero_grad()
            g_loss_sum.backward()
            self.optimizer_gen.step()

            # statistics
            self.iter_info['g_loss'] = g_loss_sum.data.item()
            self.iter_info['d_loss'] = d_loss_sum.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            g_loss_value.append(self.iter_info['g_loss'])
            d_loss_value.append(self.iter_info['d_loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_g_loss'] = np.mean(g_loss_value)
        self.epoch_info['mean_d_loss'] = np.mean(d_loss_value)
        self.show_epoch_info()
        #self.io.print_timer()



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

        # self.model.eval()
        loader = self.data_loader['test']

        out_data_seq_list = []

        for data_list, label in loader:
            
            # get data
            data = data_list[0]
            data_mask = data_list[1]

            data = data.float().to(self.dev)
            data_mask = data_mask.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            x_stage1, x_stage2 = self.generator(data, data_mask)
            x_com = x_stage2 * data_mask + data * (1 - data_mask)
            # pdb.set_trace()
            np_data = data.cpu().numpy()
            np_data_mask = data_mask.cpu().numpy()
            np_x_com = x_com.cpu().detach().numpy()

            # shape [1,7,64(len_data),25(keypoints),2(person)]
            # axis=1 --> 7
            # [0:3]--> original data  [3]--> mask    [4:]--> generated data
            out_data_item = np.concatenate((np_data,np_data_mask,np_x_com), axis=1)


            # out_data_seq
            # TODO : check if the list can be saved
            out_data_seq_list.append(out_data_item)

        # transfer list to numpy array
        N, C, L, K, M = out_data_seq_list[0].shape
        out_data_seq_np = np.zeros((0, C, L, K, M))
        for item in out_data_seq_list:
            out_data_seq_np = np.concatenate((out_data_seq_np, item), axis=0)
        # shape [N,7,len_data, num_keypoints, num_person]
        np.save('{}/inference_data.npy'.format(self.arg.work_dir), out_data_seq_np)





            # save the data, x_com, data_mask as numpy data

            # with torch.no_grad():
            #     output = self.model(data, data_mask)
            # result_frag.append(output.data.cpu().numpy())

        #     # get loss
        #     if evaluation:
        #         loss = self.loss(output, label)
        #         loss_value.append(loss.item())
        #         label_frag.append(label.data.cpu().numpy())
        #
        # self.result = np.concatenate(result_frag)
        # if evaluation:
        #     self.label = np.concatenate(label_frag)
        #     self.epoch_info['mean_loss']= np.mean(loss_value)
        #     self.show_epoch_info()
        #
        #     # show top-k accuracy
        #     for k in self.arg.show_topk:
        #         self.show_topk(k)

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
        parser.add_argument('--recon_loss_weight', type=int, default=1, help='lambda for reconstuction loss')
        # endregion yapf: enable

        return parser
