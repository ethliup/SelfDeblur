import torch
import os
import numpy as np
from torch.optim import lr_scheduler

class ModelBase():
    def save_network(self, network, network_label, epoch_label, save_dir, on_gpu=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)

        if on_gpu:
            network.cuda()

    def load_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))
        print('load network from ', save_path)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def eval(self):
        with torch.no_grad():
            return self.forward()

    def build_lr_scheduler(self):
        self.lr_schedulers = []
        for name in self.optimizer_names:
            if isinstance(name, str):
                optimizer = getattr(self, 'optimizer_' + name)
                self.lr_schedulers.append(lr_scheduler.StepLR(optimizer, step_size=self.opts.lr_step, gamma=0.5))

    def update_lr(self):
        for scheduler in self.lr_schedulers:
            scheduler.step()

        for name in self.optimizer_names:
            if isinstance(name, str):
                optimizer = getattr(self, 'optimizer_' + name)
                for param_group in optimizer.param_groups:
                    print('optimizer_'+name+'_lr', param_group['lr'])
