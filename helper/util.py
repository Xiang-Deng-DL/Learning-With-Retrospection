from __future__ import print_function

import torch
from dataset.cifar100 import get_cifar100_dataloaders
from torch import nn
from torch.nn import functional as F



def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    
    if opt.arch=='vgg':
        steps = int(epoch/20)
        if steps > 0:
            new_lr = opt.learning_rate * (0.5 ** steps)#0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
    elif opt.arch=='resnet' or opt.arch=='shuffle' or opt.arch=='mobile':
        if epoch > 100 and epoch <=150:
            new_lr = opt.learning_rate * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        elif epoch > 150:
            new_lr = opt.learning_rate * 0.1 * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
    elif opt.arch=='wrn':
        if epoch > 60 and epoch <=120:
            new_lr = opt.learning_rate * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        elif epoch > 120 and epoch <=180:
            new_lr = opt.learning_rate * 0.2 * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        elif epoch > 180:
            new_lr = opt.learning_rate * 0.2 * 0.2 * 0.2
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr 

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_traintarget():
    targets=torch.zeros(50000, dtype=torch.long)
    batch_size=200
    train_loader, val_loader, n = get_cifar100_dataloaders(batch_size=batch_size, num_workers=8, is_instance=True, is_soft=False, is_shuffle=False)
    for idx, (input, target, index) in enumerate(train_loader):
        targets[index] = target
    return targets