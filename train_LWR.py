from __future__ import print_function

import os
import argparse
import socket
import time
import numpy as np
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter, refinelogits, get_traintarget
from helper.loops import train_vanilla as train1, train_lwr as train2, validate

from zoo import Softmax_T, KL_ays, KL

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=782, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet56',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_16_4', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', 'ResNet50', 'ResNet18'])
    parser.add_argument('--arch', type=str, default='resnet', choices=['resnet', 'vgg', 'shuffle', 'mobile', 'wrn'], help='architecture')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    
    parser.add_argument('-r', '--gamma', type=float, default=0.1, help='weight for classification')
    
    parser.add_argument('-e', '--eta', type=int, default=5, help='k in the paper')

    parser.add_argument('--kd_T', type=float, default=10, help='\tau in the paper')

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')    
    

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.05

    # set the path according to the environment
    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'


    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()
    print('method: %s' %'Ours')
    print('model: %s' %opt.model)
    print('architecture: %s' %opt.arch)
    print('weight_decay %f' %opt.weight_decay)
    print('batch_size: %d' %opt.batch_size)
    print('r %f' %opt.gamma)
    #print('a %f' %opt.alpha)
    print('e %f' %opt.eta)
    print('KD_T %f' %opt.kd_T)
    

    # dataloader
    if opt.dataset == 'cifar100':                
        n_cls = 100


    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()





    criterion_cls = nn.CrossEntropyLoss()    
    
    criterion_soft = Softmax_T(opt.kd_T)
    
    criterion_kl = KL(opt.kd_T)
    
    

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_soft)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kl)     # other knowledge distillation loss



    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # routine
    #targets=get_traintarget()
    
    train_logits = torch.zeros((50000,100))
    
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=False, 
                                                            is_shuffle=True, is_soft=True, train_softlabels=train_logits)
    
    for epoch in range(1, opt.epochs + 1):
        
        


        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if epoch<=opt.eta:
            train_acc, train_loss, train_logits= train1(epoch, train_loader, model, criterion, optimizer, opt, train_logits)
        else:
            train_acc, train_loss, train_logits= train2(epoch, train_loader, model, criterion_list, optimizer, opt, train_logits)
         
        
        train_logits = train_logits.detach().cpu()
            
        if epoch>=opt.eta and epoch%opt.eta==0:
            print('label refinery')
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers, is_instance=False, is_shuffle=True, is_soft=True, train_softlabels=train_logits)
        
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)



    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)
  


if __name__ == '__main__':
    main()
