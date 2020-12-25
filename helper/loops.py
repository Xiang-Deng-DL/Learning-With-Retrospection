from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
from .util import AverageMeter, accuracy
from dataset.cifar100 import get_cifar100_dataloaders

from torch.autograd import Variable


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt, train_logits):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    if torch.cuda.is_available():
        train_logits=train_logits.cuda()
    
    for idx, (input, target, softlabel, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        train_logits[index] = output
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx>0 and idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, train_logits



def train_lwr(epoch, train_loader, model, criterion_list, optimizer, opt, train_logits):
    """lwr"""
    # set modules as train()
    model.train()
        
    # set teacher as eval()

    if torch.cuda.is_available():
        train_logits=train_logits.cuda()

    criterion_cls = criterion_list[0]
    critetion_soft = criterion_list[1]
    criterion_kl = criterion_list[2]


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    for idx, data in enumerate(train_loader):
        input, target, logits, index = data

        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            logits = logits.cuda()

        # ===================forward=====================
        
        soft_label = critetion_soft(logits)
        
        preact = False

        logit_s = model(input, is_feat=False, preact=preact)
        
        #if epoch>=opt.eta and epoch%opt.eta==0:
        train_logits[index] = logit_s
        


        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        
        loss_kl = criterion_kl(logit_s, soft_label)

        # other kd beyond KL divergence


        #loss = opt.gamma * loss_cls + opt.alpha * loss_kl
        
        if epoch<=opt.eta:
            loss = loss_cls 
        else:
            num_5 = int(epoch/opt.eta)
            cure = num_5*opt.eta
            
            loss = (opt.gamma+(1-cure/240)*(1-opt.gamma)) * loss_cls + cure/240*(1-opt.gamma)* loss_kl
            #loss = (1-opt.gamma-cure/240*0.8)* loss_cls + (opt.gamma+cure/240*0.8) * loss_kl
            #loss = loss_cls + opt.gamma* loss_kl

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx>0 and idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, train_logits




def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx>0 and idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg



def ECE(model, criterion):
    """validation"""


    # switch to evaluate mode#loss.item()
    _, val_loader = get_cifar100_dataloaders(batch_size=200, num_workers=8, is_instance=False, 
                                                        is_shuffle=False, is_soft=False)
 

    
    targets = []
    
    logits = []
    
    model.eval()

    with torch.no_grad():
        
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                
                input = input.cuda()
                target = target.cuda()
           
            targets+=[target]

            # compute output
            output = model(input)
            
            logits+=[output]
        
        logits = torch.cat(logits, dim=0)
        
        targets = torch.stack(targets).view(-1)
        
        err, bin_cof, bin_acc, probin = criterion(logits, targets)
        
        
    return err, bin_cof, bin_acc, probin


