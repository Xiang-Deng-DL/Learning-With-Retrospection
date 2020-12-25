#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
    
class KL_ays(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self):
        super(KL_ays, self).__init__()
        
    def forward(self, y_s, p_t):
        p_s = F.log_softmax(y_s, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) / p_s.shape[0]
        return loss
    