
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import numpy as np

# Basic GCN Layer 
# Reference: https://towardsdatascience.com/program-a-simple-graph-net-in-pytorch-e00b500a642d
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)

# GC block module with stacked GCN and BatchNorm layers
# BN is for standardising input to layer for each mini-batch. Helps to stablise the learning process
class GCBlock(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, is_resi=True):
        super(GCBlock, self).__init__()
        self.in_features = in_features
        self.out_features = in_features
        # Set residual links to deal with gradient outflow
        self.is_resi = is_resi

        self.gc1 = GCNLayer(in_features, in_features)
        self.bn1 = nn.BatchNorm1d(55 * in_features)

        self.gc2 = GCNLayer(in_features, in_features)
        self.bn2 = nn.BatchNorm1d(55 * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        # Use residual links
        if self.is_resi:
            return y + x
        else:
            return y

# Multi GCBlock model with FC final layer for classification
class GCNMultiBlock(nn.Module):
    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage=1, is_resi=True):
        super(GCNMultiBlock, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GCNLayer(input_feature, hidden_feature)
        self.bn1 = nn.BatchNorm1d(55 * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GCBlock(hidden_feature, p_dropout=p_dropout, is_resi=is_resi))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.fc_out = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        out = torch.mean(y, dim=1)
        out = self.fc_out(out)

        return out