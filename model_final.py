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
        # adding activation function ReLu
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
        
        # GCN layer with batchnorm
        self.gcn_1 = GCNLayer(in_features, in_features)
        self.batchnorm1 = nn.BatchNorm1d(55 * in_features)

        # GCN layer with batchnorm
        self.gcn_2 = GCNLayer(in_features, in_features)
        self.batchnorm2 = nn.BatchNorm1d(55 * in_features)

        # Dropout and TanH activation function
        self.dropout = nn.Dropout(p_dropout)
        self.tanlayer = nn.Tanh()

    def forward(self, x):
        y = self.gcn_1(x)
        b, n, f = y.shape
        y = self.batchnorm1(y.view(b, -1)).view(b, n, f)
        y = self.tanlayer(y)
        y = self.dropout(y)

        y = self.gcn_2(y)
        b, n, f = y.shape
        y = self.batchnorm2(y.view(b, -1)).view(b, n, f)
        y = self.tanlayer(y)
        y = self.dropout(y)
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

        self.gcn_1 = GCNLayer(input_feature, hidden_feature)
        self.batchnorm1 = nn.BatchNorm1d(55 * hidden_feature)

        self.gcnblock = []
        # adding gcb blocks into list
        for i in range(num_stage):
            self.gcnblock.append(GCBlock(hidden_feature, p_dropout=p_dropout, is_resi=is_resi))
        
        self.gcnblock = nn.ModuleList(self.gcnblock)

        self.dropout = nn.Dropout(p_dropout)
        self.tanhlayer = nn.Tanh()

        self.fc_layer = nn.Linear(hidden_feature, num_class)

    def forward(self, x):
        y = self.gcn_1(x)
        b, n, f = y.shape
        y = self.batchnorm1(y.view(b, -1)).view(b, n, f)
        y = self.tanhlayer(y)
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.gcnblock[i](y)

        out = torch.mean(y, dim=1)
        # final FC layer for classification
        out = self.fc_layer(out)

        return out
