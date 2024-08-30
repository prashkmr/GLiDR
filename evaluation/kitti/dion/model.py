#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch, torch.nn as nn, numpy as np

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def circlefn(i, j, n):
    r = np.sqrt((i - n/2.)**2 + (j - n/2.)**2)
    return np.exp(-(r - n/3.)**2/(n*2))


def gen_circle(n):
    beta = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            beta[i,j] = circlefn(i,j,n)
    return torch.from_numpy(beta).cuda()

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(64)
        self.bn10 = nn.BatchNorm1d(3)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(args.emb_dims, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv1d(64, 3, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.Tanh())
        # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # print(x.shape)                          #1, 3, 8192
        batch_size = x.size(0)
        
        x = get_graph_feature(x, k=self.k)
        # print(x.shape)                          #1, 6, 8192, 20]
        x = self.conv1(x)
        # print(x.shape)                          #1, 64, 8192, 20
        x1 = x.max(dim=-1, keepdim=False)[0]
        # print('x1 ' ,x1.shape)                         #1, 64, 8192
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        # print(x.shape)
        x2 = x.max(dim=-1, keepdim=False)[0]
        # print('x2 ', x2.shape)                         #1, 128, 8192
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        # print('x3 ', x3.shape)                         #1, 256, 8192
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        # print('x4 ', x4.shape)                         

        x = torch.cat((x1, x2, x3, x4), dim=1)  
        # print('concat ',x.shape)                            #1, 512, 8192
        # exit(0)
        
        x = self.conv5(x) # 32, 1024, 1024
        # print('C5' ,x.shape)
        x = self.conv6(x)
        # print('C6' ,x.shape)
        x = self.conv7(x)
        # print('C7' ,x.shape)
        x = self.conv8(x)
        # print('C8' ,x.shape)
        x = self.conv9(x)
        # print('C9' ,x.shape)
        x = self.conv10(x)
        # print('C10' ,x.shape)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        return x




class DGCNN_Topo(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_Topo, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(64)
        self.bn10 = nn.BatchNorm1d(3)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(args.emb_dims, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv9 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv1d(64, 3, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.Tanh())
        # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        print('x1' ,x1.shape)
        
        x_hidden2 = get_graph_feature(x1, k=self.k)
        x = self.conv2(x_hidden2)
        x2 = x.max(dim=-1, keepdim=False)[0]
        print('x2' ,x2.shape)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        print('x3' ,x2.shape)

        
        x_hidden1 = get_graph_feature(x3, k=self.k)   #[batch, x,y]
        x = self.conv4(x_hidden1)
        x4 = x.max(dim=-1, keepdim=False)[0]
        print('x4' ,x2.shape)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        print('x' ,x.shape)

        x = self.conv5(x) # 32, 1024, 1024
        x0 = x.max(dim=-1, keepdim=False)[0]     # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feat = x0                   # (batch_size, num_points) -> (batch_size, 1, emb_dims)


        x = self.conv6(x)
        print('C6' ,x.shape)
        
        x = self.conv7(x)
        print('C7' ,x.shape)
        
        x = self.conv8(x)
        print('C8' ,x.shape)
        
        x = self.conv9(x)
        print('C9' ,x.shape)
        
        x = self.conv10(x)
        print('C10' ,x.shape)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        return x, x_hidden1, x_hidden2, feat



# for input of shape [3,8192]

# x1  torch.Size([-1, 64, 8192])
# x2  torch.Size([-1, 64, 8192])
# x3  torch.Size([-1, 128, 8192])
# x4  torch.Size([-1, 256, 8192])
# concat  torch.Size([-1, 512, 8192])
# C5 torch.Size([-1, 1024, 8192])
# C6 torch.Size([-1, 512, 8192])
# C7 torch.Size([-1, 256, 8192])
# C8 torch.Size([-1, 128, 8192])
# C9 torch.Size([-1, 64, 8192])
# C10 torch.Size([-1, 3, 8192])




# class DGCNN_pruned(nn.Module):
#     def __init__(self, args, output_channels=40):
#         super(DGCNN_pruned, self).__init__()
#         self.args = args
#         self.k = 28
        
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(1024)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.bn8 = nn.BatchNorm1d(128)
#         self.bn9 = nn.BatchNorm1d(64)
#         self.bn10 = nn.BatchNorm1d(3)

#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv6 = nn.Sequential(nn.Conv1d(args.emb_dims, 512, kernel_size=1, bias=False),
#                                    self.bn6,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
#                                    self.bn7,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
#                                    self.bn8,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv9 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
#                                    self.bn9,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv10 = nn.Sequential(nn.Conv1d(64, 3, kernel_size=1, bias=False),
#                                    self.bn10,
#                                    nn.Tanh())
#         # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
#         # self.bn6 = nn.BatchNorm1d(512)
#         # self.dp1 = nn.Dropout(p=args.dropout)
#         # self.linear2 = nn.Linear(512, 256)
#         # self.bn7 = nn.BatchNorm1d(256)
#         # self.dp2 = nn.Dropout(p=args.dropout)
#         # self.linear3 = nn.Linear(256, output_channels)

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k)
#         x = self.conv1(x)
#         x1 = x.max(dim=-1, keepdim=False)[0]
        
#         x = get_graph_feature(x1, k=self.k)
#         x = self.conv2(x1)
#         x2 = x.max(dim=-1, keepdim=False)[0]
        
#         x = get_graph_feature(x2, k=self.k)
#         x = self.conv3(x2)
#         x3 = x.max(dim=-1, keepdim=False)[0]
        
#         x = get_graph_feature(x3, k=self.k)
#         x = self.conv4(x)
#         x4 = x.max(dim=-1, keepdim=False)[0]
        
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         # x = torch.cat((x1, x3), dim=1)
        
#         x = self.conv5(x) # 32, 1024, 1024
#         bottleneck = x
        
#         x = self.conv6(x)
        
#         x = self.conv7(x)
#         x = get_graph_feature(x, k=self.k)
    
#         x = self.conv8(x)
#         x = get_graph_feature(x, k=self.k)
#         x = self.conv9(x)
        
#         x = self.conv10(x)
#         # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         # x = torch.cat((x1, x2), 1)

#         # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#         # x = self.dp1(x)
#         # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#         # x = self.dp2(x)
#         # x = self.linear3(x)
#         return x, bottleneck





# class DGCNN_pruned(nn.Module):
#     def __init__(self, args, output_channels=40):
#         super(DGCNN_pruned, self).__init__()
#         self.args = args
#         self.k = 28
        
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm1d(1024)
#         self.bn6 = nn.BatchNorm1d(512)
#         self.bn7 = nn.BatchNorm1d(256)
#         self.bn8 = nn.BatchNorm1d(128)
#         self.bn9 = nn.BatchNorm1d(64)
#         self.bn10 = nn.BatchNorm1d(3)

#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
#                                    self.bn1,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                    self.bn2,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
#                                    self.bn3,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
#                                    self.bn4,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
#                                    self.bn5,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv6 = nn.Sequential(nn.Conv1d(args.emb_dims, 512, kernel_size=1, bias=False),
#                                    self.bn6,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv7 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
#                                    self.bn7,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv8 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
#                                    self.bn8,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv9 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
#                                    self.bn9,
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv10 = nn.Sequential(nn.Conv1d(64, 3, kernel_size=1, bias=False),
#                                    self.bn10,
#                                    nn.Tanh())
#         # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
#         # self.bn6 = nn.BatchNorm1d(512)
#         # self.dp1 = nn.Dropout(p=args.dropout)
#         # self.linear2 = nn.Linear(512, 256)
#         # self.bn7 = nn.BatchNorm1d(256)
#         # self.dp2 = nn.Dropout(p=args.dropout)
#         # self.linear3 = nn.Linear(256, output_channels)

#     def forward(self, x):
#         batch_size = x.size(0)
#         x = get_graph_feature(x, k=self.k)
#         x = self.conv1(x)
#         x1 = x.max(dim=-1, keepdim=False)[0]
        
#         # x = get_graph_feature(x1, k=self.k)
#         x = self.conv2(x1)
#         x2 = x.max(dim=-1, keepdim=False)[0]
        
#         x = get_graph_feature(x2, k=self.k)
#         x = self.conv3(x2)
#         x3 = x.max(dim=-1, keepdim=False)[0]
        
#         # x = get_graph_feature(x3, k=self.k)
#         x = self.conv4(x3)
#         x4 = x.max(dim=-1, keepdim=False)[0]
        
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         # x = torch.cat((x1, x3), dim=1)
        
#         x = self.conv5(x) # 32, 1024, 1024
#         bottleneck = x
        
#         x = self.conv6(x)
        
#         x = self.conv7(x)
#         x = get_graph_feature(x, k=self.k)
    
#         x = self.conv8(x)
#         x = get_graph_feature(x, k=self.k)
#         x = self.conv9(x)
        
#         x = self.conv10(x)
#         # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         # x = torch.cat((x1, x2), 1)

#         # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
#         # x = self.dp1(x)
#         # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
#         # x = self.dp2(x)
#         # x = self.linear3(x)
#         return x, bottleneck
class TopologicalSignatureDistance(nn.Module):
    """Topological signature."""

    def __init__(self, sort_selected=False, use_cycles=False,
                 match_edges=None):
        """Topological signature computation.
        Args:
            p: Order of norm used for distance computation
            use_cycles: Flag to indicate whether cycles should be used
                or not.
        """
        super().__init__()
        self.use_cycles = use_cycles

        self.match_edges = match_edges

        # if use_cycles:
        #     use_aleph = True
        # else:
        #     if not sort_selected and match_edges is None:
        #         use_aleph = True
        #     else:
        #         use_aleph = False

        # if use_aleph:
        #     print('Using aleph to compute signatures')
        ##self.signature_calculator = AlephPersistenHomologyCalculation(
        ##    compute_cycles=use_cycles, sort_selected=sort_selected)
        # else:
        # print('Using python to compute signatures')
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances):
        pairs_0, pairs_1 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0, pairs_1

    def _select_distances_from_pairs(self, distance_matrix, pairs):
        # Split 0th order and 1st order features (edges and cycles)
        pairs_0, pairs_1 = pairs
        selected_distances = distance_matrix[(pairs_0[:, 0], pairs_0[:, 1])]

        if self.use_cycles:
            edges_1 = distance_matrix[(pairs_1[:, 0], pairs_1[:, 1])]
            edges_2 = distance_matrix[(pairs_1[:, 2], pairs_1[:, 3])]
            edge_differences = edges_2 - edges_1

            selected_distances = torch.cat(
                (selected_distances, edge_differences))

        return selected_distances

    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1)

    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))

    @staticmethod
    def _get_nonzero_cycles(pairs):
        all_indices_equal = np.sum(pairs[:, [0]] == pairs[:, 1:], axis=-1) == 3
        return np.sum(np.logical_not(all_indices_equal))

    # pylint: disable=W0221
    def forward(self, distances1, distances2):
        """Return topological distance of two pairwise distance matrices.
        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2
        Returns:
            distance, dict(additional outputs)
        """
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs1[0], pairs2[0])
        }
        # Also count matched cycles if present
        if self.use_cycles:
            distance_components['metrics.matched_pairs_1D'] = \
                self._count_matching_pairs(pairs1[1], pairs2[1])
            nonzero_cycles_1 = self._get_nonzero_cycles(pairs1[1])
            nonzero_cycles_2 = self._get_nonzero_cycles(pairs2[1])
            distance_components['metrics.non_zero_cycles_1'] = nonzero_cycles_1
            distance_components['metrics.non_zero_cycles_2'] = nonzero_cycles_2

        if self.match_edges is None:
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            distance = self.sig_error(sig1, sig2)

        elif self.match_edges == 'symmetric':
            sig1 = self._select_distances_from_pairs(distances1, pairs1)
            sig2 = self._select_distances_from_pairs(distances2, pairs2)
            # Selected pairs of 1 on distances of 2 and vice versa
            sig1_2 = self._select_distances_from_pairs(distances2, pairs1)
            sig2_1 = self._select_distances_from_pairs(distances1, pairs2)

            distance1_2 = self.sig_error(sig1, sig1_2)
            distance2_1 = self.sig_error(sig2, sig2_1)

            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs1[0])
            pairs1 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs2 = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sig1_1 = self._select_distances_from_pairs(
                distances1, (pairs1, None))
            sig1_2 = self._select_distances_from_pairs(
                distances2, (pairs1, None))

            sig2_2 = self._select_distances_from_pairs(
                distances2, (pairs2, None))
            sig2_1 = self._select_distances_from_pairs(
                distances1, (pairs2, None))

            distance1_2 = self.sig_error(sig1_1, sig1_2)
            distance2_1 = self.sig_error(sig2_1, sig2_2)
            distance_components['metrics.distance1-2'] = distance1_2
            distance_components['metrics.distance2-1'] = distance2_1

            distance = distance1_2 + distance2_1

        return distance, distance_components