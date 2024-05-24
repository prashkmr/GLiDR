#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python main.py --exp_name=carla_dytost_29oct --model=dgcnn --num_points=1024 --k=20
"""

@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM

"""


from __future__ import print_function

import argparse
import os

import numpy as np
# import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from model import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
# from tqdm import trange
from util import IOStream, cal_loss
from torchsummary import summary
from data import *
from topologylayer.nn import AlphaLayer, BarcodePolyFeature
from topologylayer.functional.utils_dionysus import *
from topologylayer.functional.rips_dionysus import Diagramlayer as DiagramlayerRips
from topologylayer.functional.levelset_dionysus import Diagramlayer as DiagramlayerToplevel

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')





# def getTopoLoss(z):
#     print(z.shape)
#     layer = AlphaLayer(maxdim = 0)
#     f1 = BarcodePolyFeature(0,2,0)   #dim(homology dimension to work over), p, q   # first should be same as the alphalyer dimension

#     # latentlayer = AlphaLayer(maxdim=0)
#     # f2 = BarcodePolyFeature(0,2,0)
#     loss = 0model
#     for i in trange(z.shape[0]):
#         # print(z[i].squeeze().shape)
#         # exit(0)
#         loss += f1(layer(z[i].squeeze()))
#     loss /= z.shape[0]
#     return loss


# Setup for Topology
width, height = 4,4
axis_x = np.arange(0, width)
axis_y = np.arange(0, height)
grid_axes = np.array(np.meshgrid(axis_x, axis_y))    # 4*4 mesh  
grid_axes = np.transpose(grid_axes, (1, 2, 0))
from scipy.spatial import Delaunay
tri = Delaunay(grid_axes.reshape([-1, 2]))
faces = tri.simplices.copy()
F = DiagramlayerToplevel().init_filtration(faces)
diagramlayerToplevel = DiagramlayerToplevel.apply





def train_kitti(args, io):
    # train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
    #                          batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    # trainList = [args.data + '/kitti/lidar/k0.npy', 
    #              args.data + '/kitti/lidar/k1.npy', 
    #              args.data + '/kitti/lidar/k2.npy', 
    #              args.data + '/kitti/lidar/k3.npy']
    npy = [0,1,2,4,5,6,7,9,10]
    skip = [6,2,6,1,4,2,2,3,2]
    # skip = [3,1,3,1,2,1,1,2,1]
    headstart = [3,1,3,0,2,1,1,2,1]
    # npy = [3]
    # skip = [1]
    # headstart = [0]
    trainloaders = load_kitti_DyToSt(npy, skip, headstart, args)
    npy = [3]
    skip = [1]
    headstart = [0]
    testloaders = load_kitti_DyToSt(npy, skip, headstart, args)
    
    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_Topo(args).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))
    print('Summary')
    # model = nn.DataParallel(model)
    summary(model, (3,args.beam * int(1024/args.dim)))
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # weight = torch.load("/home/prashant/scratch/code/dgcnn/pytorch/checkpoints/carla_dytost_29oct/models/model_49.t7")
    # model.load_state_dict(weight['state_dict'])
    # opt.load_state_dict(weight['optimizer'])
    # print("Weights and optim loaded")
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1)
    # loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1)
    # criterion = cal_loss
    criterion = loss_fn

    best_test_loss = 100000000.0
    k = 0

    if args.reload != '':
        checkpoint = torch.load(args.reload)
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
    k = 0
    for epoch in trange(0, 100):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        k = (k+1)%2
    
        for data in trainloaders[k]:
            
            # k+=1
            dynamic_data = data[1].to(device)
            static_data = data[2].to(device)
            batch_size = dynamic_data.size()[0]
            opt.zero_grad()
            # print('going')
            logits, hidden1, hidden2, _ = model(dynamic_data)


            top_loss_out     = top_batch_cost(logits.cpu(), diagramlayerToplevel, F)
            top_loss_hidden1 = top_batch_cost(hidden1.cpu(), diagramlayerToplevel, F)
            top_loss_hidden2 = top_batch_cost(hidden2.cpu(), diagramlayerToplevel, F)
            loss = criterion(logits, static_data)
            loss += top_loss_out + top_loss_hidden1 + top_loss_hidden2
            loss.backward()
            opt.step()
            # preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            # train_true.append(static_data.cpu().numpy())
            # train_pred.append(preds.detach().cpu().numpy())
    # train_true = np.concatenate(train_true)
    # train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f' % (epoch, train_loss*1.0/count)
                                                                                #  metrics.accuracy_score(
                                                                                #      train_true, train_pred),
                                                                                #  metrics.balanced_accuracy_score(
                                                                                #      train_true, train_pred))
        io.cprint(outstr)
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'optimizer': opt.state_dict()}
        torch.save(state, f'checkpoints/{args.exp_name}/models/model_{epoch}.t7')
        # continue
        




        ####################
        # Test
        ####################
        if epoch % 2 == 0 :
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
        
            for data in testloaders[0]:
                dynamic_data = data[1].to(device)
                static_data = data[2].to(device)
                batch_size = dynamic_data.size()[0]
                logits,_,_,_ = model(dynamic_data)
                loss = criterion(logits, static_data)
                # preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                # test_true.append(static_data.cpu().numpy())
                # test_pred.append(preds.detach().cpu().numpy())
            # test_true = np.concatenate(test_true)
            # test_pred = np.concatenate(test_pred)
            outstr = 'Test %d, loss: %.6f' % (epoch, test_loss*1.0/count)
            test_loss = test_loss*1.0/count
            io.cprint(outstr)

            if test_loss < best_test_loss:
                print('Latest Epoch', epoch)
                best_test_loss = test_loss
                # state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),'optimizer': opt.state_dict()}
                # torch.save(state, f'checkpoints/{args.exp_name}/models/model_{epoch}.t7')
                #print('Saved')
        # torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/model_{epoch}.t7')







if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data', type=str, default='', metavar='N',
                        help='Location of dataset')                 
    parser.add_argument('--mode', type=str, default='carla', required =True,
                        help='Locati512on of dataset')         
    parser.add_argument('--reload', type=str, default='',
                        help='Location of dataset')         
    parser.add_argument('--dim', type=int, default= 4,
                        help='Location of dataset')         
    parser.add_argument('--beam', type=int, default=16,
                        help='Location of dataset')                                                                        
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if args.mode == 'kitti':
        train_kitti(args, io)
    