#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
# import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


# def load_data(partition):
#     download()
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_DIR = os.path.join(BASE_DIR, 'data')
#     all_data = []
#     all_label = []
#     for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
#         f = h5py.File(h5_name)
#         data = f['data'][:].astype('float32')
#         label = f['label'][:].astype('int64')
#         f.close()
#         all_data.append(data)
#         all_label.append(label)
#     all_data = np.concatenate(all_data, axis=0)
#     all_label = np.concatenate(all_label, axis=0)
#     return all_data, all_label

def from_polar_np(velo):
    angles = np.linspace(0, np.pi * 2, velo.shape[-1])
    dist, z = velo[:, 0], velo[:, 1]
    x = np.cos(angles) * dist
    y = np.sin(angles) * dist
    out = np.stack([x,y,z], axis=1)
    return out.astype('float32')

class Attention_loader(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, lidar):
        super(Attention_loader, self).__init__()

        self.lidar = lidar

    def __len__(self):
        return self.lidar.shape[0]

    def __getitem__(self, index):
        
        return index, self.lidar[index]




class Attention_loader_dytost(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dynamic, static):
        super(Attention_loader_dytost, self).__init__()

        self.dynamic = dynamic
        self.static = static
    

    def __len__(self):
        return self.dynamic.shape[0]

    def __getitem__(self, index):
        
        return index, self.dynamic[index], self.static[index]

# def load(npyList):
#     retList=[]
#     for i in npyList:
#         print(i)
#         lidar_train = np.load(os.path.join(i))
#         lidar_train = lidar_train[:,:,5:45,::2].astype('float32') # 2048 [, 2, 40, 256]     this has 2048    samples
#         lidar_train = from_polar_np(lidar_train).reshape(-1, 3, 1024) # 2048, 3, 40, 256 = > 20480, 3, 1024  
        
#         data_train = Attention_loader(lidar_train)

#         train_loader  = torch.utils.data.DataLoader(data_train, batch_size=48,
#                         shuffle=True, num_workers=4, drop_last=True)
#         #del data_train
#         retList.append(train_loader)
#     return retList

def load_dytost(dynamic, static,args):
    retList=[]
    for i, j in zip(dynamic, static):
        dynamic_lidar = np.load(args.data + 'd' + str(i) + '.npy')
        static_lidar = np.load(args.data + 's' + str(i) + '.npy')
        # print(dynamic_lidar.shape, static_lidar.shape)
        

        dynamic_lidar = from_polar_np(dynamic_lidar[:,:,::int(64/args.beam),::args.dim]).astype('float32') # 2048 [, 2, 40, 256]     this has 2048    samples
        # print(dynamic_lidar.shape)
        dynamic_lidar = (dynamic_lidar).reshape(-1, 3, args.beam * int(512/args.dim))      # 2048, 3, 40, 256 = > 20480, 3, 1024
        print(dynamic_lidar.shape)

        static_lidar = from_polar_np(static_lidar[:,:,::int(64/args.beam),::args.dim]).astype('float32')   # 2048 [, 2, 40, 256]     this has 2048    samples
        # print(static_lidar.shape)
        static_lidar = (static_lidar).reshape(-1, 3, args.beam * int(512/args.dim))        # 2048, 3, 40, 256 = > 20480, 3, 1024  
        print(static_lidar.shape)
        
        data_train = Attention_loader_dytost(dynamic_lidar, static_lidar)
        train_loader  = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size,
                        shuffle=True, num_workers=4, drop_last=True)
        #del data_train
        retList.append(train_loader)
    return retList

# Attention_loader_dytost


def load_kitti_DyToSt(npy, skip ,headstart, args):
    print('Coming to load')
    # npy = [str(i) for i in range(10)]
    # npy.remove('8')
    # skip = [6,2,6,2,1,4,2,2,3,2]
    # headstart = [3,1,3,1,0,2,1,1,2,1]
    st1 = []
    dy1 = []
    st2 = []
    dy2 = []
    commonSt = []
    commonDy = []

    for i in trange(len(npy)):
        # st = np.load(args.data + 'static/' + str(npy[i])+ '.npy')[:,:,5:45,::args.dim].astype('float32').reshape(-1, 3, args.beam * int(1024/args.dim))
        # dy = np.load(args.data + 'dynamic/' + str(npy[i])+ '.npy')[:,:,5:45,::args.dim].astype('float32').reshape(-1, 3, args.beam * int(1024/args.dim))
        st = np.load(args.data + 'static/' + str(npy[i])+ '.npy')[:,:,::int(64/args.beam),::args.dim].astype('float32').reshape(-1, 3, args.beam * int(1024/args.dim))
        dy = np.load(args.data + 'dynamic/' + str(npy[i])+ '.npy')[:,:,::int(64/args.beam),::args.dim].astype('float32').reshape(-1, 3, args.beam * int(1024/args.dim))

        # mask= np.load()
        print(st.shape, dy.shape)   #correct
        
        st1.append(st[::skip[i]])
        dy1.append(dy[::skip[i]])
        
        
        st2.append(st[headstart[i]:][::skip[i]])
        dy2.append(dy[headstart[i]:][::skip[i]])
        

    st1 = np.concatenate(st1, axis=0)
    dy1 = np.concatenate(dy1, axis=0)
    

    st2 = np.concatenate(st2, axis=0)
    dy2 = np.concatenate(dy2, axis=0)
        # dy = np.load(args.data + 'dynamic/' + str(npy[i])+ '.npy')[:,:,::int(64/args.beam),::args.dim].astype('float32').reshape(-1, 3, args.beam * int(1024/args.dim)
    print('Final Shape')
    print(st1.shape,dy1.shape,st2.shape,dy2.shape)

    data1 = Attention_loader_dytost(dy1, st1)
    data2 = Attention_loader_dytost(dy2, st2)
    loader1  = torch.utils.data.DataLoader(data1, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    loader2  = torch.utils.data.DataLoader(data2, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    return [loader1, loader2]
    # return [loader1]





# def load_carla_DyToSt(npy, skip ,headstart, args):
#     # npy = [str(i) for i in range(10)]
#     # npy.remove('8')
#     # skip = [6,2,6,2,1,4,2,2,3,2]
#     # headstart = [3,1,3,1,0,2,1,1,2,1]
#     st1 = []
#     dy1 = []
#     st2 = []
#     dy2 = []
#     commonSt = []
#     commonDy = []

#     for i in trange(len(npy)):
#         st = np.load(args.data + 'static/' + str(npy[i])+ '.npy')[:,:,::int(64/args.beam),::args.dim].astype('float32').reshape(-1, 3, args.beam * int(1024/args.dim))
#         dy = np.load(args.data + 'dynamic/' + str(npy[i])+ '.npy')[:,:,::int(64/args.beam),::args.dim].astype('float32').reshape(-1, 3, args.beam * int(1024/args.dim))
#         # print(st.shape, dy.shape)   #correct
        
#         st1.append(st[::skip[i]])
#         dy1.append(dy[::skip[i]])
        
#         st2.append(st[headstart[i]:][::skip[i]])
#         dy2.append(dy[headstart[i]:][::skip[i]])

#     st1 = np.concatenate(st1, axis=0)
#     dy1 = np.concatenate(dy1, axis=0)
#     st2 = np.concatenate(st2, axis=0)
#     dy2 = np.concatenate(dy2, axis=0)
#     print(st1.shape, dy1.shape, st2.shape, dy2.shape)


    data1 = Attention_loader_dytost(dy1, st1)
    data2 = Attention_loader_dytost(dy2, st2)
    loader1  = torch.utils.data.DataLoader(data1, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    loader2  = torch.utils.data.DataLoader(data2, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    return [loader1, loader2]







            
            

        
            






def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    npyList = ['data_custom/s/tests0.npy', 'data_custom/s/tests1.npy', 'data_custom/s/tests2.npy', 'data_custom/s/tests3.npy', 'data_custom/s/tests4.npy', 'data_custom/s/tests5.npy', 'data_custom/s/tests6.npy', 'data_custom/s/tests7.npy', 'data_custom/d/testd0.npy', 'data_custom/d/testd1.npy', 'data_custom/d/testd2.npy', 'data_custom/d/testd3.npy', 'data_custom/d/testd4.npy', 'data_custom/d/testd5.npy', 'data_custom/d/testd6.npy', 'data_custom/d/testd7.npy']
    npyList = load(npyList)
    print(npyList)

    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')
    for dataloader in npyList:
        for data in dataloader:
            print(np.array(data).shape)
            break
        # print(data.shape)
        # print(label.shape)
        # break
