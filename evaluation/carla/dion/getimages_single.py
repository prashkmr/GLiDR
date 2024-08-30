from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils512 import *
import numpy as np
import os
import open3d as o3d
from open3d import open3d
from tqdm import trange



parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',       type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--folder',     type=str,   default='',              help='Location of the orignal LiDAR')
parser.add_argument('--debug', action='store_true')
# args = parser.parse_args(args=['atlas_baseline=0, autoencoder=1,panos_baseline=0'])





# Encoder must be trained with all types of frames,dynmaic, static all

npy = [0]

args = parser.parse_args()


original = np.load(args.data)[:,:,5:45,::2][:]
print(original.shape)

if not os.path.exists('samplesVid/'+str(args.folder) ):
	os.makedirs('samplesVid/'+str(args.folder) )


i=0
for frame_num in trange(  original.shape[0]):
	if(i<2048):
		i+=1
		frame = from_polar(torch.Tensor(original[frame_num:frame_num+1]).cuda())
		frame = frame.detach().cpu().numpy()
		# frame=from_polar(torch.Tensor(original[frame_num:frame_num+1,:,:,:]).cuda()).detach().cpu()
		plt.figure()
		plt.xlim([-0.15, 0.15])
		plt.ylim([-0.20, 0.20])
		# plt.xlim(-0.09,0.09)
		# plt.ylim(-0.20,0.20)
		# if(frame[:,2] >0.016):
		plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
		plt.savefig('samplesVid/'+str(args.folder)+'/'+str(frame_num)+'.jpg') 
	else:
		break
                                                              
