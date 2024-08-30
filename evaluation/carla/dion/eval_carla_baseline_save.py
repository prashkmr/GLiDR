#This is the code where we evaluate our method/model on CARLA on EMD/Chamfer / Result is 210 ans 1.
from __future__ import print_function

import argparse
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
import torch
import sys
from torchsummary import summary
import numpy as np
import os
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
# from data import ModelNet40, load
from model import *
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
from utils512 import *


parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,           help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
# parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=0,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--ae_weight',          type=str,   default='',             help='Location of the weights')
parser.add_argument('--data',               type=str,   default='',             help='Loction of the dataset')
parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Model to use, [pointnet, dgcnn]')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40'])
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
parser.add_argument('--num_points', type=int, default=outer,
                    help='num of points to use')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--emb_dims', type=int, default=outer, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--model_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--dim', type=int, default=8,
                    help='dim to reduce last data dimension')
parser.add_argument('--beam', type=int, default=8,
                    help='dim to reduce last data dimension')
parser.add_argument('--debug', action='store_true')


'''
Expect two arguments: 
    1) path_to_model_folder
    2) epoch of model you wish to load
    3) metric to evaluate on 
e.g. python eval.py runs/test_baseline 149 emd
'''

#---------------------------------------------------------------
#Helper Function and classes
class Pairdata(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, lidar):
        super(Pairdata, self).__init__()
        
        self.lidar = lidar

    def __len__(self):
        return self.lidar.shape[0]

    def __getitem__(self, index):
        
        return index, self.lidar[index]

#---------------------------------------------------------------
args = parser.parse_args()

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





# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200
# out_dir = os.path.join(sys.argv[1], 'final_samples')
# maybe_create_dir(out_dir)
save_test_dataset = False

fast = True

# fetch metric
# # if 'emd' in sys.argv[3]: 
# #     loss = EMD
# # elif 'chamfer' in sys.argv[3]:
# #     loss = get_chamfer_dist
# else:
#     raise ValueError("{} is not a valid metric for point cloud eval. Either \'emd\' or \'chamfer\'"\
#             .format(sys.argv[2]))

# loss1 = EMD
# loss_fn1 = loss1()
loss = get_chamfer_dist
# size = 10 if 'emd' in sys.argv[3] else 5
outer = 512

npydata = [8,9,10,11,12,13,14,15]
# npydata = [0]
orig = []
pred = []
total = 0
totalhd = 0

with torch.no_grad():
  
  for i in npydata:
    ii=0
    # 1) load trained model
    # model = load_model_from_file(sys.argv[1], epoch=int(sys.argv[2]), model='gen')[0]
    # model = VAE(args).cuda()
    # model = VAE(args).cuda()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = DGCNN(args).to(device)
    
    # model = nn.DataParallel(model)
    # summary(model, (3,8192))
    # exit(0)

    model = model.cuda()
    # network=torch.load(args.ae_weight)
    # print(network.keys())
    # model.load_state_dict(network['state_dict'])
    # model.load_state_dict(network['gen_dict'])
    # model = nn.DataParallel(model)
    weight = torch.load(args.ae_weight)
    model.load_state_dict(weight['state_dict'])
    # opt.load_state_dict(weight['optimizer'])
    
    model.eval() 
    # summary(model, (3,8192))
    # exit(0)

    # lidar_static    = np.load(args.data + "static/{}.npy".format(str(i)))[:,:,::int(64/args.beam),::args.dim].astype('float32')
    lidar_dynamic   = np.load(args.data + "dynamic/{}.npy".format(str(i)))[:,:,::int(64/args.beam),::args.dim].astype('float32')

    # lidar_dynamic    = (np.load(args.data + "dynamic/{}.npy".format(str(i)))[:,:,5:45,::args.dim].astype('float32'))
    # # lidar_dynamic   = (np.load(args.data + "dynamic/{}.npy".format(str(i)))[:,:,::int(64/args.beam),::args.dim].astype('float32'))
    lidar_static = lidar_dynamic
    out = np.ndarray(shape=(lidar_dynamic.shape[0],3,args.beam, int(outer/args.dim)))
    print(lidar_static.shape)  #correct

    
    lidar_dynamic = lidar_dynamic.reshape(-1, 3, args.beam *  int(outer/args.dim))
    
    print(lidar_dynamic.shape) #correct
    
    
    test_loader    = Attention_loader_dytost(lidar_dynamic, lidar_static)
    
    loader = (torch.utils.data.DataLoader(test_loader, batch_size=args.batch_size,
                        shuffle=False, num_workers=4, drop_last=False)) #False))

    loss_fn = loss()
    # process_input = (lambda x : x) if model.args.no_polar else to_polar
    process_input = from_polar if args.no_polar else lambda x : x
    
    # noisy reconstruction
    losses = []
   
    
    for batch in loader:
        lidar_d = batch[1].cuda()
        lidar_s = batch[2].cuda()


        # print(lidar.shape)
        # print(i* args.batch_size)
        
        recon = model( lidar_d )
        # exit(0)
        # print(recon.shape)
        # print("started")
        # print(recon.shape)
        recon = recon.reshape(-1,3, args.beam, int(outer/args.dim))
        # print(recon.shape, lidar_static.shape)
        
        # losses += [loss_fn(recon , lidar_static)]
        # losses1 += [directed_hausdorff(recon.reshape(-1,3,16 * int(outer/args.dim), (lidar_static).reshape(-1,3,16 * int(outer/args.dim))))]
        
        if ((ii+1) * args.batch_size >= lidar_dynamic.shape[0]) :
            end = lidar_dynamic.shape[0]
        else:
            end = (ii+1) * args.batch_size
        print(end)
        out[ii*args.batch_size:end]   =    recon.detach().cpu().numpy().reshape(-1, 3, args.beam, int(outer/args.dim))
        
        ii+=1# print(total/len(npydata))

    np.save(str(i)+'.npy', out)
    print('Saved ',i)
    # losses = torch.stack(losses).mean().item()
    # print('Chamfer Loss for {}: {:.4f}'.format(i, losses))
    # total += losses
    #--------------------------------------------------------------------


    # totalhd += losses1
    # print('EMD Loss for {}: {:.4f}'.format(i, losses1))

    # del recon


# print(total/len(npydata))

