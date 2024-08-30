import warnings
warnings.filterwarnings('ignore', message='not allowed')
#This uses polar data and all the configuation is correct  no_polar =1
# python eval_carla_final_.py --data ~/scratch/data/kitti/polar/ --ae_weight 16/gen_990.pth --dim 8 --beam 16
import argparse
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
import torch
import sys
from torchsummary import summary
import numpy as np
import os
# import __init__
# from emd import EMD
from utils512 import * 
# from models256 import * 
# from model import PointNet, DGCNN
from pyemd import emd, emd_samples
# from models64 import *





parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=128,            help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=1,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=160,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--ae_weight',          type=str,   default='',             help='Location of the weights')
parser.add_argument('--data',               type=str,   default='',             help='Loction of the dataset')
parser.add_argument('--dim',                type=int,   default=8,             help='Location of the weights')
parser.add_argument('--beam',               type=int,   default=16,             help='Loction of the dataset')
parser.add_argument('--emb_dims', type=int,          default=1024, metavar='N', help='Dimension of embeddings')


parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


if args.beam == 64:
    from models64 import *
    print('Models 64 imported')
else:
    from models16 import *
    print('Models 16 imported')



def directed_hausdorff(point_cloud1:torch.Tensor, point_cloud2:torch.Tensor, reduce_mean=True):
    """

    :param point_cloud1: (B, 3, N)
    :param point_cloud2: (B, 3, M)
    :return: directed hausdorff distance, A -> B
    """
    n_pts1 = point_cloud1.shape[2]
    n_pts2 = point_cloud2.shape[2]

    pc1 = point_cloud1.unsqueeze(3)
    pc1 = pc1.repeat((1, 1, 1, n_pts2)) # (B, 3, N, M)
    pc2 = point_cloud2.unsqueeze(2)
    pc2 = pc2.repeat((1, 1, n_pts1, 1)) # (B, 3, N, M)

    l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1)) # (B, N, M)

    shortest_dist, _ = torch.min(l2_dist, dim=2)

    hausdorff_dist, _ = torch.max(shortest_dist, dim=1) # (B, )

    if reduce_mean:
        hausdorff_dist = torch.mean(hausdorff_dist)

    return hausdorff_dist


#--------------------------------------------------------
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

def calc_mmd_loss(x, y, alpha=1):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    B = x.shape[0]

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    K = torch.exp(-alpha * (rx.t() + rx - 2 * xx))
    L = torch.exp(-alpha * (ry.t() + ry - 2 * yy))
    P = torch.exp(-alpha * (rx.t() + ry - 2 * zz))

    beta = 1.0 / (B * (B - 1))
    gamma = 2.0 / (B * B)

    return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)





def calc_mmd_loss1(x, y, alpha=1):
    # print(x.size())
    x = x.view(x.size(0), x.size(1) * x.size(2) )
    y = y.view(y.size(0), y.size(1) * y.size(2) )
    B = x.size(0)

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*zz))

    beta = (1./(B*(B-1)))
    gamma = (2./(B*B)) 

    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)








class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)).log_softmax(-1), q.view(-1, q.size(-1)).log_softmax(-1)
        m = (0.5 * (p + q))
        return 0.5 * (self.kl(m, p) + self.kl(m, q))

jsd = JSD()








#--------------------------------------------------------


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

    def __init__(self,pairDynamic, pairStatic):
        super(Pairdata, self).__init__()
        
        self.pairDynamic       = pairDynamic
        self.pairStatic      = pairStatic

    def __len__(self):
        return self.pairDynamic.shape[0]

    def __getitem__(self, index):
        
        return index, self.pairDynamic[index], self.pairStatic[index]

#-------------------------------------------------------------------------------





np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

nb_samples = 200

save_test_dataset = False

fast = True




model = VAE(args).cuda()


model = model.cuda()
network=torch.load(args.ae_weight)

print('Weights loaded')

# summary(model, (3,args.beam,128))

model.load_state_dict(network['gen_dict'])
# model.load_state_dict(network['state_dict_gen_st'])
model.eval() 
print('its there')
outer = 1024
# loss1 = EMD
# loss_fn1 = loss1()
loss = get_chamfer_dist
# size = 10 if 'emd' in sys.argv[3] else 5
# size = 2

npydata =[8]
tot = 0
criterion = nn.MSELoss()
with torch.no_grad():
    
  for i in npydata:
    print(i)

    lidarDy    = (np.load(args.data + "dynamic/d{}.npy".format(str(i)))[:,:,::int(64/args.beam),::args.dim]).astype('float32')
    lidarSt  = (np.load(args.data + "static/s{}.npy".format(str(i)))[:,:,::int(64/args.beam),::args.dim]).astype('float32')
    # lidarDy    = (np.load(args.data + "dynamic/{}.npy".format(str(i)))[:,:,::int(64/args.beam),::args.dim]).astype('float32')
    # lidarSt  = (np.load(args.data + "static/{}.npy".format(str(i)))[:,:,::int(64/args.beam),::args.dim]).astype('float32')
    print(lidarDy.shape, lidarSt.shape)
    
    test_loader    = Pairdata(lidarDy, lidarSt)
    loader = (torch.utils.data.DataLoader(test_loader, batch_size= args.batch_size,
                        shuffle=False, num_workers=1, drop_last=True)) #False))

    loss_fn = loss()
    # process_input = (lambda x : x) if model.args.no_polar else to_polar
    process_input = from_polar if args.no_polar else lambda x : x
    
    # noisy reconstruction
    for noise in [0]:#0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.][::(2 if fast else 1)]: 
        losses0 = []
        losses1 = []
        losses2 = []
        losses3 = []
        losses4 = []
        ind = -1 
        for batch in loader:
            lidar = batch[1].cuda()
    
            lidar_s=batch[2].cuda()
       
            recon,_,_  = model((process_input(lidar)))
            # print(recon.shape)  output is polar
            recon = (recon[:,:,:args.beam])
            # print(recon.shape, lidar_s.shape)
            # exit(0)
            # print(recon.shape)

            recon = from_polar(recon)
            lidar_s = from_polar(lidar_s)
            # recon = F.log_softmax((recon))
            # lidar_s = F.log_softmax((lidar_s))

            # middle =(recon+lidarStat)/2
            # print(recon.shape, lidar_s.shape)
            # exit(0)
            ind+=1
            # print(ind)
            
            # losses += [directed_hausdorff(from_polar(recon).reshape(-1,3,args.beam*int(outer/args.dim)), from_polar(lidarStat).reshape(-1,3,args.beam*int(outer/args.dim)))]
            # losses += [ (criterion(recon.reshape(-1,3, args.beam * int(outer/args.dim)) , lidarStat.reshape(-1,3, args.beam * int(outer/args.dim)) )) ]
            losses0 += [ torch.sqrt(criterion(recon.reshape(-1,3, args.beam * int(outer/args.dim)) , lidar_s.reshape(-1,3, args.beam * int(outer/args.dim)) )) ]
            # print(torch.stack(losses).mean().item())    
            
            # losses  += [ (kl_loss(recon.reshape(-1,3, args.beam, int(outer/args.dim)) , lidarStat.reshape(-1,3, args.beam, int(outer/args.dim)) )) ]
            # losses1 += [ kl_loss(lidarStat.reshape(-1,3, args.beam, int(outer/args.dim)) , recon.reshape(-1,3, args.beam, int(outer/args.dim) )) ]
            losses1 += [ jsd(lidar_s.reshape(-1,3, args.beam, int(outer/args.dim)) , recon.reshape(-1,3, args.beam, int(outer/args.dim) )) ]
            # losses2 += [ calc_mmd_loss(((lidar_s.reshape(-1,3, args.beam* int(outer/args.dim)))) , (recon.reshape(-1,3, args.beam* int(outer/args.dim)) )) ]
            # losses3 += torch.Tensor([emd_samples(recon.detach().cpu().numpy(), lidar_s.detach().cpu().numpy(), bins = 40)])
            losses4 += [loss_fn(recon.reshape(-1, 3, args.beam, int(outer/args.dim)), lidar_s.reshape(-1, 3, args.beam ,int(outer/args.dim))).cuda()]    
            # print(torch.stack(losses).mean().item())    
            
            
 
 

        losses0  = torch.stack(losses0).mean().item()
        losses1 = torch.stack(losses1).mean().item()
        # losses2 = torch.stack(losses2).mean().item()
        # losses3 = torch.stack(losses3).mean().item()
        losses4 = torch.stack(losses4).mean().item()
        # print(losses, losses1)
        # mse+=losses
        # rmse+=losses1
        print('RMSE {}: {:.7f}'.format(i, (losses0)))
        print('JSD  {}: {:.7f}'.format(i, losses1))
        # print('MMD  {}: {:.7f}'.format(i, losses2))
        # print('EMD {}: {:.7f}'.format(i, losses3))
        print('CD {}: {:.7f}'.format(i, losses4))

       
# print(mse/len(npydata))
# print(rmse/len(npydata))
   

 
 

       