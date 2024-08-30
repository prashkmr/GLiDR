import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import numpy as np
# from vit_pytorch import ViT
import os
import pdb
from utils512 import *


# --------------------------------------------------------------------------
# Core Models
# --------------------------------------------------------------------------
class netG(nn.Module):  #decoder
    def __init__(self, args, nz=100, ngf=64, nc=3, base=4, ff=(1,2)):
        super(netG, self).__init__()
        self.args = args
        conv = nn.ConvTranspose2d

        layers  = []
        layers += [nn.ConvTranspose2d(nz, ngf * 8, ff, 1, 0, bias=False)]
        layers += [nn.BatchNorm2d(ngf * 8)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 8, ngf * 4, (4,4), stride=2, padding=(0,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 4)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 4, ngf * 4, (2,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 4)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 4, ngf * 2, (2,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 2)]
        layers += [nn.ReLU(True)]


        layers += [nn.ConvTranspose2d(ngf * 2, ngf * 2, (2,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 2)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 2, ngf, (1,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)]
        layers += [nn.Tanh()]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(-1).unsqueeze(-1)

        return self.main(input)




class netD(nn.Module): #encoder
    def __init__(self, args, ndf=64, nc=2, nz=1, lf=(1,2)):
        super(netD, self).__init__()
        self.encoder = True if nz > 1 else False

        layers  = []
        layers += [nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)]

        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf*2, ndf * 2, 3, 2, 1, bias=False)]

        layers += [nn.BatchNorm2d(ndf * 2)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)]

        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False)]

        layers += [nn.BatchNorm2d(ndf * 4)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]
        layers += [nn.Conv2d(ndf * 4, ndf * 8, (2,4), 2, (0,1), bias=False)]

        layers += [nn.BatchNorm2d(ndf * 8)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        self.main = nn.Sequential(*layers)
        self.out  = nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False)

    def forward(self, input, return_hidden=False):
        if input.size(-1) == 3:
            input = input.transpose(1, 3)

        output_tmp = self.main(input)
        output = self.out(output_tmp)

        if return_hidden:
            return output, output_tmp

        return output if self.encoder else output.view(-1, 1).squeeze(1)









class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args

        if args.atlas_baseline or args.panos_baseline:
            self.AE = AE_AtlasNet(bottleneck_size=args.z_dim,
                                  AE=args.autoencoder,
                                  nb_primitives=args.atlas_baseline)
            self.encode = self.AE.encode
            self.decode = self.AE.decode if args.atlas_baseline else PointGenPSG2(nz=args.z_dim)
        else:
            mult = 1 if args.autoencoder else 2
            self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
            self.decode = netG(args, nz=args.z_dim, nc=2)

    def forward(self, x):
        z = self.encode(x)
        while z.dim() != 2:
            z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)







class scene_discriminator(nn.Module):
    def __init__(self, pose_dim, nf=256):
        super(scene_discriminator, self).__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
                # nn.Dropout(p=0.5),
                nn.Linear(pose_dim*2, int(pose_dim)),
                nn.Sigmoid(),
                # nn.Dropout(p=0.5),
                nn.Linear(int(pose_dim), int(pose_dim/2)),
                nn.Sigmoid(),
                # nn.Dropout(p=0.5),
                nn.Linear(int(pose_dim/2), int(pose_dim/4)),
                nn.Sigmoid(),
                nn.Linear(int(pose_dim/4),int(pose_dim/8)),
                nn.Sigmoid(),
                nn.Linear(int(pose_dim/8),1),
                nn.Sigmoid()
                )


    def forward(self, input1,input2):
        output = self.main(torch.cat((input1, input2),1).view(-1, self.pose_dim*2))
        return output








class netGM(nn.Module):  #decoder
    def __init__(self, args, nz=100, ngf=64, nc=3, base=4, ff=(1,4)):
        super(netGM, self).__init__()
        self.args = args
        conv = nn.ConvTranspose2d

        layers  = []
        layers += [nn.ConvTranspose2d(nz, ngf * 8, ff, 1, 0, bias=False)]
        layers += [nn.BatchNorm2d(ngf * 8)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 8, ngf * 4, (2,4), stride=2, padding=(0,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 4)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 4, ngf * 4, (4,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 4)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 4, ngf * 2, (3,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 2)]
        layers += [nn.ReLU(True)]


        layers += [nn.ConvTranspose2d(ngf * 2, ngf * 2, (2,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf * 2)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf * 2, ngf, (1,4), stride=2, padding=(1,1), bias=False)]
        layers += [nn.BatchNorm2d(ngf)]
        layers += [nn.ReLU(True)]

        layers += [nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)]
        layers += [nn.Sigmoid()]

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if len(input.shape) == 2:
            input = input.unsqueeze(-1).unsqueeze(-1)

        return self.main(input)








class VAEVit256_mask(nn.Module):
    def __init__(self, args):
        super(VAEVit256_mask, self).__init__()
        self.args = args
        mult = 1
        
        v = ViT(
            image_size = 256,
            patch_size = 8,
            num_classes = 1000,
            dim = args.z_dim * mult,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        y = list(v.transformer.layers.children())[0]
        y1 = list(v.transformer.layers.children())[1]

        self.v1 = nn.Sequential(*list(v.children())[:2])
        self.v2 = nn.Sequential(*y)
        self.v3 = nn.Sequential(*y1)

        m = ViT(
            image_size = 256,
            patch_size = 8,
            num_classes = 1000,
            dim = args.z_dim * mult,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        m1 = list(m.transformer.layers.children())[0]
        m2 = list(m.transformer.layers.children())[1]

        self.vm1 = nn.Sequential(*list(m.children())[:2])
        self.vm2 = nn.Sequential(*m1)
        self.vm3 = nn.Sequential(*m2)

        

        # self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
        self.decode = netG(args, nz=args.z_dim, nc=2)

    def forward(self, x, mask):
        mask1 = self.vm1(mask)
        mask1 = self.vm2(mask1)
        mask2 = self.vm3(mask1)

        x = self.v1(x)
        x = self.v2(x)
        x = x*mask1
        z = self.v3(x)
        z = z*mask2

        # x = self.v4(x)
        # z = self.v5(x)
        z = z.sum(1)
        z = z.view(-1, 160, 1, 1)
        # print(z.shape)


        # while z.dim() != 2:
        #     z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z.view(-1,1)
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z.view(-1,1)

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)






class VAEVit256_4Layers(nn.Module):
    def __init__(self, args):
        super(VAEVit256_4Layers, self).__init__()
        self.args = args
        mult = 1
        
        v = ViT(
            image_size = 256,
            patch_size = 8,
            num_classes = 1000,
            dim = args.z_dim * mult,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        y = list(v.transformer.layers.children())[0]
        y1 = list(v.transformer.layers.children())[1]
        y2 = list(v.transformer.layers.children())[2]
        y3 = list(v.transformer.layers.children())[3]


        self.v1 = nn.Sequential(*list(v.children())[:2])
        self.v2 = nn.Sequential(*y)
        self.v3 = nn.Sequential(*y1)
        self.v4 = nn.Sequential(*y2)
        self.v5 = nn.Sequential(*y3)
 
        

        # self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
        self.decode = netG(args, nz=args.z_dim, nc=2)

    def forward(self, x):
        x = self.v1(x)
        x = self.v2(x)
        x = self.v3(x)
        x = self.v4(x)
        z = self.v5(x)
        z = z.sum(1)
        z = z.view(-1, 160, 1, 1)
        # print(z.shape)


        # while z.dim() != 2:
        #     z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)

class VAEVit256(nn.Module):
    def __init__(self, args):
        super(VAEVit256, self).__init__()
        self.args = args
        mult = 1
        
        v = ViT(
            image_size = 256,
            patch_size = 8,
            num_classes = 1000,
            dim = args.z_dim * mult,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        y = list(v.transformer.layers.children())[0]
        y1 = list(v.transformer.layers.children())[1]


        self.v1 = nn.Sequential(*list(v.children())[:2])
        self.v2 = nn.Sequential(*y)
        self.v3 = nn.Sequential(*y1)
 
        

        # self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
        self.decode = netG(args, nz=args.z_dim, nc=2)

    def forward(self, x):
        x = self.v1(x)
        x = self.v2(x)
        z = self.v3(x)
        z = z.sum(1)
        z = z.view(-1, 160, 1, 1)
        # print(z.shape)


        # while z.dim() != 2:
        #     z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)







class VAEMask2Mask(nn.Module):
    def __init__(self, args):
        super(VAEMask2Mask, self).__init__()
        self.args = args

        if args.atlas_baseline or args.panos_baseline:
            self.AE = AE_AtlasNet(bottleneck_size=args.z_dim,
                                  AE=args.autoencoder,
                                  nb_primitives=args.atlas_baseline)
            self.encode = self.AE.encode
            self.decode = self.AE.decode if args.atlas_baseline else PointGenPSG2(nz=args.z_dim)
        else:
            mult = 1 if args.autoencoder else 2
            self.maskattention = masknet2(args, nz=args.z_dim * mult, nc=2)
            self.encode = netD2(args, nz=args.z_dim * mult, nc=2)
            self.decode = netG(args, nz=args.z_dim, nc=2)
            self.maskdecode = netGM(args, nz=args.z_dim, nc=2)

    def forward(self, x, x_mask):
        k, o = self.maskattention(x_mask)
        z = self.encode(x, k)
        w = self.maskdecode(o)
        # print(z.shape)
        # while z.dim() != 2:
        #     z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z, w
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE2.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z, w

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)

    
class VAE3(nn.Module):
    def __init__(self, args):
        super(VAE3, self).__init__()
        self.args = args

        if args.atlas_baseline or args.panos_baseline:
            self.AE = AE_AtlasNet(bottleneck_size=args.z_dim,
                                  AE=args.autoencoder,
                                  nb_primitives=args.atlas_baseline)
            self.encode = self.AE.encode
            self.decode = self.AE.decode if args.atlas_baseline else PointGenPSG2(nz=args.z_dim)
        else:
            mult = 1 if args.autoencoder else 2
            self.maskattention = masknet(args, nz=args.z_dim * mult)
            self.encode = netD3(args, nz=args.z_dim * mult, nc=3)
            self.decode = netGA(args, nz=args.z_dim, nc=2)

    def forward(self, x):
        #k = self.maskattention(x_mask)
        z = self.encode(x)
        # print(z.shape)
        # while z.dim() != 2:
        #     z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z) #None,z
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out #kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)












class VAEVit(nn.Module):
    def __init__(self, args):
        super(VAEVit, self).__init__()
        self.args = args
        mult = 1
        
        v = ViT(
            image_size = 512,
            patch_size = 8,
            num_classes = 1000,
            dim = args.z_dim * mult,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )

        y = list(v.transformer.layers.children())[0]
        y1 = list(v.transformer.layers.children())[1]
        # y2 = list(v.transformer.layers.children())[2]
        # y3 = list(v.transformer.layers.children())[3]

        self.v1 = nn.Sequential(*list(v.children())[:2])
        self.v2 = nn.Sequential(*y)
        self.v3 = nn.Sequential(*y1)
        # self.v4 = nn.Sequential(*y2)
        # self.v5 = nn.Sequential(*y3)
        

        # self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
        self.decode = netG(args, nz=args.z_dim, nc=2)

    def forward(self, x):
        x = self.v1(x)
        x = self.v2(x)
        z = self.v3(x)
        # x = self.v4(x)
        # z = self.v5(x)
        z = z.view(-1, 160, 64, 5)
        z = z.mean(2).unsqueeze(2)
        # print(z.shape)


        # while z.dim() != 2:
        #     z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)






class VAEM(nn.Module):
    def __init__(self, args):
        super(VAEM, self).__init__()
        self.args = args

        if args.atlas_baseline or args.panos_baseline:
            self.AE = AE_AtlasNet(bottleneck_size=args.z_dim,
                                  AE=args.autoencoder,
                                  nb_primitives=args.atlas_baseline)
            self.encode = self.AE.encode
            self.decode = self.AE.decode if args.atlas_baseline else PointGenPSG2(nz=args.z_dim)
        else:
            mult = 1 if args.autoencoder else 2
            self.encode = netD(args, nz=args.z_dim * mult, nc=3 if args.no_polar else 2)
            self.decode = netGM(args, nz=args.z_dim, nc=1)

    def forward(self, x):
        z = self.encode(x)
        while z.dim() != 2:
            z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z
        else:
            #This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)          #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

        # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
        
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAEM.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)














# --------------------------------------------------------------------------
# Baseline (AtlasNet), taken from https://github.com/ThibaultGROUEIX/AtlasNet
# --------------------------------------------------------------------------
class PointNetfeat_(nn.Module):
    def __init__(self, num_points = 40 * 256, global_feat = True):
        super(PointNetfeat_, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
        
        
    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 128):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AE_AtlasNet(nn.Module):
    def __init__(self, num_points = 40 * 256, bottleneck_size = 1024, nb_primitives = 2, AE=True):
        super(AE_AtlasNet, self).__init__()
        bot_enc = bottleneck_size if AE else 2 * bottleneck_size
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat_(num_points, global_feat=True),
        nn.Linear(1024, bot_enc),
        nn.BatchNorm1d( bot_enc),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 + self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def encode(self, x):
        if x.dim() == 4 :
            if x.size(1) != 3:
                assert x.size(-1) == 3
                x = x.permute(0, 3, 1, 2).contiguous()
            x = x.reshape(x.size(0), 3, -1)
        else:
            if x.size(1) != 3:
                assert x.size(-1) == 3
                x = x.transpose(-1, -2).contiguous()

        x = self.encoder(x)
        return x

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = (torch.cuda.FloatTensor(x.size(0),2,self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()



#if __name__ == '__main__':
 #   points = torch.cuda.FloatTensor(10, 3, 40, 256).normal_()
  #  AE = AE_AtlasNet(num_points = 40 * 256).cuda()
   # out = AE(points)
    #loss = get_chamfer_dist()(points, out)
    #x =1


# --------------------------------------------------------------------------
# Baseline (Panos's paper)
# --------------------------------------------------------------------------
class PointGenPSG2(nn.Module):
    def __init__(self, nz=100, num_points = 40 * 256):
        super(PointGenPSG2, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3 // 2)

        self.fc11 = nn.Linear(nz, 256)
        self.fc21 = nn.Linear(256, 512)
        self.fc31 = nn.Linear(512, 1024)
        self.fc41 = nn.Linear(1024, self.num_points * 3 // 2)
        self.th = nn.Tanh()
        self.nz = nz


    def forward(self, x):
        batchsize = x.size()[0]

        x1 = x
        x2 = x
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, -1)

        x2 = F.relu(self.fc11(x2))
        x2 = F.relu(self.fc21(x2))
        x2 = F.relu(self.fc31(x2))
        x2 = self.th(self.fc41(x2))
        x2 = x2.view(batchsize, 3, -1)

        return torch.cat([x1, x2], 2)



class masknet2(nn.Module): #encoder
    def __init__(self, args, ndf=64, nc=1, nz=1, lf=(1,4)):
        super(masknet2, self).__init__()

        
        self.layers1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.layers2 = nn.LeakyReLU(0.2, inplace=True)
        self.layers3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)

        self.layers4 = nn.LeakyReLU(0.2, inplace=True)
        self.layers5 = nn.Conv2d(ndf*2, ndf * 2, 3, 2, 1, bias=False)

        self.layers6 = nn.BatchNorm2d(ndf * 2)
        self.layers7 = nn.LeakyReLU(0.2, inplace=True)
        self.layers8 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)

        self.layers9 = nn.LeakyReLU(0.2, inplace=True)
        self.layers10 = nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False)

        self.layers11 = nn.BatchNorm2d(ndf * 4)
        self.layers12 = nn.LeakyReLU(0.2, inplace=True)
        self.layers13 = nn.Conv2d(ndf * 4, ndf * 8, (2,4), 2, (0,1), bias=False)

        self.layers14 = nn.BatchNorm2d(ndf * 8)
        self.layers15 = nn.LeakyReLU(0.2, inplace=True)

        self.out  = nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False)

    def forward(self, inputs):
        if inputs.size(-1) == 1:
            inputs = inputs.transpose(1, 3)
        #print('mask:', inputs.shape)
        
        #print('mask:',inputs.shape)
        l1 = self.layers1(inputs)
        l2 = self.layers2(l1)
        l3 = self.layers3(l2)
        l4 = self.layers4(l3)
        l5 = self.layers5(l4)
        l6 = self.layers6(l5)
        l7 = self.layers7(l6)
        l8 = self.layers8(l7)
        l9 = self.layers9(l8)
        l10 = self.layers10(l9)
        l11 = self.layers11(l10)
        l12 = self.layers12(l11)
        l13 = self.layers13(l12)
        l14 = self.layers14(l13)
        l15 = self.layers15(l14)

        output = self.out(l15)

        return [l3, l5, l8, l10, l13], output




class netD2(nn.Module): #encoder
    def __init__(self, args, ndf=64, nc=2, nz=1, lf=(1,8)):
        super(netD2, self).__init__()
        self.encoder = True if nz > 1 else False

        
        self.layers1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.layers2 = nn.LeakyReLU(0.2, inplace=True)
        self.layers3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)

        self.layers4 = nn.LeakyReLU(0.2, inplace=True)
        self.layers5 = nn.Conv2d(ndf*2, ndf * 2, 3, 2, 1, bias=False)

        self.layers6 = nn.BatchNorm2d(ndf * 2)
        self.layers7 = nn.LeakyReLU(0.2, inplace=True)
        self.layers8 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)

        self.layers9 = nn.LeakyReLU(0.2, inplace=True)
        self.layers10 = nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False)

        self.layers11 = nn.BatchNorm2d(ndf * 4)
        self.layers12 = nn.LeakyReLU(0.2, inplace=True)
        self.layers13 = nn.Conv2d(ndf * 4, ndf * 8, (2,4), 2, (0,1), bias=False)

        self.layers14 = nn.BatchNorm2d(ndf * 8)
        self.layers15 = nn.LeakyReLU(0.2, inplace=True)

    
        self.out  = nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False)

    def forward(self, inputs, laterals, return_hidden=False):
        if inputs.size(-1) == 3:
            inputs = inputs.transpose(1, 3)
        #print('netd:',inputs.shape)

        l1 = self.layers1(inputs)
        l2 = self.layers2(l1)
        l3 = self.layers3(l2)
        l3 = l3*laterals[0]

        l4 = self.layers4(l3)
        l5 = self.layers5(l4)
        l5 = l5*laterals[1]

        l6 = self.layers6(l5)
        l7 = self.layers7(l6)
        l8 = self.layers8(l7)
        l8 = l8*laterals[2]

        l9 = self.layers9(l8)
        l10 = self.layers10(l9)
        l10 = l10*laterals[3]

        l11 = self.layers11(l10)
        l12 = self.layers12(l11)
        l13 = self.layers13(l12)
        l13 = l13*laterals[4]
        #print(l13.shape)
        

        l14 = self.layers14(l13)
        l15 = self.layers15(l14)
        output = self.out(l15)
        

        if return_hidden:
            return output, output_tmp

        return output if self.encoder else output.view(-1, 1).squeeze(1)

class masknet(nn.Module): #encoder
    def __init__(self, args, ndf=64, nc=2, nz=1, lf=(1,8)):
        super(masknet, self).__init__()

        
        self.layers1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.layers2 = nn.LeakyReLU(0.2, inplace=True)
        self.layers3 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)

        self.layers4 = nn.LeakyReLU(0.2, inplace=True)
        self.layers5 = nn.Conv2d(ndf*2, ndf * 2, 3, 2, 1, bias=False)

        self.layers6 = nn.BatchNorm2d(ndf * 2)
        self.layers7 = nn.LeakyReLU(0.2, inplace=True)
        self.layers8 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)

        self.layers9 = nn.LeakyReLU(0.2, inplace=True)
        self.layers10 = nn.Conv2d(ndf * 4, ndf * 4, 3, 2, 1, bias=False)

        self.layers11 = nn.BatchNorm2d(ndf * 4)
        self.layers12 = nn.LeakyReLU(0.2, inplace=True)
        self.layers13 = nn.Conv2d(ndf * 4, ndf * 8, (2,4), 2, (0,1), bias=False)


    def forward(self, inputs):
        if inputs.size(-1) == 1:
            inputs = inputs.transpose(1, 3)
        
        #print('mask:',inputs.shape)
        l1 = self.layers1(inputs)
        l2 = self.layers2(l1)
        l3 = self.layers3(l2)
        l4 = self.layers4(l3)
        l5 = self.layers5(l4)
        l6 = self.layers6(l5)
        l7 = self.layers7(l6)
        l8 = self.layers8(l7)
        l9 = self.layers9(l8)
        l10 = self.layers10(l9)
        l11 = self.layers11(l10)
        l12 = self.layers12(l11)
        l13 = self.layers13(l12)

        return [nn.AdaptiveAvgPool2d(1)(l3), nn.AdaptiveAvgPool2d(1)(l5), nn.AdaptiveAvgPool2d(1)(l8), 
                nn.AdaptiveAvgPool2d(1)(l10), nn.AdaptiveAvgPool2d(1)(l13)
                ]


class VAE2(nn.Module):
    def __init__(self, args):
        super(VAE2, self).__init__()
        self.args = args

        if args.atlas_baseline or args.panos_baseline:
            self.AE = AE_AtlasNet(bottleneck_size=args.z_dim,
                                  AE=args.autoencoder,
                                  nb_primitives=args.atlas_baseline)
            self.encode = self.AE.encode
            self.decode = self.AE.decode if args.atlas_baseline else PointGenPSG2(nz=args.z_dim)
        else:
            mult = 1 if args.autoencoder else 2
            self.maskattention = masknet(args, nz=args.z_dim * mult)
            n =3 if args.no_polar else 2
            self.encode = netD2(args, nz=args.z_dim * mult, nc=n)
            self.decode = netG(args, nz=args.z_dim, nc=2)

    def forward(self, x, x_mask):
        k = self.maskattention(x_mask)
        z = self.encode(x, k)
        # print(z.shape)
        # raise SystemError
        # while z.dim() != 2:
        #     z = z.squeeze(-1)

        if self.args.autoencoder:
            return self.decode(z), None,z
        else:
        	#This is mu and sigma of the distribution that is sampled
            mu, logvar = torch.chunk(z, 2, dim=1)    #sample mu and variace, here it took logvar for numbeical stabilility purpose, therefore below it took exp for the 
            std = torch.exp(0.5 * logvar)            # get the standard deviation
            eps = torch.randn_like(std)   	     #Returns a tensor with the same size as std that is filled with random numbers from a normal distribution with mean 0 and variance 1.           

	    # This is the link of the blog that to get the understanding of the VAE Technique: https://kite.com/python/docs/torch.randn_like
	    
            # simple way to get better reconstructions. Note that this is not a valid NLL_test bd 
            z = eps.mul(std).add_(mu) if self.training else mu    # This is the equation for sampling data from the distribution that we got from the VAE.

            kl = VAE.gaussian_kl(mu, logvar)

            out = self.decode(z)
            return out, kl,z

    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1)