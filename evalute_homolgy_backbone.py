# Example from README
from topologylayer.nn import AlphaLayer, BarcodePolyFeature
import torch, numpy as np, matplotlib.pyplot as plt
import argparse, os
from tqdm import trange
import matplotlib.pyplot as plt
import open3d as o3d
# random pointcloud
import pdb

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--data',         type=str,   default=64,           help='size of minibatch used during training')
parser.add_argument('--folder',       type=str,   default='',           help='size of minibatch used during training')
parser.add_argument('--ith',       type=int,   default=610,           help='size of minibatch used during training')

parser.add_argument('--sparse',       type=int,   default=1,           help='size of minibatch used during training')
args = parser.parse_args()



def save(x,k, folder):
    y = x.detach().numpy()
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))
    ax[0].scatter(data[:,0], data[:,1], s=0.3, color = 'k')
    ax[0].set_title("Before")
    ax[1].scatter(y[:,0], y[:,1], s=1, color='red')
    ax[1].set_title("After")
    for i in range(2):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].tick_params(bottom=False, left=False)
    plt.savefig('images/' +args.folder + '/holes-'+ str(k) + '.svg',format='svg' ,dpi=1200 )


def save_topo(x,k, folder):
    y = x.detach().numpy()
    np.save('images/' +args.folder + '/' + str(k)+'.npy',y)
    fig  =plt.figure()
    


    # fig, ax = plt.subplots(ncols=1, figsize=(2,4))
    # # ax[0].scatter(data[:,0], data[:,1], s=1, color = 'blue')
    # # ax[0].set_title("Before")
    # ax.scatter(y[:,0], y[:,1], s=1, color='red')
    # # ax[0].set_title("After")


    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(y[:, 0], y[:, 1], y[:, 2], s=0.3, c='k')  # Adjust point size as needed

    # Set labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('3D Point Cloud')

    for i in range(1):
        ax.grid(False)
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_zticklabels([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.tick_params(bottom=False, left=False)
    plt.savefig('images/' +args.folder + '/holes-'+ str(k) + '.svg',format='svg' ,dpi=1200)
    



if not os.path.exists('images/'+str(args.folder)):
	os.makedirs('images/'+str(args.folder))


def convert_pcd_to_npy(pcd_file_path):
    
    # Read the PCD file
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    
    # Convert the point cloud to a NumPy array
    point_cloud_array = np.asarray(pcd.points)

    return point_cloud_array




np.random.seed(0)
data = convert_pcd_to_npy(args.data)[::args.sparse]
# pdb.set_trace()
print(data.shape)

# optimization to increase size of holes
layer = AlphaLayer(maxdim=1)
x = torch.autograd.Variable(torch.tensor(data).type(torch.float), requires_grad=True)
f1 = BarcodePolyFeature(0,1,0)  # dm, p ,q

k=0

# A lr of 1e-1 results in a very noisy backbone with diffusing points in the backbone neighbourhood - npot good 


optimizer = torch.optim.Adam([x], lr=1e-3)
for i in trange(1000):
    optimizer.zero_grad()
    loss = f1(layer(x))
    save_topo(x, k, args.folder)
    k+=1
    loss.backward() 
    optimizer.step()

# save figure
