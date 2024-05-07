<div align="center">
<h1>GLiDR: Topologically Regularized Graph Generative Network for Sparse LiDAR Point Clouds</h1>

**CVPR 2024**  
<a href='https://kshitijbhat.github.io/glidr/' style="margin-right: 20px;"><img src='https://img.shields.io/badge/Project Page-GLiDR-darkgreen' alt='Project Page'></a>
<a href="https://arxiv.org/abs/2312.00068" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-arXiv-maroon' alt='arXiv page'></a>
<a href="https://arxiv.org/abs/2312.00068" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-CvF-blue' alt='IEEE Xplore Paper'></a>
<a href="https://arxiv.org/abs/2312.00068" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Supplementary-CvF-blue' alt='IEEE Xplore Paper'></a>

[Prashant Kumar](https://prashkmr.github.io)\*,
[Kshitij Bhat](https://prashkmr.github.io),
[Vedang Nadkarni](https://scholar.google.com/citations?user=seg1E8AAAAAJ&hl=en),
[Prem Kumar Kalra](https://www.cse.iitd.ac.in/~pkalra/)<br/>

</div>

![driving.png](assets/driving.png)



## News
- [Feb 2024] GLiDR accepted in CVPR'2024.


## Installation
The code was tested using pytorch 3.9.18. 
For topology based installation requirement we use the this fantastic [TopologyLayer](https://github.com/bruel-gabrielsson/TopologyLayer) installation. It mostly works out of the box. There may be some issues while installing Dionysus Drivers for persistence calculation. Pleae refer to the [issues](https://github.com/bruel-gabrielsson/TopologyLayer/issues) at the above repository for quick resolution.

Alternatively you may also follow the below conda environment setup.
``` bash
git clone https://github.com/prashkmr/GLiDR
cd GLiDR
conda env create -f env.yml
conda activate GLiDR
```


## Dataset Setup
For training GLiDR we require paired correspondences for static and dynamic scans. These are available for [CARLA](https://github.com/dslrproject/dslr/tree/master/Data) and [ARD-16](https://github.com/dslrproject/dslr/tree/master/Data). 
For KITTI, we generate the paired correspondence using a novel method that we develop in our paper. See our Supplementary material for precise details.

- [CARLA] - The CARLA dataset consists of 15 correspondence numpy arrays (0..7 for training and the rest for testing) for training and testing the model. We use the first three numpy arrays(0,1,2) to train the model (due to time intensive persistence computation). We further test the model on 8 (8,9...15) numpy arrays.  For SLAM we use the 4 CARLA SLAM sequences made available. The data is available [here](https://github.com/dslrproject/dslr/tree/master/Data). Download them in the data/carla folder and split into 2 subfolders - static and dynamic. 
- [KITTI] - For KITTI we use the KITTI Odometry sequences for training the model. We transform the 11 sequences into numpy files and make them available here. We use the sequence 8 for testing the model and the rest for training our model For SLAM we use all the Odometry sequences (0 to 10). The static-dynamic corresponding pairs for KITTI are available [here](https://www.kaggle.com/datasets/prashk1312/kitti-static-dynamic-correpsondence). Download them in the data/kitti folder and split into 2 subfolders - static and dynamic. 
- [ARD-16] - The ARD-16 dataset consists of 4 numpy arrays (0..3). We use three (0..2) numpy arrays for training our model and the fourth numpy array for testing our model. The data is available [here](https://github.com/dslrproject/dslr/tree/master/Data). Download them in the data/ard folder and split into 2 subfolders - static and dynamic. 





## Training 
We trained our models on a single NVIDIA A100 GPU. We have 2 version per dataset - sparse and dense for training.  For KITTI and CARLA sparse and dense version consists of 16 and 64 beam respectively. For KITT the sparse and dense version consists of 8 and 16 beams respectively.

1. **Train on KITTI dataset**:  
  Sparse Version

  `cd kitti/`
  
   `python GLiDR_kitti.py --data data/kitti/ --exp_name glidr_kitti_sparse --beam 16 --dim 8 --batch_size 32 --mode kitti`

   Dense Version

  `cd kitti/`
  
  `python GLiDR_kitti.py --data data/kitti/ --exp_name glidr_kitti_dense --beam 64 --dim 8 --batch_size 8 --mode kitti`

   --beam : Denotes the number of beam that ar allowed in the LiDAR. 
   --dim  : Sparsifies the outermost dimension of the range image (for CARLA, outermost dimesion is 1024). For more details on this, please refer to Section 5.2 of the paper.


1. **Train on CARLA dataset**:  
  Sparse Version

 ` cd carla/`
 
` python GLiDR_kitti.py --data data/carla/ --exp_name glidr_carla_sparse --beam 16 --dim 4 --batch_size 32 --mode carla`

  Dense Version

  `cd carla/`
  
  `python GLiDR_kitti.py --data data/kitti/ --exp_name glidr_carla_dense --beam 64 --dim 4 --batch_size 8 --mode carla`

   --beam : Denotes the number of beam that ar allowed in the LiDAR. 
   --dim  : Sparsifies the outermost dimension of the range image (for CARLA, outermost dimesion is 512). For more details on this, please refer to Section 5.2 of the paper.



### Contact
If you have any questions about our code or paper, kindly raise an issue on this repository.

### Acknowledgment
We thank [Kartik Anand](https://github.com/k-styles) for assistance with the experiments. 
Our source code is inspired from [VPD](https://github.com/wl-zhao/VPD) and [PixelFormer](https://github.com/ashutosh1807/PixelFormer). We thank their authors for publicly releasing the code.

### BibTeX (Citation)
If you find our work useful in your research, please consider citing using:
``` bibtex
@article{patni2024ecodepth,
  title={ECoDepth: Effective Conditioning of Diffusion Models for Monocular Depth Estimation},
  author={Patni, Suraj and Agarwal, Aradhye and Arora, Chetan},
  journal={arXiv preprint arXiv:2403.18807},
  year={2024}
}
```


