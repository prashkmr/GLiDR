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

- [CARLA] - The CARLA dataset consists of 15 correspondence numpy arrays (0..7 for training and the rest for testing) for training and testing the model. We use the first three numpy arrays(0,1,2) to train the model (due to time intensive persistence computation). We further test the model on 8 (8,9...15) numpy arrays.  For SLAM we use the 4 CARLA SLAM sequences made available. The data is available [here](https://github.com/dslrproject/dslr/tree/master/Data).
- [KITTI] - For KITTI we use the KITTI Odometry sequences for training the model. We transform the 11 sequences into numpy files and make them available here. We use the sequence 8 for testing the model and the rest for training our model For SLAM we use all the Odometry sequences (0 to 10). The static-dynamic corresponding pairs for KITTI are available [here](https://www.kaggle.com/datasets/prashk1312/kitti-static-dynamic-correpsondence).
- [ARD-16] - The ARD-16 dataset consists of 4 numpy arrays (0..3). We use three (0..2) numpy arrays for training our model and the fourth numpy array for testing our model. The data is available [here](https://github.com/dslrproject/dslr/tree/master/Data).
``` bash


##Training 
We trained our models on 32 batch size using 8xNVIDIA A100 GPUs. Inside the `train_{kitti,nyu}.sh` set the `NPROC_PER_NODE` variable and `--batch_size` argument to the desired values as per your system resources. For our method we set them as `NPROC_PER_NODE=8` and `--batch_size=4` (resulting in a total batch size of 32). Afterwards, navigate to the `depth` directory by executing `cd depth` and follow the instructions:

1. **Train on NYUv2 dataset**:  
`bash train_nyu.sh`  

1. **Train on KITTI dataset**:  
`bash train_kitti.sh`

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


