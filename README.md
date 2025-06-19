<div align="center">
<h1>GLiDR: Topologically Regularized Graph Generative Network for Sparse LiDAR Point Clouds</h1>

**CVPR 2024**

<a href='https://kshitijbhat.github.io/glidr/' style="margin-right: 20px;"><img src='https://img.shields.io/badge/Project Page-GLiDR-darkgreen' alt='Project Page'></a>
<a href="https://arxiv.org/abs/2312.00068" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-arXiv-maroon' alt='arXiv page'></a>
<a href="https://www.youtube.com/watch?v=oqJLJL-mYqg" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Video-yt?logo=youtube&logoColor=red&labelColor=grey&color=grey' alt='Video'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Kumar_GLiDR_Topologically_Regularized_Graph_Generative_Network_for_Sparse_LiDAR_Point_CVPR_2024_paper.pdf" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Paper-CvF-blue' alt='IEEE Xplore Paper'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Kumar_GLiDR_Topologically_Regularized_Graph_Generative_Network_for_Sparse_LiDAR_Point_CVPR_2024_paper.html" style="margin-right: 20px;"><img src='https://img.shields.io/badge/Supplementary-CvF-blue' alt='IEEE Xplore Paper'></a>

[Prashant Kumar](https://prashkmr.github.io)\*,
[Kshitij Bhat](https://kshitijbhat.github.io),
[Vedang Nadkarni](https://scholar.google.com/citations?user=seg1E8AAAAAJ&hl=en),
[Prem Kumar Kalra](https://www.cse.iitd.ac.in/~pkalra/)<br/>

</div>

![GLiDR Poster](GLiDR_Poster.png)



## News
- [Aug 2024] Evaluation code published.
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

- [CARLA] - The CARLA dataset consists of 15 correspondence numpy arrays (0..7 for training and the rest for testing) for training and testing the model. We use the first three numpy arrays(0,1,2) to train the model (due to time intensive persistence computation). We further test the model on 8 (8,9...15) numpy arrays.  For SLAM we use the 4 CARLA SLAM sequences made available. The data is available [here](https://github.com/dslrproject/dslr/tree/master/Data). Download them in the data/carla folder and split into 2 subfolders - static and dynamic.  CARLA data is by deafult in polar format for range image (2,64,1024). To convert it into cartesian format, use from_polar_np(..) from utils512.py.
- [KITTI] - For KITTI we use the KITTI Odometry sequences for training the model. We transform the 11 sequences into numpy files and make them available here. We use the sequence 8 for testing the model and the rest for training our model For SLAM we use all the Odometry sequences (0 to 10). The static-dynamic corresponding pairs for KITTI are available [here](https://www.kaggle.com/datasets/prashk1312/kitti-static-dynamic-correpsondence). Download them in the data/kitti folder and split into 2 subfolders - static and dynamic. 
- [ARD-16] - The ARD-16 dataset consists of 4 numpy arrays (0..3). We use three (0..2) numpy arrays for training our model and the fourth numpy array for testing our model. The data is available [here](https://github.com/dslrproject/dslr/tree/master/Data). Download them in the data/ard folder and split into 2 subfolders - static and dynamic. 





## Training 
We trained our models on a single NVIDIA A100 GPU. We have 2 version per dataset - sparse and dense for training.  For KITTI and CARLA sparse and dense version consists of 16 and 64 beam respectively. For KITTI the sparse and dense version consists of 8 and 16 beams respectively.

1. **Train on KITTI dataset**:

The training requires 2 arguments.
--beam : Denotes the number of beam that ar allowed in the LiDAR. 
--dim  : Sparsifies the outermost dimension of the range image (for KITTI and ARD-16, outermost dimesion is 1024, for CARLA it is 512). For more details on this, please refer to Section 5.2 of the paper.
   
Sparse Version

``` bash
cd kitti/

python GLiDR_kitti.py --data data/kitti/ --exp_name glidr_kitti_sparse --beam 16 --dim 8 --batch_size 32 --mode kitti
```

Dense Version

``` bash
cd kitti/
python GLiDR_kitti.py --data data/kitti/ --exp_name glidr_kitti_dense --beam 64 --dim 8 --batch_size 8 --mode kitti
``` 

2. **Train on CARLA dataset**:  

Sparse Version
``` bash
cd carla/
python GLiDR_kitti.py --data data/carla/ --exp_name glidr_carla_sparse --beam 16 --dim 4 --batch_size 32 --mode carla
```

Dense Version

``` bash
cd carla/
python GLiDR_kitti.py --data data/kitti/ --exp_name glidr_carla_dense --beam 64 --dim 4 --batch_size 8 --mode carla
```

3. **Train on CARLA dataset**:  
Sparse Version

``` bash
cd carla/
python GLiDR_kitti.py --data data/carla/ --exp_name glidr_carla_sparse --beam 16 --dim 4 --batch_size 32 --mode carla
```
Dense Version

``` bash
cd carla/
python GLiDR_kitti.py --data data/kitti/ --exp_name glidr_carla_dense --beam 64 --dim 4 --batch_size 8 --mode carla
``` 


### Contact
If you have any questions about our code or paper, kindly raise an issue on this repository.


### BibTeX (Citation)
If you find our work useful in your research, please consider citing using:
``` bibtex
@InProceedings{Kumar_2024_CVPR,
    author    = {Kumar, Prashant and Bhat, Kshitij Madhav and Nadkarni, Vedang Bhupesh Shenvi and Kalra, Prem},
    title     = {GLiDR: Topologically Regularized Graph Generative Network for Sparse LiDAR Point Clouds},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {15152-15161}
}

```


