

## Evalaution For 16-beam results for CARLA using GliDR 

Chamfer Distance
``` bash

 python eval_carla_baseline_nosave_cd.py --data ~/../scratch/prashant/martini-scratch-v1/data/DSLR/ --dim 4 --beam 16 --ae_weight 16/16-128/model_99_prev.t7   --batch_size 32 `
```

JSD

``` bash

'python eval_carla_baseline_nosave_jsd.py --data ~/../scratch/prashant/martini-scratch-v1/data/DSLR/ --dim 4 --beam 16 --ae_weight 16-128/model_99_prev.t7   --batch_size 32
```      

RMSE
``` bash

python eval_carla_baseline_rmse.py --data ~/../scratch/prashant/martini-scratch-v1/data/DSLR/ --dim 4 --beam 16 --ae_weight 16-128/model_99_prev.t7   --batch_size 32 '
```      

EMD
``` bash

python eval_carla_baseline_emd.py --data ~/../scratch/prashant/martini-scratch-v1/data/DSLR/ --dim 4 --beam 16 --ae_weight 16-128/model_99_prev.t7   --batch_size 32 
```


--beam : Denotes the number of beam that ar allowed in the LiDAR. 

--dim  : Sparsifies the outermost dimension of the range image (for CARLA, outermost dimesion is 1024). For more details on this, please refer to Section 5.2 of the paper.





## Evalaution For 16-beam results for CARLA using the best baseline - MOVES

MOVES works on the datasets in polar format - therefore we use dataset in polar format for testing MOVES


Chamfer Distance

``` bash

python eval_carla_final_cd.py --data [location of data in polar format] --dim 4 --beam 16 --ae_weight gen_105.pth --batch_size 64
```

JSD

``` bash

python eval_carla_final_jsd.py --data [location of data in polar format] --dim 4 --beam 16 --ae_weight gen_105.pth --batch_size 64
```      

RMSE

``` bash
python eval_carla_final_rmse.py --data [location of data in polar format] --dim 4 --beam 16 --ae_weight gen_105.pth --batch_size 64 
```      

EMD
``` bash

python eval_carla_final_emd.py --data [location of data in polar format] --dim 4 --beam 16 --ae_weight gen_105.pth --batch_size 64
```


--beam : Denotes the number of beam that ar allowed in the LiDAR. 

--dim  : Sparsifies the outermost dimension of the range image (for CARLA, outermost dimesion is 1024). For more details on this, please refer to Section 5.2 of the paper.




