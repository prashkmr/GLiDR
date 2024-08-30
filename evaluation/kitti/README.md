## Evalaution For 16-beam results for KITTI using GliDR 

Chamfer Distance

`python eval_carla_baseline__chamfer.py --data [path to paired scan folder] --dim 8 --beam 16  --ae_weight model_80.t7 `


JSD

'python eval_carla_baseline_jsd.py --data  [path to paired scan folder] --dim 8 --beam 16  --ae_weight model_80.t7 '
      

RMSE

'python eval_carla_baseline_rmse.py --data  [path to paired scan folder] --dim 8 --beam 16  --ae_weight model_80.t7 '
      

EMD

'python eval_carla_baseline_emd.py --data [path to paired scan folder] --dim 8 --beam 16  --ae_weight model_80.t7 '



--beam : Denotes the number of beam that ar allowed in the LiDAR. 

--dim  : Sparsifies the outermost dimension of the range image (for CARLA, outermost dimesion is 1024). For more details on this, please refer to Section 5.2 of the paper.





## Evalaution For 16-beam results for KITTI using the best baseline - MOVES

MOVES works on the datasets in polar format - therefore we use dataset in polar format for testing MOVES


Chamfer Distance

`python eval_carla_final_cd.py --data [path to the dataset in polar format]    --dim 8 --beam 16 --ae_weight gen_990.pth  --batch_size 64`


JSD

'python eval_carla_final_rmse.py --data [path to the dataset in polar format]    --dim 8 --beam 16 --ae_weight gen_990.pth  --batch_size 64'
      

RMSE

'ppython eval_carla_final_jsd.py --data [path to the dataset in polar format]    --dim 8 --beam 16 --ae_weight gen_990.pth  --batch_size 64 '
      

EMD

'python eval_carla_final_emd.py --data [path to the dataset in polar format]    --dim 8 --beam 16 --ae_weight gen_990.pth  --batch_size 64'



--beam : Denotes the number of beam that ar allowed in the LiDAR. 

--dim  : Sparsifies the outermost dimension of the range image (for CARLA, outermost dimesion is 1024). For more details on this, please refer to Section 5.2 of the paper.













