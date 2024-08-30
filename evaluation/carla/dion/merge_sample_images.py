import cv2
import numpy as np
import os, shutil
from tqdm import tqdm
import glob
import shutil, os

font = cv2.FONT_HERSHEY_SIMPLEX
org = (84, 840)
fontScale = 3
   

color = (255, 255, 255)
thickness = 2


if os.path.exists('merged/'):
    shutil.rmtree('merged')
    os.makedirs('merged')

# baseline = sorted(glob.glob('/content/drive/Shareddrives/Classification/lidar/dslr/overleaf-images/baseline_kitti_2048/*.png'))

# our = sorted(glob.glob('samplesVid/dd9-cropped/*.jpg'))
# original = sorted(glob.glob('samplesVid/movable-9/*.jpg'))

file_num = [i for i in range(2048)]



# 904:1509
our      = ['samplesVid/15-ours/'+str(i)+'.jpg' for i in range(1274)][850:1150]
original = ['samplesVid/d15/'+str(i)+'.jpg' for i in range(1274)][850:1150]
st       = ['samplesVid/s15/'+str(i)+'.jpg' for i in range(1274)][850:1150]
bl       = ['samplesVid/15-bl/'+str(i)+'.jpg' for i in range(1274)][850:1150]
# print(len(baseline), len(our), len(original))

print(len(our), len(original))
count = 0

for static, dynamic_file, reconstructed_file, ours in tqdm(zip(st, original,bl, our), total=len(our)):

    static_img_arr = cv2.imread(static)
    static_img_arr = cv2.putText(static_img_arr, 'Baseline', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    dynamic_img_arr = cv2.imread(dynamic_file)
    dynamic_img_arr = cv2.putText(dynamic_img_arr, 'Baseline', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    reconstructed_img_arr = cv2.imread(reconstructed_file)
    reconstructed_img_arr = cv2.putText(reconstructed_img_arr, 'Ours', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    our_img_arr = cv2.imread(ours)
    our_img_arr = cv2.putText(our_img_arr, 'Baseline', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    # ori = cv2.imread(orig)
    # ori = cv2.putText(ori, 'Original', org, font, 
    #                fontScale, color, thickness, cv2.LINE_AA)
    
    # merged_img_arr = np.concatenate((dynamic_img_arr, reconstructed_img_arr, ori), axis=1)
    merged_img_arr = np.concatenate((static_img_arr, dynamic_img_arr, reconstructed_img_arr, our_img_arr), axis=1)
    cv2.imwrite('merged/{}.png'.format(count), merged_img_arr)
    count += 1


