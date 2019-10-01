import glob
import numpy as np
import os

file_list = glob.glob(pathname="Label2D/*.mat")
rand_num = np.random.rand(len(file_list))

#os.system("mv " + "Label2D/val_backup/* " + "Label2D/")
#os.system("mv " + "Label2D/train_backup/* " + "Label2D/")
#os.system("mv " + "JPEGCrop/val/* " + "JPEGCrop/")
#os.system("mv " + "JPEGCrop/train/* " + "JPEGCrop/")

for i in range(len(file_list)):
    if rand_num[i] <= 0.25:
        os.system("mv " + file_list[i] + " Label2D/val_backup/")
        #os.system("mv " + "JPEGCrop/" + file_list[i][8:16] + "*.jpg " + "JPEGCrop/val/")
    else:
        os.system("mv " + file_list[i] + " Label2D/train_backup/")
        #os.system("mv " + "JPEGCrop/" + file_list[i][8:16] + "*.jpg " + "JPEGCrop/train/")
