#! /usr/bin/env python
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

#===============my modules====================
from my_libs import Patcher as pat



#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
run_times = 10
directory = 'imgs'

Intrinsic = []
Extrinsic = []
WorkSpace = [[-10000, 10000],	# x coordinate
			 [-10000, 10000], 	# y coordinate
			 [0, 1000]]			# z coordinate		[Back, Front]

fig, axes = plt.subplots(1,2)
fig.suptitle('Patcher Function Results', fontsize=14, fontweight='bold')


#===============================================================
#---------------------- Get images -----------------------------
#===============================================================
Names=[]
for catg in os.listdir(directory):
	for obj in os.listdir(directory+'/'+catg):
		for img in os.listdir(directory+'/'+catg+'/'+obj):
			if img.endswith('_crop.png'):
				Names.append(catg+'/'+obj+'/'+img[:-9])





#===============================================================
#---------------------- Patch imgs -----------------------------
#===============================================================
for n, img in enumerate(Names):
	#rgb
	bgr = cv2.imread(directory +'/'+ img +'_crop.png', -1)
	b,g,r = cv2.split(bgr)
	rgb = cv2.merge([r,g,b])
	#depth
	depth = cv2.imread(directory +'/'+ img +'_depthcrop.png', -1)
	depth = np.expand_dims(depth,axis=2)
	#rgbD
	RGBD = np.concatenate((rgb,depth),axis=2)



	#Extract Patches
	Patches, patch_centers, patch_real = pat.patch_sampling(RGBD = RGBD,
															metric = 50,
															spacing = 10,
															Intrins = Intrinsic,
															Extrins = Extrinsic,
															Workspace = WorkSpace)



	#===============================================================
	#---------------------- Visualize ------------------------------
	#===============================================================
	pat.Visualize_Patches(axes=axes ,
						  rgb_img=rgb, 
						  Patches=Patches, 
						  patch_centers=patch_centers)



	#only run first 3 images
	if n % run_times == run_times-1:
		print 'RUNTIMEs break'
		break







print('this is the end!')
#raw_input('press Enter to exit.')
cv2.destroyAllWindows()










