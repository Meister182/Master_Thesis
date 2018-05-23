#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
import os
import cv2
import time
import random
import numpy as np
import torch
import copy

#===============my modules====================
from my_libs import Patcher as pat
from my_libs import CAE_net
from my_libs import Kinect_class as Kin







#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
Name = 'OLD_CAE'					#Netwwork Name, first CAE fully trained
#Name = 'CAE_Network'				#Netwwork Name


#Visualize Results
img_num	= 3
show_patches = 5	#patches per image
show_time = 0.3		#time per patch
stats = []

fig, axes = plt.subplots(2,3)
fig.suptitle('CAE_Net Results', fontsize=14, fontweight='bold')



#on Baxter's head:  from calc_transform.py
Extrins = np.array([[ -0.0133  ,-0.5089  ,0.8607  ,80.0000  ],
					 [ -0.9997  ,0.0262   ,0.0000  ,0.0000   ],
					 [ -0.0225  ,-0.8604  ,-0.5090 ,950.0000 ],
					 [ 0.0000   ,0.0000   ,0.0000  ,1.0000  ]])

#Baxter's Workspace
Workspace = [[950, 1750],         # x coordinate      [Back, Front]
			  [-550, 550 ],        # y coordinate      [Right, Left]
			  [-15, 700]]          # z coordinate      [Down, Up]



#===============================================================
#----------------------- CAE Network ---------------------------
#===============================================================
CAE = CAE_net.CAE()
CAE.initialize(Name + '.pt')

loss_fn = torch.nn.MSELoss(size_average=False)		#Loss function: Mean Squared Error


#===============================================================
#------------------------ Kinect ------------------------------
#===============================================================
Kinect = Kin.Kinect_object()
print 'Waiting for kinect..'
#Wait for kinect to be ready
while not np.any(Kinect.imgRGB):
	continue
time.sleep(1)
print 'ready!'


#===============================================================
#---------------------- Visualize ------------------------------
#===============================================================
cicle = 0
while(1):
	cicle += 1
	#Fetch image--------------------------------------------------
	tmp_rgb = copy.deepcopy(Kinect.imgRGB)		#copy
	tmp_depth = copy.deepcopy(Kinect.imgDepth)	#copy
	tmp_depth = np.expand_dims(tmp_depth,axis=2)
	#rgbD
	RGBD = np.concatenate((tmp_rgb, tmp_depth),axis=2)
	
	

	#generate Batch of input patches
	Patches, patch_centers, patch_real = pat.patch_sampling(RGBD = RGBD,
															metric = 50,
															spacing = 10,
															Intrins = [],
															Extrins = Extrins,
															Workspace = Workspace)
	if not np.any(Patches):
		#In case there are no patches extracted
		print 'No Patches:', i, sample,
		continue


	#wrap input
	inputs = CAE.wrap_input(Patches)
	#Feed CAE_Net
	features = CAE.encode(inputs)
	outputs = CAE.decode(features)
	#calculate loss
	loss = loss_fn(outputs,inputs)
	
	stats.append(loss.data[0])
	axes[1,1].clear()
	axes[1,1].set_title('loss per image:')
	axes[1,1].plot(np.arange(len(stats)),stats)
	plt.pause(0.1)

	CAE.Visualize_Results(axes = axes, 
						  axis11 = 'on',
						  inputs = inputs[0:show_patches], 
						  Features = features, 
						  outputs = outputs[0:show_patches], 
						  rerun = False, 
						  show_time = show_time)

	if cicle % img_num == img_num-1:
		print '='*30
		print 'Enter: run', img_num,'more images.\nq: to quit'
		opt=raw_input('::')
		if opt in ['q','Q']:
			break

print 'Finished!'











