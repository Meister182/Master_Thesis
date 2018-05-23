#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import cv2
import time
import copy

import random
import numpy as np
import torch

#===============my modules====================
from my_libs import Obj_Detector
from my_libs import Patcher as pat
from my_libs import Kinect_class as Kin


#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
CAE_Name = 'CAE_Network.pt'			#Netwwork Name

Codebook = 'codebook_350Mb.h5'		#Codebook
#Codebook = 'codebook_500Mb.h5'		#Codebook
#Codebook = 'codebook_1GB.h5'		#Codebook


#======plot configs======
fig = plt.figure()
axes1 = fig.add_subplot(2, 2, 1)
axes2 = fig.add_subplot(2, 2, 2, projection= '3d')
axes3 = fig.add_subplot(2, 2, 3)
axes4 = fig.add_subplot(2, 2, 4)

#===============================================================
#------------------------ Kinect ------------------------------
#===============================================================
Kinect = Kin.Kinect_object()

#Wait for kinect to be ready
while not np.any(Kinect.imgRGB):
	continue
time.sleep(1)


#===============================================================
#------------------------ Detector -----------------------------
#===============================================================
Detector = Obj_Detector.Detector()
#------------------
# Detector Configs:
#------------------
Detector.ConvolutionalAutoEncoder = CAE_Name
Detector.CodeBookFile = Codebook
#Detector.Intrins = []
#Detector.Extrins = []
#Detector.Workspace = []

Detector.initialize()



#===============================================================
#----------------------- Test_loop -----------------------------
#===============================================================
while(1):
	#Fetch image--------------------------------------------------
	tmp_rgb = copy.deepcopy(Kinect.imgRGB)		#copy
	tmp_depth = copy.deepcopy(Kinect.imgDepth)	#copy
	tmp_rgbd = Kinect.get_RGBD()


	#Estimate Objects in image------------------------------------
	start = time.time()
	position_predictions, rotation_predictions = Detector.Estimate_Objects_6DOF(RGBDimage = tmp_rgbd)
	if (not np.any(position_predictions)) or (not np.any(rotation_predictions)):
		#In case there are no predictions
		print 'Bad sample!'
		continue
	end = time.time()
	


	#project positions to image:
	projected = pat.from_3D_to_image(position_predictions)
	#Transform to Baxter referencial:
	Baxter_pred = pat.from_CAM_to_WS(points_on_CAM = position_predictions, CAM_on_WS = Detector.Extrins)






	#===============================================================
	#----------------------- Visualize -----------------------------
	#===============================================================
	#AXES 1-----------------------------
	axes1.set_title('RGB image')
	axes1.axis('off')
	axes1.imshow(tmp_rgb)
	for proj in projected:
		sct1 = axes1.scatter(proj[0],proj[1])



	#AXES 2-----------------------------
	axes2.set_title('Objects in Baxter\'s Ws')
	axes2.view_init(0, 180)
	axes2.set_xlabel('XX')
	axes2.set_ylabel('YY')
	axes2.set_zlabel('ZZ')
	axes2.set_xlim(Detector.Workspace[0][0]-10, Detector.Workspace[0][1]+10)
	axes2.set_ylim(Detector.Workspace[1][0]-10, Detector.Workspace[1][1]+10)
	axes2.set_zlim(Detector.Workspace[2][0]-10, Detector.Workspace[2][1]+10)



	color=['r','g','b']
	for preal, o in zip(Baxter_pred, rotation_predictions):
		Rot_mat = Detector.euler_to_rotation(o)
		#Rot_mat = np.dot(Detector.Extrins[0:3,0:3], Rot_mat)
		sct2 = axes2.scatter(preal[0], preal[1], preal[2], s=50)
		for iw, r in enumerate(Rot_mat):
			xx = [preal[0], preal[0]+(r[0]*100)]
			yy = [preal[1], preal[1]+(r[1]*100)]
			zz = [preal[2], preal[2]+(r[2]*100)]
			h = axes2.plot_wireframe(xx,yy,zz, color=color[iw])




	#AXES 3-----------------------------
	axes3.set_title('Depth image')
	axes3.axis('off')
	axes3.imshow(tmp_depth)


	#AXES 4-----------------------------
	axes4.set_title('Interpolated Voting Space')
	axes4.axis('off')
	axes4.imshow(Detector.vote_cells_space)


	
	plt.pause(0.1)
	print '-'*30+'\nDuration: ', end-start,'s'
	opt = raw_input('\nEnter to continue.\ns/S to save figure\nq/Q to quit.\n::')

	if opt in ['q','Q']:
		break
	elif opt in ['s','S']:
		fig.savefig('0'+str(save_counter)+'.png', format = 'png')
		save_counter += 1
	sct1.remove()
	sct2.remove()
	axes1.clear()
	axes2.clear()
	axes3.clear()
	axes4.clear()

Detector.closeCodeBook()
