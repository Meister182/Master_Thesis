#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import cv2
import copy
import time
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


print_stats = 5

save_counter = 0
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
print 'Waiting for kinect..'
while not np.any(Kinect.imgRGB):
	continue


#===============================================================
#------------------------ Detector -----------------------------
#===============================================================
Detector = Obj_Detector.Detector()
#------------------
# Detector Configs:
#------------------
Detector.ConvolutionalAutoEncoder = CAE_Name
Detector.CodeBookFile = Codebook
Detector.patch_spacing = 10
Detector.interp_smoothing = 0

#Detector.Intrins = []
#Detector.Extrins = []
#Detector.Workspace = []
Detector.Workspace = [[950, 1750],         # x coordinate      [Back, Front]
				  	  [-550, 550 ],        # y coordinate      [Right, Left]
				  	  [-15, 700]]          # z coordinate      [Down, Up]

Detector.initialize()

#===============================================================
#----------------------- Test_loop -----------------------------
#===============================================================
while(1):
	#Fetch image--------------------------------------------------
	tmp_rgb = copy.deepcopy(Kinect.imgRGB)		#copy
	#tmp_depth = copy.deepcopy(Kinect.imgDepth)	#copy
	tmp_rgbd = Kinect.get_RGBD()


	start = time.time()
	#===============================================================
	#--------------- Estimate Objects in image ---------------------
	#===============================================================
	#1-Extract features--------------------------------------------------
	features, centers , real_centers = Detector.Extract_Features(RGBDimage = tmp_rgbd)
	if (not np.any(features)) or (not np.any(centers)) or (not np.any(real_centers)):
		#In case there are no patches extracted
		print 'no features!'
		continue

	#2-KNN search--------------------------------------------------
	results, distances = Detector.Knn_search(features)
	if not np.any(results):
		print 'no neighboors!'
		continue

	#3-Vote--------------------------------------------------
	Detector.Cast_Votes(results, distances, real_centers)

	Detector.Votes_filter_1_counter(np.shape(tmp_rgbd))
	vote_space0 = np.array(Detector.vote_cells_space)

	Detector.Votes_filter_1_acumulator()
	vote_space1 = np.array(Detector.vote_cells_space)

	local_max_x, local_max_y = Detector.Votes_filter_2()

	position_predictions, rotation_predictions = Detector.Votes_filter_3(local_max_x, local_max_y)

	if not np.any(position_predictions):
		print 'No predictions..'
		continue

	#Data------------
	#project positions to image:
	projected = pat.from_3D_to_image(position_predictions)

	#Transform to Baxter referencial:
	Baxter_real = pat.from_CAM_to_WS(points_on_CAM = real_centers, CAM_on_WS = Detector.Extrins)
	Baxter_pred = pat.from_CAM_to_WS(points_on_CAM = position_predictions, CAM_on_WS = Detector.Extrins)


	end = time.time()
	#===============================================================
	#----------------------- Visualize -----------------------------
	#===============================================================
	#AXES 1-----------------------------
	#axes1.imshow(tmp_rgbd[:,:,0:3])
	axes1.set_title('RGB image')
	axes1.axis('off')
	axes1.imshow(tmp_rgb)
	sct1_1 = axes1.scatter(centers[:,1], centers[:,0], s=1, c='b')
	for proj in projected:
		sct1_2 = axes1.scatter(proj[0],proj[1])

	#AXES 2-----------------------------
	axes2.set_title('3D Patch centers in Baxter\'s Ws')
	axes2.view_init(0, 180)
	axes2.set_xlabel('XX')
	axes2.set_ylabel('YY')
	axes2.set_zlabel('ZZ')
	axes2.set_xlim(Detector.Workspace[0][0]-10, Detector.Workspace[0][1]+10)
	axes2.set_ylim(Detector.Workspace[1][0]-10, Detector.Workspace[1][1]+10)
	axes2.set_zlim(Detector.Workspace[2][0]-10, Detector.Workspace[2][1]+10)

	sct2_1 = axes2.scatter(Baxter_real[:,0], Baxter_real[:,1], Baxter_real[:,2], s=5)


	color=['r','g','b']
	for preal, o in zip(Baxter_pred, rotation_predictions):
		Rot_mat = Detector.euler_to_rotation(o)
		#Rot_mat = np.dot(Detector.Extrins[0:3,0:3], Rot_mat)
		sct2_2 = axes2.scatter(preal[0], preal[1], preal[2], s=50)
		for iw, r in enumerate(Rot_mat):
			xx = [preal[0], preal[0]+(r[0]*100)]
			yy = [preal[1], preal[1]+(r[1]*100)]
			zz = [preal[2], preal[2]+(r[2]*100)]
			h = axes2.plot_wireframe(xx,yy,zz, color=color[iw])



	#AXES 3-----------------------------
	axes3.set_title('Raw Voting Space')
	axes3.axis('off')
	axes3.imshow(vote_space0)

	#AXES 4-----------------------------
	axes4.set_title('Interpolated Voting Space')
	axes4.axis('off')
	axes4.imshow(vote_space1)
	sct4 = axes4.scatter(local_max_y, local_max_x, s=5, c='r', marker = 'x')







	
	plt.pause(0.1)
	print '-'*30+'\nDuration: ', end-start,'s'
	opt = raw_input('\nEnter to continue.\ns/S to save figure\nq/Q to quit.\n::')

	if opt in ['q','Q']:
		break
	elif opt in ['s','S']:
		fig.savefig('0'+str(save_counter)+'.png', format = 'png')
		save_counter += 1

	sct1_1.remove()
	sct1_2.remove()
	sct2_1.remove()
	sct2_2.remove()
	sct4.remove()
	axes1.clear()
	axes2.clear()
	axes3.clear()
	axes4.clear()

Detector.closeCodeBook()
