#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import cv2
import time
import copy
import numpy as np

#===============my modules====================
from my_libs import My_Obj_Detector
from my_libs import Patcher as pat
from my_libs import Kinect_class as Kin









#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
CAE_Name = 'CAE_Network.pt'			#Netwwork Name
Predictor = 0


Control = 2

#======plot configs======
fig = plt.figure()
axes1 = fig.add_subplot(2, 2, 1)
axes2 = fig.add_subplot(2, 2, 2, projection= '3d')
axes3 = fig.add_subplot(2, 2, 3)
axes4 = fig.add_subplot(2, 2, 4)






#===============================================================
#------------------------ Kinect -------------------------------
#===============================================================
Kinect = Kin.Kinect_object()

#Wait for kinect to be ready
print 'Waiting for Kinect..'
while not np.any(Kinect.imgRGB):
	continue
time.sleep(1)





#===============================================================
#------------------------ Detector -----------------------------
#===============================================================
Detector = My_Obj_Detector.Detector()
#------------------
# Detector Configs:
#------------------
Detector.ConvolutionalAutoEncoder = CAE_Name
Detector.use_Predictor = Predictor
#Detector.Intrins = []
#Detector.Extrins = []
#Detector.Workspace[2] = [-15, 700]

Detector.initialize()




#===============================================================
#----------------------- Test_loop -----------------------------
#===============================================================
counter = 0
while(1):
	counter += 1
	#Fetch image--------------------------------------------------
	tmp_rgb = copy.deepcopy(Kinect.imgRGB)		#copy
	tmp_depth = copy.deepcopy(Kinect.imgDepth)	#copy
	tmp_rgbd = Kinect.get_RGBD()






	#Estimate Objects in image------------------------------------
	start = time.time()
	position_predict, rotation_predic = Detector.Estimate_Objects_6DOF(RGBDimage = tmp_rgbd)
	end = time.time()


	if (not np.any(position_predict)) or (not np.any(rotation_predic)):
		#In case there are no predictions
		print 'Bad sample!'
	else:
		#project positions to image:
		projected = pat.from_3D_to_image(position_predict)
		#Transform to Baxter referencial:
		Baxter_pred = pat.from_CAM_to_WS(points_on_CAM = position_predict, CAM_on_WS = Detector.Extrins)







	#===============================================================
	#----------------------- Visualize -----------------------------
	#===============================================================
	#AXES 1-----------------------------
	axes1.set_title('RGB image')
	axes1.axis('off')
	axes1.imshow(tmp_rgb)

	
	if (np.any(position_predict)) and (np.any(rotation_predic)):
		for proj in projected:
			sct1 = axes1.scatter(proj[0],proj[1])


	#AXES 2-----------------------------
	axes2.set_title('Objects in Baxter\'s Ws')
	axes2.view_init(0, 180)
	axes2.set_xlabel('XX')
	axes2.set_ylabel('YY')
	axes2.set_zlabel('ZZ')
	axes2.set_xlim(Detector.Workspace[0][0]-100, Detector.Workspace[0][1]+100)
	axes2.set_ylim(Detector.Workspace[1][0]-100, Detector.Workspace[1][1]+100)
	axes2.set_zlim(Detector.Workspace[2][0]-100, Detector.Workspace[2][1]+100)

	if (np.any(position_predict)) and (np.any(rotation_predic)):
		color=['r','g','b']
		for preal, o in zip(Baxter_pred, rotation_predic):
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
	axes4.set_title('Voting Space')
	axes4.axis('off')
	axes4.imshow(Detector.Vote_Space)





	#Control Station
	plt.pause(0.1)
	print '-'*30+'\nDuration: ', end-start,'s'
	if counter % Control == Control -1:
		opt = raw_input('\nEnter to continue.\nq/Q to quit.\n::')

		if opt in ['q','Q']:
			break


	#Clear
	axes1.clear()
	axes2.clear()
	axes3.clear()
	axes4.clear()




















