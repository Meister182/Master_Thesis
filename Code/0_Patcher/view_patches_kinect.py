#! /usr/bin/env python
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

#===============my modules====================
from my_libs import Patcher as pat
from my_libs import Kinect_class as Kin



#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
run_times = 10
directory = 'imgs'

Intrins = np.array([[572.4114  ,0.         ,325.2611],
					[0.        ,573.2611   ,242.04899],
					[0.        ,0.         ,1.       ]])

Extrins = np.array([[ -0.0133  ,-0.5089  ,0.8607  ,80.0000  ],
					[ -0.9997  ,0.0262   ,0.0000  ,0.0000   ],
					[ -0.0225  ,-0.8604  ,-0.5090 ,950.0000 ],
					[ 0.0000   ,0.0000   ,0.0000  ,1.0000  ]])

Workspace = [[950 , 1750],         # x coordinate      [Back, Front]
			 [-550, 550 ],        # y coordinate      [Right, Left]
			 [-15 , 700 ]] 


fig, axes = plt.subplots(1,2)
fig.suptitle('Patcher Function Results', fontsize=14, fontweight='bold')


#===============================================================
#------------------------ Kinect ------------------------------
#===============================================================
Kinect = Kin.Kinect_object()
#wait for kinect to be ready
print 'Waiting for kinect..'
while np.shape(Kinect.imgRGB)[0]<10:
	continue

raw_input('go')

#===============================================================
#---------------------- Patch imgs -----------------------------
#===============================================================
for i in xrange(5):


	#rgb
	rgb = Kinect.imgRGB
	#depth
	depth = Kinect.imgDepth
	depth = np.expand_dims(depth,axis=2)
	#rgbD
	RGBD = np.concatenate((rgb,depth),axis=2)




	#Extract Patches
	Patches, patch_centers, patch_real = pat.patch_sampling(RGBD = RGBD,
															metric = 50,
															spacing = 10,
															Intrins = Intrins,
															Extrins = Extrins,
															Workspace = Workspace)



	#===============================================================
	#---------------------- Visualize ------------------------------
	#===============================================================
	axes[0].imshow(rgb)
	sr = axes[0].scatter(patch_centers[:,1], patch_centers[:,0] , color='red', s=0.5)
	plt.pause(0.1)
	print np.shape(Patches)
	raw_input('')
	colage = np.zeros([32,64,3])
	for p, patch in enumerate(Patches):
		#sb = axes[0].scatter(patch_centers[p,1], patch_centers[p,0], color='blue', s=10)
		#print depth[patch_centers[p,0], patch_centers[p,1]]

		colage[:,:32,:3] = patch[:,:,:3]
		colage[: ,32:64,0] = patch[:,:,3]
		colage[: ,32:64,1] = patch[:,:,3]
		colage[: ,32:64,2] = patch[:,:,3]

		
		axes[1].imshow(colage)
		plt.pause(0.1)

		opt = raw_input('')
		if opt in ['s']:
			break
		#sb.remove()
		axes[1].clear()
	axes[0].clear()
	axes[1].clear()






print('this is the end!')
#raw_input('press Enter to exit.')
cv2.destroyAllWindows()










