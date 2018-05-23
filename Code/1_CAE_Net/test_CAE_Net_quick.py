#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
import os
import cv2
import time
import random
import numpy as np
import torch

#===============my modules====================
from my_libs import Patcher as pat
from my_libs import CAE_net







#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
Name = 'OLD_CAE'					#Netwwork Name, first CAE fully trained
#Name = 'CAE_Network'				#Netwwork Name


dataset = 'real_objects_dataset'	#Dataset: real objects


#Visualize Results
img_num	= 10
show_patches = 4	#patches per image
show_time = 0.3		#time per patch
stats = []


fig, axes = plt.subplots(1,3)
fig.suptitle('CAE_Net Results', fontsize=14, fontweight='bold')

Inp = np.zeros([32,64,3])
Out = np.zeros([32,64,3])

#===============================================================
#------------------------ DATASET ------------------------------
#===============================================================
print 'Scanning Dataset:',dataset
chosen = []
if not os.path.exists(dataset):
	print "\nDataset doesn't exist..\n"
else:
	Objects = sorted(os.listdir(dataset))
	for oi, obj in enumerate(Objects):									#Objectos
		Obj_Instances = sorted(os.listdir(dataset+'/'+obj))
		for ii, inst in enumerate(Obj_Instances):						#instancias
			Images = sorted(os.listdir(dataset+'/'+obj+'/'+inst))
			for imi, img in enumerate(Images):							#ficheiros
				if img.endswith('_crop.png'):							#imagem rgb
					chosen.append(dataset +'/'+ obj +'/'+ inst +'/'+ img[:-9])

random.shuffle(chosen)





#===============================================================
#----------------------- CAE Network ---------------------------
#===============================================================
CAE = CAE_net.CAE()
CAE.initialize(Name + '.pt')

loss_fn = torch.nn.MSELoss(size_average=False)		#Loss function: Mean Squared Error



#===============================================================
#---------------------- Visualize ------------------------------
#===============================================================
for si, sample in enumerate(chosen):
	#get from dataset a rgb sample
	bgr = cv2.imread(sample + '_crop.png',-1)
	#convert bgr to rgb
	b,g,r = cv2.split(bgr)       # get b,g,r
	rgb = cv2.merge([r,g,b])     # switch it to rgb
	
	#get from dataset a depth sample
	depth = cv2.imread(sample + '_depthcrop.png',-1)
	depth = np.expand_dims(depth,axis=2)

	#compose rgbd image
	RGBD = np.concatenate((rgb,depth),axis=2)
	

	#generate Batch of input patches
	Patches, patch_centers, patch_real = pat.patch_sampling(RGBD = RGBD,
															metric = 50,
															spacing = 10,
															Intrins = [],
															Extrins = [],
															Workspace = [])
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
	
	#stats.append(loss.data[0])
	#axes[1,1].clear()
	#axes[1,1].set_title('loss per image:')
	#axes[1,1].plot(np.arange(len(stats)),stats)
	#plt.pause(0.1)


	inputs = CAE.unwrap_input(inputs[0:show_patches])
	outputs = CAE.unwrap_input(outputs[0:show_patches])
	features = CAE.unwrap_features(features)
	
	for img_in, img_out, feat in zip(inputs, outputs, features): 


		Inp[:,:32,:3] = img_in[:,:,:3]
		Inp[: ,32:64,0] = img_in[:,:,3]
		Inp[: ,32:64,1] = img_in[:,:,3]
		Inp[: ,32:64,2] = img_in[:,:,3]

		Out[:,:32,:3] = img_out[:,:,:3]
		Out[: ,32:64,0] = img_out[:,:,3]
		Out[: ,32:64,1] = img_out[:,:,3]
		Out[: ,32:64,2] = img_out[:,:,3]

		#Before encryption
		before_rgb = img_in[:,:,0:3]
		before_depth = img_in[:,:,3]
		axes[0].set_title('Before encyrption RGB-D:')
		axes[0].axis('off')
		axes[0].imshow(Inp)


		#Encrypted
		encrypted = feat.reshape(16,16)
		axes[1].set_title('Encrypted (16x16 representation):')
		axes[1].axis('off')
		axes[1].imshow(encrypted)

		#After Decryption
		after_rgb = img_out[:,:,0:3]
		after_depth = img_out[:,:,3]
		axes[2].set_title('After encyrption RGB-D:')
		axes[2].axis('off')
		axes[2].imshow(Out)

		plt.pause(0.2)
		raw_input('')

	if si%img_num == img_num-1:
		print '='*30
		print 'Enter: run', img_num,'more images.\nq: to quit'
		opt=raw_input('::')
		if opt in ['q','Q']:
			break

print 'Finished!'











