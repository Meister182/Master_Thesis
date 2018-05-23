#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
import os
import cv2
import time
import tables
import random
import numpy as np
import torch

#===============my modules====================
from my_libs import Patcher as pat
from my_libs import CAE_net



#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
#Name = 'OLD_CAE'				#Netwwork Name, first CAE fully trained
Name = 'CAE_Network'				#Netwwork Name
dataset = 'real_objects_dataset'	#Dataset: real objects

epochs = 5	#Train + Validation stage

print_stats = 50
speed_run = True	#Run only 10 prints per stage

#Visualize Results
view = False
#view = True
rerun = False	#Rerun Network on the inputs to see encryption
#rerun = True
show_patches = 2
show_time = 0.3

fig, axes = plt.subplots(2,3)
fig.suptitle('CAE_Net Results', fontsize=14, fontweight='bold')



#===============================================================
#---------------------- Regist Sats ----------------------------
#===============================================================
class stats(tables.IsDescription):
	sample = tables.StringCol(100, pos=0)
	epochs = tables.Int32Col(pos=1)
	loss = tables.Float32Col(pos=2)



#===============================================================
#------------------------ DATASET ---------------------_--------
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

lines = len(chosen)
random.shuffle(chosen)

DS={}
DS['train'] = chosen[0:int(0.8*lines)]		#80%
DS['valid'] = chosen[int(0.8*lines):]		#20%
#DS['test'] = chosen[int(0.8*lines):]



#===============================================================
#----------------------- CAE Network ---------------------------
#===============================================================
CAE = CAE_net.CAE()
CAE.initialize(Name + '.pt')

best_run = 999999.0
stage_loss = 0.0
runin_loss = 0.0

loss_fn = torch.nn.MSELoss(size_average=False)			#Loss function: Mean Squared Error
optimizer = torch.optim.Adam(CAE.parameters(), lr=1e-4)	#Optimzer: Adam



#===============================================================
#---------------------- TRAIN LOOP -----------------------------
#===============================================================
for epc in xrange(epochs):
	for stage in ['train', 'valid']:
		#reshuffle stage dataset
		random.shuffle(DS[stage])

		#initialize counters
		since = time.time()
		stage_loss = 0
		runin_loss = 0.0

		for i, sample in enumerate(DS[stage]):

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
			outputs = CAE(inputs)
			#calculate loss
			loss = loss_fn(outputs,inputs)
			stage_loss += loss.data[0]
			runin_loss += loss.data[0]


			#Backpropagation
			if stage == 'train':
				optimizer.zero_grad()	#Zero any previous gradients
				loss.backward()			#compute new grad
				optimizer.step()		#update weights in respect to the new grad, and Adam method in this case



			#save statistics
			with tables.open_file(Name+'.h5','a') as h5file:
				try:	regist = h5file.create_table(h5file.root, stage, stats)
				except:
					if stage == 'train' :	regist = h5file.root.train
					elif stage == 'valid':	regist = h5file.root.valid

				regist.row['sample'] = sample
				regist.row['epochs'] = epc
				regist.row['loss'] = loss.data[0]
				regist.row.append()
				regist.flush					




			#CONTROL STATION
			if i%print_stats == print_stats-1:
				now = time.time() 
				duration = now-since
				duration_img = duration/print_stats
				since = now

				runin_loss = runin_loss/print_stats
				print "[{:5} ,{:2} ,{:3}] loss:{:8.3f} | duration: {:.4f} s /img: {:.4f} s".format(stage[:5], epc, (i+1)/print_stats, runin_loss, duration, duration_img)
				runin_loss = 0

				#Visualze
				if view:
					CAE.Visualize_Results(axes = axes, 
										  inputs = inputs[0:show_patches], 
										  Features = [], 
										  outputs = outputs[0:show_patches], 
										  rerun = rerun, 
										  show_time = show_time)
				
				#run only first 10 stats print
				if (i%(10*print_stats) == (10*print_stats)-1) and speed_run:
					break

		#===============================================================
		#---------------------- Validation -----------------------------
		#===============================================================
		stage_loss = stage_loss / (i+1)
		if stage == 'valid':			
			if stage_loss <= best_run:
				CAE.save(net_name = Name+'.pt')
				best_run = stage_loss
				print 'CAE_Net saved!  Best_run:', best_run
			print '-'*50

			#break	#only 1st element of DS
		#break	#only train stage
	#break	#only 1st epoch

