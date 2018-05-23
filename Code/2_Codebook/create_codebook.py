#! /usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
import cv2
import math
import time
import tables
import numpy as np

#===============my modules====================
from my_libs import Patcher as pat
from my_libs import CAE_net




#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
run_only = []
#codebook file:
codebook='codebook.h5'

#dataset: imagens rgb-d de objetos sinteticos, c/GroundTruth
dataset = "synth_dataset"

#Convolotional Auto Encoder
CAE_name = 'CAE_Network.pt'


print_stats = 1
speed_run = True	#Run only first 10*print_stats images for each object


view = True
#======plot configs======
fig = plt.figure()
ax2d = fig.add_subplot(1, 2, 1)
ax3d = fig.add_subplot(1, 2, 2, projection= '3d')

ax3d.set_xlabel('x axis')
ax3d.set_ylabel('y axis')
ax3d.set_zlabel('z axis')
#==========================


#===============================================================
#------------------- Codebook Format ---------------------------
#===============================================================
class synth_patch(tables.IsDescription):
    features = tables.Float32Col(pos=1, shape=[1,256])       
    Translation = tables.Float32Col(pos=2, shape=[1,3])
    euler_angles = tables.Float32Col(pos=3, shape=[1,3])
    obj_ids = tables.Int32Col(pos=4)



#===============================================================
#----------------- Get dataset sample --------------------------
#===============================================================
def get_sample(location, sample):
	try:
		#Fetch image--------------------------------------------------
		tmp_bgr = cv2.imread(location + 'rgb/' + sample,-1)
		#convert bgr to rgb
		tmp_rgb = cv2.cvtColor(tmp_bgr, cv2.COLOR_BGR2RGB)

		#Depth
		tmp_depth = cv2.imread(location + 'depth/' + sample,-1)
		tmp_depth = np.expand_dims(tmp_depth,axis=2)
		#rgbD
		tmp_rgbd = np.concatenate([tmp_rgb, tmp_depth], axis=2)


		#Fetch ground truth--------------------------------------------
		with open(location + 'gt.yml','r') as gt:
			all_lines = gt.readlines()
			chosen = str(int(sample[:4])) + ':'
			tmp_rot=np.zeros(9)	

			for li, line in enumerate(all_lines):
				if chosen in line.split():
					#Orientation
					words = all_lines[li+1].split()
					tmp_rot[0] = float(words[2][1:-1])
					for i in range(3,11):
						tmp_rot[i-2] = float(words[i][:-1])
					tmp_rot = tmp_rot.reshape(3,3)

					#Translation
					words = all_lines[li+2].split()
					tmp_trans = np.array(map(float,[words[1][1:-1],words[2][:-1],words[3][:-1]]))


		return tmp_rgbd, tmp_trans, tmp_rot
	except:
		return [],[]



def rotation_to_euler(R) :
	#Check if R is rotation matrix
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	assert(n < 1e-6)

	#Calculate euler angles
	sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	singular = sy < 1e-6

	if  not singular :
		x = math.atan2(R[2,1] , R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else :
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0

	return np.array([x, y, z])



#===============================================================
#----------------------- CAE Network ---------------------------
#===============================================================
CAE = CAE_net.CAE()
CAE.initialize(CAE_name)



#===============================================================
#-------------------- Create codebook --------------------------
#===============================================================
#O Codebook vai ser gerado uma imagem de cada vez....para nao haver problemas de memoria n' shit..
for directory in sorted(os.listdir(dataset)):
	#----make specific codebook----
	if np.any(run_only):
		if directory not in run_only:
			continue
	#------------------------------
	location = dataset + '/' + directory + '/'
	ax3d.view_init(0, 0)
	since = time.time()
	for i, img in enumerate(sorted(os.listdir(location +'rgb'))):

		#get sample
		rgbd, Translation, Rotation = get_sample(location, img)
		if (rgbd == []) or (Translation == []) or (Rotation == []):
			print 'error for:', location+'/'+img
			continue


		#Extract Patches
		Patches, patch_centers, patch_real = pat.patch_sampling(RGBD = rgbd,
																metric = 50,
																spacing = 10)
		if not np.any(Patches):
			#In case there are no patches extracted
			print 'No Patches:', location+'/'+img
			continue


		#wrap input
		inputs = CAE.wrap_input(Patches)
		#Encrypt Patches
		Features = CAE.encode(inputs)
		#unwrap features
		Features = CAE.unwrap_features(Features)






		#===============================================================
		#-----------------------Predictions-----------------------------
		#===============================================================
		#Extract relative translations
		relative_T = np.subtract(Translation, patch_real)

		#Extract relative Orientation
		relative_O = rotation_to_euler(Rotation)




		#===============================================================
		#---------------------Save in codebook--------------------------
		#===============================================================
		with tables.open_file(codebook, mode='a') as h5file:
			try: cbk = h5file.create_table(h5file.root, 'DATA', synth_patch)
			except: cbk=h5file.root.DATA

			for f,t in zip(Features,relative_T):
				cbk.row['features'] = f
				cbk.row['Translation'] = t
				cbk.row['euler_angles'] = relative_O
				cbk.row['obj_ids'] = int(directory)
				cbk.row.append()
			cbk.flush



		#Control Sation
		if i%print_stats == print_stats-1:
			now = time.time()
			print '[{:4}, {:28}, {:4f}s]'.format(i+1, location+img, now-since)
			since = now

			#Visualize results
			if view:
				#image patch centers
				ax2d.imshow(rgbd[:,:,0:3])	#Why funny colors? no colorspace convertion works..
				s_grid1 = ax2d.scatter(patch_centers[:,1], patch_centers[:,0], c='b', s=1)

				#3D patch center and Camera position
				ax3d.scatter(Translation[0],Translation[1],Translation[2] , c='r', marker='x', s=100)		#object center
				ax3d.scatter(patch_real[:,0], patch_real[:,1], patch_real[:,2], c='b', marker='o', s=5)
				print 'Relative translations:'
				for rt, pc in zip(relative_T, patch_real):
					xx = [pc[0], pc[0]+rt[0]]
					yy = [pc[1], pc[1]+rt[1]]
					zz = [pc[2], pc[2]+rt[2]]
					h = ax3d.plot_wireframe(xx,yy,zz, color='g', linewidth=0.1)
					print '   ', rt
					#plt.pause(0.1)
					#h.remove()

				plt.pause(0.1)
				raw_input('')
				ax2d.clear()
				ax3d.clear()





		if speed_run and (i%(10*print_stats) == (10*print_stats-1)):
			break

	#break	#run only first object








