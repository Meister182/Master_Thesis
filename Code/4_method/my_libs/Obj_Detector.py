#! /usr/bin/env python
#===============IMPORTS======================
import tables
import pyflann
import numpy as np
from scipy import interpolate
import scipy.ndimage.filters as filters
from sklearn.cluster import MeanShift, estimate_bandwidth

#===============my modules====================
from my_libs import Patcher as pat
from my_libs import CAE_net as conv



class Detector(object):
	#===============================================================
	#------------------------ Configs ------------------------------
	#===============================================================
	def __init__(self):
		#Depth Camera: Kinect
		self.Intrins = np.array([[572.4114  ,0.         ,325.2611],
								 [0.        ,573.2611   ,242.04899],
								 [0.        ,0.         ,1.       ]])
		
		#on Baxter's head:  from calc_transform.py
		self.Extrins = np.array([[ -0.0133  ,-0.5089  ,0.8607  ,80.0000  ],
								 [ -0.9997  ,0.0262   ,0.0000  ,0.0000   ],
								 [ -0.0225  ,-0.8604  ,-0.5090 ,950.0000 ],
								 [ 0.0000   ,0.0000   ,0.0000  ,1.0000  ]])

		#Baxter's Workspace
		self.Workspace = [[950, 1750],         # x coordinate      [Back, Front]
						  [-550, 550 ],        # y coordinate      [Right, Left]
						  [-15, 700]]          # z coordinate      [Down, Up]

		#Feature extracter
		self.ConvolutionalAutoEncoder = 'CAE_Network.pt'

		#CodeBook
		self.CodeBookFile = 'rapid_test_codebook.h5'
		self.CodeBook_Read_size = 2160000

		#Patches caracteristics:
		self.patch_metric_size = 50             #5cm
		self.patch_spacing = 5                 #8 pixels between patch centers

		#K-Nearest Neighboors:
		self.Neighboors_K = 5
		self.Neighboors_distance_threshold = 50
		self.flann = pyflann.FLANN()

		#Voting parameters:
		#filter 1
		self.cell_size = 5              #5x5 pixels
		self.voting_threshold = 3       #ignore cells with 5 or less votes
		self.weight_trust = 0.8      	#

		#filter 2
		self.interp_smoothing = 0
		self.localmax_window = 15   #5x5 cells

		#filter 3
		self.Neighboor_Cells = 2            #check Votes in the Neighboors cells
		self.Translation_Kernel = 100       #25                 #2.5 cm
		self.Orientation_Kernel = 2*np.pi       #((2*3.1416)/360)*7 #7 degrees


	#=======================================================================================================
	#-----------------------------------methods:------------------------------------------------------------
	#=======================================================================================================
	#auxiliares
	def initialize(self):
		self.initialize_CAE()
		self.initialize_CodeBook()

	def initialize_CAE(self):
		self.CAE_Net = conv.CAE()
		self.CAE_Net.initialize(self.ConvolutionalAutoEncoder)

	def initialize_CodeBook(self):
		self.CodeBook = tables.open_file(self.CodeBookFile,'r')
		self.CodeBookData = self.CodeBook.root.DATA
		self.CodeBookLength = self.CodeBookData.shape[0]
		self.CodeBookParts = int(np.ceil(self.CodeBookLength/self.CodeBook_Read_size))

	def closeCodeBook(self):
		self.CodeBook.close()

	def round_int(self, x):
		try: r = int(x)
		except: r = int(0)
		return(r)

	def rotation_to_euler(self, R) :
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

	def euler_to_rotation(self, euler_angles):
		Rx = np.eye(3)
		Ry = np.eye(3)
		Rz = np.eye(3)
		#Setting XX rotation
		Rx[1,1] = np.cos(euler_angles[0])
		Rx[1,2] = -np.sin(euler_angles[0])
		Rx[2,1] = np.sin(euler_angles[0])
		Rx[2,2] = np.cos(euler_angles[0])
		#Setting YY rotation
		Ry[0,0] = np.cos(euler_angles[1])
		Ry[0,2] = np.sin(euler_angles[1])
		Ry[2,0] = -np.sin(euler_angles[1])
		Ry[2,2] = np.cos(euler_angles[1])
		#Setting ZZ rotation
		Rz[0,0] = np.cos(euler_angles[2])
		Rz[0,1] = -np.sin(euler_angles[2])
		Rz[1,0] = np.sin(euler_angles[2])
		Rz[1,1] = np.cos(euler_angles[2])

		#Rotation Matrix:
		Rot = np.dot(Ry,Rx)
		Rot = np.dot(Rz,Rot)

		return(Rot)

	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
					#mmm   mmmmmmmmmmmmm mmmmmm   mmm mmmmmmm  mmmm  mmmmm 
					#   "m #        #    #      m"   "   #    m"  "m #   "#
					#    # #mmmmm   #    #mmmmm #        #    #    # #mmmm"  #
					#    # #        #    #      #        #    #    # #   "m
					#mmm"  #mmmmm   #    #mmmmm  "mmm"   #     #mm#  #    "  #
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Estimate_Objects_6DOF(self, RGBDimage):
		#1-Extract features
		features, centers , real_centers = self.Extract_Features(RGBDimage = RGBDimage)
		if (not np.any(features)) or (not np.any(centers)) or (not np.any(real_centers)):
			#In case there are no patches extracted
			print 'no features!'
			return [],[]


		#2-KNN search
		results, distances = self.Knn_search(features)
		if not np.any(results):
			print 'no neighboors!'
			return [],[]


		#3-Vote
		self.Cast_Votes(results, distances, real_centers)

		#4-Filter Votes
		self.Votes_filter_1_counter(np.shape(RGBDimage))
		vote_space0 = np.array(self.vote_cells_space)

		self.Votes_filter_1_acumulator()

		local_max_x, local_max_y = self.Votes_filter_2()
		vote_space1 = np.array(self.vote_cells_space)


		position_predictions, rotation_predictions = self.Votes_filter_3(local_max_x, local_max_y)


		return position_predictions, rotation_predictions




	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
					 #mmmmm mmmmmm   mm  mmmmmmm m    m mmmmm  mmmmmm  mmmm        
					 #      #        ##     #    #    # #   "# #      #"   "       
					 #mmmmm #mmmmm  #  #    #    #    # #mmmm" #mmmmm "#mmm    #   
					 #      #       #mm#    #    #    # #   "m #          "#       
					 #      #mmmmm #    #   #    "mmmm" #    " #mmmmm "mmm#"   #
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================

	def Extract_Features(self, RGBDimage):
		#Extract Patchs from Image
		Patches, patch_centers, patch_real = pat.patch_sampling(RGBD = RGBDimage,
																metric = self.patch_metric_size,
																spacing = self.patch_spacing,
																Intrins = self.Intrins,
																Extrins = self.Extrins,
																Workspace = self.Workspace)
		if not np.any(Patches):     #In case there are no patches extracted
			return [],[],[]
		
		#convert patches to features, with CAE_Net()
		Patches = self.CAE_Net.wrap_input(Patches)
		features = self.CAE_Net.encode(Patches)
		features = self.CAE_Net.unwrap_features(features)

		return features, patch_centers, patch_real





	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
			 #    m        mm   m mm   m         mmmm  mmmmmm   mm   mmmmm    mmm  m    m
			 #  m"         #"m  # #"m  #        #"   " #        ##   #   "# m"   " #    #
			 #m#           # #m # # #m #        "#mmm  #mmmmm  #  #  #mmmm" #      #mmmm#  #
			 #  #m         #  # # #  # #            "# #       #mm#  #   "m #      #    #
			 #   "m        #   ## #   ##        "mmm#" #mmmmm #    # #    "  "mmm" #    #  #
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Knn_search(self, features):
		s = 0
		e = self.CodeBook_Read_size

		cdbk = self.CodeBookData[s:e]

		all_Results, all_Dists = self.flann.nn(cdbk['features'].squeeze(), features, self.Neighboors_K)

		#========================
		#only runs if the codebook has more than one part!
		for part in xrange(self.CodeBookParts - 1):
			s=e
			if (e + self.CodeBook_Read_size > self.CodeBookLength): 
				e += self.CodeBookLength % self.CodeBook_Read_size
			else:
				e += chunk_size

			cdbk = self.CodeBookData[s:e]
			Results_tmp, Dists_tmp = self.flann.nn(cdbk['features'].squeeze(), features, self.Neighboors_K)

			all_Results = np.concatenate([all_Results,Results_tmp], axis=1)
			all_Dists = np.concatenate([all_Dists,Dists_tmp], axis=1)
		#Finished reading from codebook
		#========================

		#Sort Results and extract the K best
		Results   = np.zeros([np.shape(all_Results)[0], self.Neighboors_K]).astype(int)
		Distances = np.zeros([np.shape(all_Dists)[0]  , self.Neighboors_K])
		
		for ri, r in enumerate(all_Results):
			chosen = all_Dists[ri].argsort()[:self.Neighboors_K] #indices of the K best Neighboors
			Results[ri] = [r[j] for j in chosen]
			Distances[ri] = [all_Dists[ri][j]**0.5 for j in chosen]

		return Results, Distances






	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	""" 				m    m  mmmm mmmmmmm mmmmm  mm   m   mmm 
						"m  m" m"  "m   #      #    #"m  # m"   "
						 #  #  #    #   #      #    # #m # #   mm  #
						 "mm"  #    #   #      #    #  # # #    #
						  ##    #mm#    #    mm#mm  #   ##  "mmm"  # """
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Cast_Votes(self, results, distances, patch_real_centers):
		self.votes={'Weigth': [], 'RealPosition': [], 'ProjectedPoint': [], 'Orientation': [], 'Obj_id': []}

		for patch, neighboor in zip(*np.where(distances <= self.Neighboors_distance_threshold)):
			reading = self.CodeBookData[results[patch,neighboor]]
			self.votes['RealPosition'].extend(reading['Translation'] + patch_real_centers[patch])
			self.votes['Orientation'].extend(reading['euler_angles'])
			self.votes['Obj_id'].append(reading['obj_ids'])
			#weight = np.exp(-distances[patch,neighboor])                               #from original papper
			#Modificacao
			ratio = self.Neighboors_distance_threshold / np.abs(np.log(0.0001))         #Votos no threshold tem weight 0.001
			weight = np.exp(-distances[patch,neighboor]/ratio)                          #tentativa de dar mais peso a distancia entre features
			self.votes['Weigth'].append(weight)     


		#Project points
		if len(self.votes['RealPosition']) > 0:         
			self.votes['ProjectedPoint'] = pat.from_3D_to_image(points = self.votes['RealPosition'], Intrins = self.Intrins)



	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
					#mmmmm mmmmm  m     mmmmmmm mmmmmm mmmmm         mmm   
					#        #    #        #    #      #   "#          #   
					#mmmmm   #    #        #    #mmmmm #mmmm"          #    #
					#        #    #        #    #      #   "m          #   
					#      mm#mm  #mmmmm   #    #mmmmm #    "        mm#mm  #
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Votes_filter_1_counter(self, img_shape):
		dim = [d/self.cell_size for d in img_shape[:2]]


		self.vote_cells_space = np.zeros(dim)   #cell space
		self.vote_indx_regist = {}              #regists wich votes are in each cells

		#count votes into the cell space
		for vote_indx, projpoint in enumerate(self.votes['ProjectedPoint']):
			cell_indx = [self.round_int(np.floor(k/self.cell_size)) for k in projpoint[:2]] #project votes into cells
		#########################################################################################################################
			cell_indx = cell_indx[::-1]	#invert axes  #!!!!!!!!!!!!! wtf bug....T_T  why??? !!!!!!!!!!!!!!!!!!!!!
		#########################################################################################################################


			if 0 <= cell_indx[0] < dim[0] and 0 <= cell_indx[1] < dim[1]:   #projection is in cell space
				self.vote_cells_space[cell_indx[0],cell_indx[1]] += 1           #count
				if str(cell_indx) not in self.vote_indx_regist.keys():  #creat regist
					self.vote_indx_regist[str(cell_indx)] = []
				self.vote_indx_regist[str(cell_indx)].append(vote_indx) #regist

	def Votes_filter_1_acumulator(self):
		#Filter cell space
		try:
			num_of_votes = np.sum(self.vote_cells_space)
			num_cells_w_votes = np.nonzero(self.vote_cells_space)[0].shape[0]
			vote_threshold = np.ceil(num_of_votes/num_cells_w_votes).astype(int)+1
			if vote_threshold < self.voting_threshold:
				vote_threshold = self.voting_thresholds
		except: 
			vote_threshold = self.voting_threshold


		#convert cell space, from counter to weight acumulator
		for ri, row in enumerate(self.vote_cells_space):
			for rj, value in enumerate(row):
				if value <= vote_threshold: #ignore this cell
					self.vote_cells_space[ri][rj] = 0
					if str([ri,rj]) in self.vote_indx_regist.keys():    #deletes cell registry, if there's any
						del self.vote_indx_regist[str([ri,rj])]
				else:   #acumulate vote weights 
					weight_sum = 0          
					for vote_indx in self.vote_indx_regist[str([ri,rj])]:
						weight_sum = self.votes['Weigth'][vote_indx]

					#tentativa de melhorar resultados
					#self.vote_cells_space[ri][rj] = (1-self.weight_trust) * (2/(1+np.exp((vote_threshold-self.vote_cells_space[ri][rj])))-1)   #Trust in counter
					self.vote_cells_space[ri][rj] = (1-self.weight_trust) * (np.exp(-self.vote_cells_space[ri][rj])*10) 	#Trust in counter
					self.vote_cells_space[ri][rj] += self.weight_trust * weight_sum




		

	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
					#mmmmm mmmmm  m     mmmmmmm mmmmmm mmmmm          mmmm 
					#        #    #        #    #      #   "#        "   "#
					#mmmmm   #    #        #    #mmmmm #mmmm"            m"  #
					#        #    #        #    #      #   "m          m"  
					#      mm#mm  #mmmmm   #    #mmmmm #    "        m#mmmm  #
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Votes_filter_2(self):
		dim = np.shape(self.vote_cells_space)
		x = np.arange(dim[0]).astype(int)
		y = np.arange(dim[1]).astype(int)
			
		#Bilinear Interpolation:
		f = interpolate.RectBivariateSpline(x,y,self.vote_cells_space,s=self.interp_smoothing) 
		self.vote_cells_space_interpolated = f(x,y)

		#Local max extraction
		cells_space_max_filter = filters.maximum_filter(self.vote_cells_space_interpolated, self.localmax_window)
		local_max_x, local_max_y = np.where(self.vote_cells_space_interpolated == cells_space_max_filter)
		return local_max_x, local_max_y




	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
					#mmmmm mmmmm  m     mmmmmmm mmmmmm mmmmm          mmmm 
					#        #    #        #    #      #   "#        "   "#
					#mmmmm   #    #        #    #mmmmm #mmmm"          mmm"		#
					#        #    #        #    #      #   "m            "#
					#      mm#mm  #mmmmm   #    #mmmmm #    "        "mmm#		#
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Votes_filter_3(self, local_max_x, local_max_y):
		position_predictions = []
		rotation_predictions = []

		for ix,iy in zip(local_max_x, local_max_y):
			positions = []
			rotations = []

			#check neighbor cells
			neighboors = np.linspace(-self.Neighboor_Cells, self.Neighboor_Cells, 1+2*self.Neighboor_Cells).astype(int)
			for nx in neighboors:
				for ny in neighboors:
					key = str([ix+nx, iy+ny])
					if key in self.vote_indx_regist.keys():
						for vote_indx in self.vote_indx_regist[key]:
							positions.append(self.votes['RealPosition'][vote_indx])
							rotations.append(self.votes['Orientation'][vote_indx])

			#Perform Meanshift
			if len(positions)>0:
				msTranslation = MeanShift(bandwidth = self.Translation_Kernel, bin_seeding=False)
				msTranslation.fit(positions)
				position_predictions.extend(msTranslation.cluster_centers_)

			if len(rotations)>0:
				msOrientation = MeanShift(bandwidth = self.Orientation_Kernel, bin_seeding=False)
				msOrientation.fit(rotations)
				rotation_predictions.extend(msOrientation.cluster_centers_)

		return position_predictions, rotation_predictions
