#! /usr/bin/env python
#===============IMPORTS======================
import tables
import pyflann
import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters
from sklearn.cluster import MeanShift, estimate_bandwidth

#===============my modules====================
from my_libs import Patcher as pat
from my_libs import CAE_net as conv

#===============Predictors====================
from my_libs import predictor_0 as p0
from my_libs import predictor_1 as p1
from my_libs import predictor_2 as p2
from my_libs import predictor_3 as p3



class Detector():
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

		#Predictor
		self.use_Predictor	= 0

		#Patches caracteristics:
		self.patch_metric_size = 50             #5 cm
		self.patch_spacing = 5                  #5 pixels between patch centers


		#Voting parameters:
		self.cell_size = 5              #5x5 pixels
		self.Vote_Space = []

		self.Vote_space_minimum = 0
		self.Vote_space_maximum = 1000

		#Filter parameters:
		#Local Max
		self.interp_smoothing = 0
		self.localmax_window = 15   #5x5 cells
		self.localmax_thresh = 0.1

		#fMeanShift
		self.Neighboor_Cells = 2            #check Votes in the Neighboors cells
		self.Translation_Kernel = 100       #25                 #2.5 cm
		self.Orientation_Kernel = (2*np.pi)*(7/360.)       #((2*3.1416)/360)*7 #7 degrees


	#=======================================================================
	#-----------------------------------methods:----------------------------
	#=======================================================================
	def initialize(self):
		self.initialize_CAE()
		self.initialize_Predictor(predictor = self.use_Predictor)

	def initialize_CAE(self):
		self.CAE_Net = conv.CAE()
		self.CAE_Net.initialize(self.ConvolutionalAutoEncoder)


	def initialize_Predictor(self, predictor = 0):
		Predictor_List = {0: p0.Predictor(), 1: p1.Predictor(), 2: p2.Predictor(), 3: p3.Predictor()}
		Name = ['Predictor_0','Predictor_1','Predictor_2','Predictor_3']
		self.Predictor = Predictor_List[predictor]
		self.Predictor.initialize(Name[predictor]+'.pt')


	def initialize_VoteSpace(self, image_shape):
		dim = [d/self.cell_size for d in image_shape]
		self.Vote_Space = np.ones(dim) * 0.5

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
		#0- Create Vote Space:
		if self.Vote_Space == []:
			self.initialize_VoteSpace(image_shape = np.shape(RGBDimage)[:2])

		#1-Extract features
		features, centers , real_centers = self.Extract_Features(RGBDimage = RGBDimage)
		if (not np.any(features)) or (not np.any(centers)) or (not np.any(real_centers)):
			#In case there are no patches extracted
			print 'no features!'
			return [],[]

		#2- Make Predictions
		Position, Orientation = self.Predict(features, real_centers)

		#3- Cast Votes and Update VoteSpace
		self.Cast_Votes(Position, Orientation)
		self.Update_VoteSpace()

		#4- Filter VoteSpace local Maxima
		X_max, Y_max = self.Votes_filter_local_Max()

		#5- Filter VoteRegist of local max
		position_predict, rotation_predic = self.Votes_Regist_filter(X_max, Y_max)

		return position_predict, rotation_predic




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
					 #mmmm  mmmmm  mmmmmm mmmm   mmmmm    mmm mmmmmmm       
					 #   "# #   "# #      #   "m   #    m"   "   #          
					 #mmm#" #mmmm" #mmmmm #    #   #    #        #      #   
					 #      #   "m #      #    #   #    #        #          
					 #      #    " #mmmmm #mmm"  mm#mm   "mmm"   #      # 
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Predict(self, features, real_centers):
		#Feed features to predictor
		features = self.Predictor.wrap_input(features)
		predictions = self.Predictor(features)
		predictions = self.Predictor.unwrap_variable(predictions)

		Position = real_centers + predictions[:,:3]
		Orientation = predictions[:,3:]

		return Position, Orientation





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
	def Cast_Votes(self, Positions, Orientations):
		self.votes_regist = {'Position': [], 'Orientation': [], 'Projection': []}

		self.votes_regist['Position'] = Positions
		self.votes_regist['Orientation'] = Orientations
		self.votes_regist['Projection'] = pat.from_3D_to_image(points = Positions, Intrins = self.Intrins)[:,:2]


	def Update_VoteSpace(self):
		self.votes_in_cell = {}              #regists wich votes are in each cells

		shape = np.shape(self.Vote_Space)
		Counter = np.zeros(shape).astype(float)
		Readings = np.ones(shape)*0

		#get cell index of each projection
		cell_indx = np.floor(self.votes_regist['Projection']/self.cell_size).astype(int)

		# possible bug from pat.from3d_to_img()
		cell_indx = cell_indx [:,::-1]		# switch cell indexs to resolv

		for vi, c in enumerate(cell_indx):								#For each vote cell index
			if (0 <= c[0] <= shape[0]) and (0 <= c[1] <= shape[1]):		#	Check if its inside votespace
				Counter[c[0],c[1]] += 1									#		increment counter value
				if str([c[0],c[1]]) not in self.votes_in_cell.keys():  			#		check if this cell is registed
					self.votes_in_cell[str([c[0],c[1]])] = []						#			create regist if not
				self.votes_in_cell[str([c[0],c[1]])].append(vi) 					#		regist Vote index to cell


		#Normalize Counter
		if not self.votes_in_cell == {}:
			Counter = Counter / np.sum(Counter)
		i = np.nonzero(Counter)

		Counter = (1 + Counter) / 2
		Readings[i] = Counter[i]

		#Convert to log odd
		Readings = np.log( Readings / (1-Readings))
		
		self.Vote_Space += Readings
		self.Vote_Space[np.where(self.Vote_Space < self.Vote_space_minimum)] = self.Vote_space_minimum
		self.Vote_Space[np.where(self.Vote_Space < self.Vote_space_maximum)] = self.Vote_space_maximum








	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
					#mmmmm mmmmm  m     mmmmmmm mmmmmm mmmmm 
					#        #    #        #    #      #   "#
					#mmmmm   #    #        #    #mmmmm #mmmm" 	#
					#        #    #        #    #      #   "m
					#      mm#mm  #mmmmm   #    #mmmmm #    "   #
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
	def Votes_filter_local_Max(self):
		dim = np.shape(self.Vote_Space)
		x = np.arange(dim[0]).astype(int)
		y = np.arange(dim[1]).astype(int)
			
		#Bilinear Interpolation:
		f = interpolate.RectBivariateSpline(x,y,self.Vote_Space, s = self.interp_smoothing) 
		interp_space = f(x,y)

		#Local max extraction
		data_max = filters.maximum_filter(interp_space, self.localmax_window)
		data_min = filters.minimum_filter(interp_space, self.localmax_window)

		maxima = (interp_space == data_max)
		diff = ((data_max - data_min) > self.localmax_thresh)
		maxima[diff == 0] = 0

		labeled, num_objects = ndimage.label(maxima)
		slices = ndimage.find_objects(labeled)
		local_max_x, local_max_y = [], []
		for dy,dx in slices:
			x_center = (dx.start + dx.stop - 1)/2
			y_center = (dy.start + dy.stop - 1)/2    
			local_max_x.append(int(x_center))
			local_max_y.append(int(y_center))


		return local_max_y, local_max_x		#TROCADO!! porcausa dos eixos no scatter





	def Votes_Regist_filter(self, local_max_x, local_max_y):
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
					if key in self.votes_in_cell.keys():
						for vote_indx in self.votes_in_cell[key]:
							positions.append(self.votes_regist['Position'][vote_indx])
							rotations.append(self.votes_regist['Orientation'][vote_indx])


			#Perform Meanshift
			if len(positions)>0:
				msTranslation = MeanShift(bandwidth = self.Translation_Kernel, bin_seeding=False)
				msTranslation.fit(positions)
				position_predictions.extend(msTranslation.cluster_centers_)

			if len(rotations)>0:
				msOrientation = MeanShift(bandwidth = self.Orientation_Kernel, bin_seeding=False)
				msOrientation.fit(rotations)
				rotation_predictions.extend(msOrientation.cluster_centers_)
#-----
		if len(position_predictions) > 0:
			msTranslation = MeanShift(bandwidth = self.Translation_Kernel, bin_seeding=False)
			msTranslation.fit(position_predictions)
			position_predictions = msTranslation.cluster_centers_



		return position_predictions, rotation_predictions





	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
							#mmmmm m    mmmmmmmm mmmmm    mm         
							#       #  #    #    #   "#   ##         
							#mmmmm   ##     #    #mmmm"  #  #    #   
							#       m""m    #    #   "m  #mm#        
							#mmmmm m"  "m   #    #    " #    #   #
	#=========================================================================================
	#-----------------------------------------------------------------------------------------
	#=========================================================================================
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
