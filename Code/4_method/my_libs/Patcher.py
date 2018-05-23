#! /usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import cv2

#returns the centers of each patch
def points_of_grid(img, spacing=10):
	grid=[]
	dim=np.shape(img)
	num_points=np.divide(dim,spacing)
	#pontos da grelha sem os primeiros e os ultimos
	px = np.linspace(0,dim[0],num_points[0],dtype=int)
	py = np.linspace(0,dim[1],num_points[1],dtype=int)

	#elimina o primeiro e o ultimo ponto
	px=np.delete(px , [0, num_points[0]-1])
	py=np.delete(py , [0, num_points[1]-1])

	grid=[]
	for x in px:
		for y in py:
			if np.any(img[x,y])  & (not np.isnan(img[x,y])):	#se a profundidade for valida
				grid.append([x,y])

	return np.array(grid)


#converts points from a rgbD img to 3D world
def from_image_to_3D(depth_img, points=[], Intrins = []):
	#Camera Intrinsics: assume Kinect's if none given
	if not np.any(Intrins):
		fx, fy, cx, cy = 572.41140, 573.57043, 325.26110, 242.04899
		Intrins = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

	#if no given points, use all points of img
	if not np.any(points):
		points = np.array(list(np.ndindex(np.shape(depth_img))))


	inv_Intrins = np.linalg.inv(Intrins)


	#Get each point depth value
	depths = [depth_img[p[0],p[1]] for p in points]
	
	colums = np.transpose(points)
	colums = np.vstack([colums, np.ones(len(colums[0]))])

#########################################################################################################################
	colums[[0,1]]=colums[[1,0]]		#!!!!!!!!!!!!! wtf bug....T_T  why??? !!!!!!!!!!!!!!!!!!!!!
#########################################################################################################################

	#Project pixels to world
	normalized_points = np.dot(inv_Intrins,colums)
	real_points = np.multiply(normalized_points, np.squeeze(depths))

	return (np.transpose(real_points))


#convert points from Camera Axis to World Axis
def from_CAM_to_WS(points_on_CAM, CAM_on_WS=[]):
	#Project grid_real_points to Work Space
	points_on_CAM = np.transpose(points_on_CAM)
	points_on_CAM = np.vstack([points_on_CAM, np.ones(len(points_on_CAM[0]))])
	points_on_WS = np.dot(CAM_on_WS, points_on_CAM)

	return(np.transpose(points_on_WS[0:3]))


#Check if points are inside the WorkSpace
def in_Workspace(points_on_WS, WorkSpace=[]):
	if not np.any(WorkSpace):
		#Default: [-10;10]meters, for all axis
		WorkSpace = [[-10000, 10000],	# x coordinate		[Down, Up]
					[-10000, 10000], 	# y coordinate		[Left, Right]
					[-10000, 10000]]	# z coordinate		[Back, Front]

	points_on_WS = np.transpose(points_on_WS)
	valid_x = (points_on_WS[0] > WorkSpace[0][0]) & (points_on_WS[0] < WorkSpace[0][1])
	valid_y = (points_on_WS[1] > WorkSpace[1][0]) & (points_on_WS[1] < WorkSpace[1][1])
	valid_z = (points_on_WS[2] > WorkSpace[2][0]) & (points_on_WS[2] < WorkSpace[2][1])

	valid_indices = np.where(valid_x & valid_y & valid_z)

	return(np.squeeze(valid_indices))


#Chack if Patch is inside the image
def check_fit(dim,center,size):
	if(center[0]>=size[0]) and (center[1]>=size[1]):	#lim inferior do size
		lim=np.subtract(dim,center)
		if(lim[0]>=size[0]) and (lim[1]>=size[1]):			#lim superior do size
			return True
	return False


#Extract normalized 32x32 Patches from image, given the centers
def Patch_extractor(patch_centers, RGB, Depth, metric=50, focal_length=[]):
	if not np.any(focal_length):
		#default: Kinects aprox focal length
		focal_length = np.array([[580], [580]])

	dim = np.shape(Depth)
	depths = np.array([[Depth[p[0],p[1]] for p in patch_centers]])	

	patch_size = metric/(depths*2.0)
	patch_size = np.dot(focal_length, patch_size).astype(int)

	patch_size = np.transpose(patch_size)
	depths = np.transpose(depths)


	Patches=[]
	invalid=[]
	for i, p in enumerate(patch_centers):
		if check_fit(dim, p, patch_size[i]):
			#extract RGB and Depth patch
			rgb_tmp   = RGB[(p[0] - patch_size[i][0]) : (p[0] + patch_size[i][0] +1),
							(p[1] - patch_size[i][1]) : (p[1] + patch_size[i][1] +1), :].astype(float)

			depth_tmp = Depth[(p[0] - patch_size[i][0]) : (p[0] + patch_size[i][0] +1),
							  (p[1] - patch_size[i][1]) : (p[1] + patch_size[i][1] +1)].astype(float)
		
			#normalize Depth patch
			depth_tmp=np.clip(np.subtract(depth_tmp,depths[i]),-metric,+metric)
			depth_tmp=cv2.normalize(depth_tmp,-1,1,norm_type=cv2.NORM_MINMAX)
			depth_tmp=cv2.resize(depth_tmp,(32,32))

			#normalize RGB patch
			rgb_tmp=cv2.normalize(rgb_tmp,-1,1,norm_type=cv2.NORM_MINMAX)
			rgb_tmp=cv2.resize(rgb_tmp,(32,32))

			#Concatenate
			depth_tmp = np.expand_dims(depth_tmp, axis=2)
			PATCH = np.concatenate((rgb_tmp,depth_tmp), axis=2)

			Patches.append(PATCH)
		else:
			invalid.append(i)	

	return(Patches, np.array(invalid))



# Work Horse: Extract Patches from an RGBD image
def patch_sampling(RGBD, metric=50, spacing=10,  Intrins=[], Extrins=[], Workspace=[]):
	RGB = RGBD[:,:,0:3]
	Depth = RGBD[:,:,3]

	#Sampling candidates
	grid = points_of_grid(Depth, spacing)
	real_points = from_image_to_3D(Depth, grid, Intrins)

	#Convert points from camera axis to world axis
	if not Extrins == []:
		points_on_WS = from_CAM_to_WS(points_on_CAM = real_points
									 ,CAM_on_WS = Extrins)
	else:
		points_on_WS = real_points

	#Filter Sampling candidates out of the Workspace
	if not Workspace == []:	
		valid_points = in_Workspace(points_on_WS, Workspace)
		valid_grid = grid[valid_points]
		valid_grid_real = real_points[valid_points]
	else:
		valid_grid = grid
		valid_grid_real = real_points

	#Extract patches
	Patches, invalid_points = Patch_extractor(valid_grid, RGB, Depth, metric=50)

	#Check if there were invalid points
	if not invalid_points == []:
		valid_grid = np.delete(valid_grid, invalid_points, 0)
		valid_grid_real = np.delete(valid_grid_real, invalid_points, 0)


	return Patches, valid_grid, valid_grid_real






#===============================================================
#----------------- Visualize Results ---------------------------
#===============================================================
def Visualize_Patches(axes=[] , rgb_img=[], Patches=[], patch_centers=[]):
	axes[0].imshow(rgb_img)
	sr = axes[0].scatter(patch_centers[:,1], patch_centers[:,0] , color='red', s=0.5)
	plt.pause(0.1)
	for p, patch in enumerate(Patches):
		sb = axes[0].scatter(patch_centers[p,1], patch_centers[p,0], color='blue', s=10)
		#print depth[patch_centers[p,0], patch_centers[p,1]]
		axes[1].imshow(patch)
		plt.pause(0.1)
		#raw_input('')
		sb.remove()
		axes[1].clear()
	axes[0].clear()
	axes[1].clear()






#===============================================================
#auxiliar
def from_3D_to_image(points, Intrins=[]):
	if not np.any(Intrins):
		fx, fy, cx, cy = 572.41140, 573.57043, 325.26110, 242.04899
		Intrins=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
	
	projection = np.dot(points, Intrins.transpose())
	projection = (projection.transpose()/projection[:,2]).transpose()
	return projection
