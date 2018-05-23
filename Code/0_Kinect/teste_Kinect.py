#! /usr/bin/env python
#================================   IMPORTS   ==================================
import matplotlib.pyplot as plt
import numpy as np
import cv2
#===============my modules====================
from my_libs import Kinect_class as Kin


#===============================================================
#------------------------ Configs ------------------------------
#===============================================================
print_time = 30	#seconds


#===============================================================
#------------------------ Kinect ------------------------------
#===============================================================
Kinect = Kin.Kinect_object()


#wait for kinect to be ready
print 'Waiting for kinect..'
while np.shape(Kinect.imgRGB)[0]<10:
	continue



since = time.time()
while(1):
	cv2.imshow('bgr', Kinect.imgRGB)
	cv2.waitKey(5)
	if (time.time() - since) > print_time:
		opt = raw_input('Enter to continue, q/Q to qui:\n::')
		if opt in ['q','Q']:
			break
		else:
			since = time.time()













